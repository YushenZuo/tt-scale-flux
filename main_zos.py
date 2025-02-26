import os
import json
from datetime import datetime

import numpy as np
import torch
from diffusers import DiffusionPipeline
from tqdm.auto import tqdm
import copy

from utils import prompt_to_filename, get_noises, TORCH_DTYPE_MAP, get_latent_prep_fn, parse_cli_args, MODEL_NAME_MAP, get_candidates_zero_order_search

# Non-configurable constants
TOPK = 1  # Always selecting the top-1 noise for the next round
MAX_SEED = np.iinfo(np.int32).max  # To generate random seeds


def sample(
    pivot_noises: dict[int, torch.Tensor],
    prompt: str,
    search_round: int,
    pipe: DiffusionPipeline,
    verifier,
    topk: int,
    root_dir: str,
    config: dict,
) -> dict:
    """
    For a given prompt, generate images using candidates in the neighborhood of the pivot noise,
    score them with the verifier, and select the best candidate.
    The images and JSON artifacts are saved under `root_dir`.
    """
    config_cp = copy.deepcopy(config)
    max_new_tokens = config_cp.pop("max_new_tokens", None)
    choice_of_metric = config_cp.pop("choice_of_metric", None)
    verifier_to_use = config_cp.pop("verifier_to_use", "gemini")
    use_low_gpu_vram = config_cp.pop("use_low_gpu_vram", False)
    batch_size_for_img_gen = config_cp.pop("batch_size_for_img_gen", 1)

    images_for_prompt = []
    noises_used = []
    idx_used = []
    prompt_filename = prompt_to_filename(prompt)

    # Convert the noises dictionary into a list of (idx, noise) tuples.
    noise_items = list(pivot_noises.items())
    # print('noise_items: ', noise_items)

    # Process the noises in batches.
    for i in range(0, len(noise_items), batch_size_for_img_gen):
        batch = noise_items[i : i + batch_size_for_img_gen]
        idx_batch, noises_batch = zip(*batch)
        filenames_batch = [
            os.path.join(root_dir, f"{prompt_filename}_i@{search_round}_s@{idx}.png") for idx in idx_batch
        ]

        if use_low_gpu_vram and verifier_to_use != "gemini":
            pipe = pipe.to("cuda:0")
        print(f"Generating images for batch with idx: {[s for s in idx_batch]}.")

        # Create a batched prompt list and stack the latents.
        batched_prompts = [prompt] * len(noises_batch)
        batched_latents = torch.stack(noises_batch).squeeze(dim=1)

        batch_result = pipe(prompt=batched_prompts, latents=batched_latents, **config_cp)
        batch_images = batch_result.images
        if use_low_gpu_vram and verifier_to_use != "gemini":
            pipe = pipe.to("cpu")

        # Iterate over the batch and save the images.
        for idx, noise, image, filename in zip(idx_batch, noises_batch, batch_images, filenames_batch):
            images_for_prompt.append(image)
            noises_used.append(noise)
            idx_used.append(idx)
            image.save(filename)

    # Prepare verifier inputs and perform inference.
    verifier_inputs = verifier.prepare_inputs(images=images_for_prompt, prompts=[prompt] * len(images_for_prompt))
    print("Scoring with the verifier.")
    outputs = verifier.score(
        inputs=verifier_inputs,
        max_new_tokens=max_new_tokens,  # Ignored when using Gemini for now.
    )
    for o in outputs:
        assert choice_of_metric in o, o.keys()

    assert len(outputs) == len(images_for_prompt), (
        f"Expected len(outputs) to be same as len(images_for_prompt) but got {len(outputs)=} & {len(images_for_prompt)=}"
    )

    results = []
    for json_dict, idx_val, noise in zip(outputs, idx_used, noises_used):
        # Attach the noise tensor so we can select top-K.
        merged = {**json_dict, "noise": noise, "idx": idx_val}
        results.append(merged)

    # Sort by the chosen metric descending and pick top-K.
    for x in results:
        assert choice_of_metric in x, (
            f"Expected all dicts in `results` to contain the `{choice_of_metric}` key; got {x.keys()}."
        )

    def f(x):
        if isinstance(x[choice_of_metric], dict):
            return x[choice_of_metric]["score"]
        return x[choice_of_metric]

    sorted_list = sorted(results, key=lambda x: f(x), reverse=True)
    topk_scores = sorted_list[:topk]

    # Print debug information.
    for ts in topk_scores:
        print(f"Prompt='{prompt}' | Best seed={ts['idx']} | Score={ts[choice_of_metric]}")

    best_img_path = os.path.join(root_dir, f"{prompt_filename}_i@{search_round}_s@{topk_scores[0]['idx']}.png")
    datapoint = {
        "prompt": prompt,
        "search_round": search_round,
        "num_noises": len(pivot_noises),
        "best_noise_idx": topk_scores[0]["idx"],
        "best_score": topk_scores[0][choice_of_metric],
        "choice_of_metric": choice_of_metric,
        "best_img_path": best_img_path,
    }
    # Save the best config JSON file alongside the images.
    best_json_filename = best_img_path.replace(".png", ".json")
    with open(best_json_filename, "w") as f:
        json.dump(datapoint, f, indent=4)
    return datapoint, topk_scores[0]["noise"]


@torch.no_grad()
def main_zos():
    """
    Main function:
      - Parses CLI arguments.
      - Creates an output directory based on verifier and current datetime.
      - Loads prompts.
      - Loads the image-generation pipeline.
      - Loads the verifier model.
      - Runs several search rounds where for each prompt a pool of random noises is generated,
        candidate images are produced and verified, and the best noise is chosen.
    """
    args = parse_cli_args()

    # Build a config dictionary for parameters that need to be passed around.
    config = {
        "max_new_tokens": args.max_new_tokens,
        "use_low_gpu_vram": args.use_low_gpu_vram,
        "choice_of_metric": args.choice_of_metric,
        "verifier_to_use": args.verifier_to_use,
        "batch_size_for_img_gen": args.batch_size_for_img_gen,
        # For zero-shot search
        # "n_candidates": args.n_candidates,
        # "d_metric": args.d_metric,
        # "lambda_val": args.lambda_val,
    }
    with open(args.pipeline_config_path, "r") as f:
        config.update(json.load(f))

    search_rounds = args.search_rounds
    num_prompts = args.num_prompts
    # For zero-shot search
    n_candidates = args.n_candidates
    d_metric = args.d_metric
    lambda_val = args.lambda_val
    lambda_dec_round = args.lambda_dec_round

    # Create a root output directory: output/{verifier_to_use}/{current_datetime}
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_name = config.pop("pretrained_model_name_or_path")
    root_dir = os.path.join(
        "output_zos",
        MODEL_NAME_MAP[pipeline_name],
        config["verifier_to_use"],
        config["choice_of_metric"],
        current_datetime,
    )
    os.makedirs(root_dir, exist_ok=True)
    print(f"Artifacts will be saved to: {root_dir}")
    with open(os.path.join(root_dir, "config.json"), "w") as f:
        config_cp = copy.deepcopy(config)
        config_cp.update(vars(args))
        json.dump(config_cp, f)

    # Load prompts from file.
    if args.prompt is None:
        with open("prompts_open_image_pref_v1.txt", "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        if num_prompts != "all":
            prompts = prompts[:num_prompts]
        print(f"Using {len(prompts)} prompt(s).")
    else:
        prompts = [args.prompt]

    # Set up the image-generation pipeline (on the first GPU if available).
    torch_dtype = TORCH_DTYPE_MAP[config.pop("torch_dtype")]
    pipe = DiffusionPipeline.from_pretrained(pipeline_name, torch_dtype=torch_dtype)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    # Load the verifier model.
    if config["verifier_to_use"] == "gemini":
        from verifiers.gemini_verifier import GeminiVerifier

        verifier = GeminiVerifier()
    else:
        from verifiers.qwen_verifier import QwenVerifier

        verifier = QwenVerifier(use_low_gpu_vram=config["use_low_gpu_vram"])

    # Main loop: For each search round and each prompt, generate images, verify, and save artifacts.
    pivot_noise = None
    for round in range(1, search_rounds + 1):
        print(f"\n=== Round: {round} ===")
        for prompt in tqdm(prompts, desc="Sampling prompts"):
            pivot_noises = get_candidates_zero_order_search(
                search_round=round,
                pivot=pivot_noise,
                distance_metric=d_metric,
                lambda_val=lambda_val,
                lambda_dec_round=lambda_dec_round,
                num_candidates=n_candidates,
                max_seed=MAX_SEED,
                height=config["height"],
                width=config["width"],
                dtype=torch_dtype,
                fn=get_latent_prep_fn(pipeline_name),
            )
            print(f"Number of noise samples: {len(pivot_noises)}")
            datapoint_for_current_round, pivot_noise = sample(
                pivot_noises=pivot_noises,
                prompt=prompt,
                search_round=round,
                pipe=pipe,
                verifier=verifier,
                topk=TOPK,
                root_dir=root_dir,
                config=config,
            )


if __name__ == "__main__":
    main_zos()
