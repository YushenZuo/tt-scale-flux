import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers import FluxPipeline
import re
import hashlib
from typing import Dict
import json
from typing import Union
from PIL import Image
import requests
import argparse
import io


TORCH_DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
MODEL_NAME_MAP = {
    "black-forest-labs/FLUX.1-dev": "flux.1-dev",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": "pixart-sigma-1024-ms",
    "stabilityai/stable-diffusion-xl-base-1.0": "sdxl-base",
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "sd-v1.5",
}


def parse_cli_args():
    """
    Parse and return CLI arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_config_path",
        type=str,
        default="configs/flux.1_dev.json",
        help="Pipeline configuration path that should include loading info and __call__() args and their values.",
    )
    parser.add_argument(
        "--search_rounds",
        type=int,
        default=4,
        help="Number of search rounds (each round scales the number of noise samples).",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Use your own prompt.")
    parser.add_argument(
        "--num_prompts",
        type=lambda x: None if x.lower() == "none" else x if x.lower() == "all" else int(x),
        default=2,
        help="Number of prompts to use (or 'all' to use all prompts from file).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=800,
        help="Maximum number of tokens for the verifier. Ignored when using Gemini.",
    )
    parser.add_argument(
        "--batch_size_for_img_gen",
        type=int,
        default=1,
        help="Controls the batch size of noises during image generation. Increasing it reduces the total time at the cost of more memory.",
    )
    parser.add_argument(
        "--use_low_gpu_vram",
        action="store_true",
        help="Flag to use low GPU VRAM mode (moves models between cpu and cuda as needed). Ignored when using Gemini.",
    )
    parser.add_argument(
        "--choice_of_metric",
        type=str,
        default="overall_score",
        choices=[
            "accuracy_to_prompt",
            "creativity_and_originality",
            "visual_quality_and_realism",
            "consistency_and_cohesion",
            "emotional_or_thematic_resonance",
            "overall_score",
        ],
        help="Metric to use from the LLM grading. When implementing something custom, feel free to relax these.",
    )
    parser.add_argument(
        "--verifier_to_use",
        type=str,
        default="gemini",
        choices=["gemini", "qwen"],
        help="Verifier to use; must be one of 'gemini' or 'qwen'.",
    )
    # Zero-order search
    parser.add_argument(
        "--n_candidates",
        type=int,
        default=4,
        help="The number of candidates in zero-order search",
    )
    parser.add_argument(
        "--d_metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan", "cosine", "mahalanobis"],
        help="Distance metric when getting candidates based on pivot in zero-order search",
    )
    parser.add_argument(
        "--lambda_val",
        type=float,
        default=0.1,
        help="Lambda value when getting candidates based on pivot in zero-order search",
    )
    parser.add_argument(
        "--lambda_dec_round",
        type=float,
        default=0.0,
        help="Lambda decrease factor via round when calculating lambda in zero-order search",
    )
    args = parser.parse_args()

    if args.prompt and args.num_prompts:
        raise ValueError("Both `prompt` and `num_prompts` cannot be specified.")
    if not args.prompt and not args.num_prompts:
        raise ValueError("Both `prompt` and `num_prompts` cannot be None.")
    return args


# Adapted from Diffusers.
def prepare_latents_for_flux(
    batch_size: int,
    height: int,
    width: int,
    generator: torch.Generator,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_latent_channels = 16
    vae_scale_factor = 8

    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    shape = (batch_size, num_latent_channels, height, width)
    latents = randn_tensor(shape, generator=generator, device=torch.device(device), dtype=dtype)
    latents = FluxPipeline._pack_latents(latents, batch_size, num_latent_channels, height, width)
    return latents


# Adapted from Diffusers.
def prepare_latents(
    batch_size: int, height: int, width: int, generator: torch.Generator, device: str, dtype: torch.dtype
):
    num_channels_latents = 4
    vae_scale_factor = 8
    shape = (
        batch_size,
        num_channels_latents,
        int(height) // vae_scale_factor,
        int(width) // vae_scale_factor,
    )
    latents = randn_tensor(shape, generator=generator, device=torch.device(device), dtype=dtype)
    return latents


def get_latent_prep_fn(pretrained_model_name_or_path: str) -> callable:
    fn_map = {
        "black-forest-labs/FLUX.1-dev": prepare_latents_for_flux,
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": prepare_latents,
        "stabilityai/stable-diffusion-xl-base-1.0": prepare_latents,
        "stable-diffusion-v1-5/stable-diffusion-v1-5": prepare_latents,
    }[pretrained_model_name_or_path]
    return fn_map


def get_candidates_zero_order_search(
    search_round: int,
    pivot: torch.Tensor,
    distance_metric: str,
    lambda_val: float,
    lambda_dec_round: float,
    num_candidates: int,
    max_seed: int,
    height: int,
    width: int,
    device="cuda",
    dtype: torch.dtype = torch.bfloat16,
    fn: callable = prepare_latents_for_flux,
) -> Dict[int, torch.Tensor]:
    if search_round == 1 or pivot is None:
        pivot_dict = get_noises(max_seed=max_seed,
                num_samples=1,
                height=height,
                width=width,
                device=device,
                dtype=dtype,
                fn=fn,)
    
        pivot = list(pivot_dict.values())[0]

    # print('lambda_val: ', lambda_val)
    # print('lambda_val: ', lambda_val.dtype)
    if lambda_val is None or lambda_val == 'None':
        # print('run this function')
        lambda_val = get_lambda_val(distance_metric, pivot)
    
    if 0.0 < lambda_dec_round < 1.0:
        lambda_val = lambda_val * (lambda_dec_round ** (search_round - 1))
    
    print('lambda value in this round: ', lambda_val)

    candidates = {}
    if distance_metric == 'euclidean':
        candidates = sample_euclidean(pivot, num_candidates, lambda_val)
    elif distance_metric == 'manhattan':
        candidates = sample_manhattan(pivot, num_candidates, lambda_val)
    elif distance_metric == 'cosine':
        candidates = sample_cosine(pivot, num_candidates, lambda_val)
    elif distance_metric == 'mahalanobis':
        cov_inv = torch.eye(pivot.shape)  
        candidates = sample_mahalanobis(pivot, num_candidates, lambda_val, cov_inv)
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    assert len(candidates) == num_candidates

    if search_round == 1:
        candidates.update({num_candidates + 1: pivot})

    return candidates


def get_noises(
    max_seed: int,
    num_samples: int,
    height: int,
    width: int,
    device="cuda",
    dtype: torch.dtype = torch.bfloat16,
    fn: callable = prepare_latents_for_flux,
) -> Dict[int, torch.Tensor]:
    seeds = torch.randint(0, high=max_seed, size=(num_samples,))

    noises = {}
    for noise_seed in seeds:
        latents = fn(
            batch_size=1,
            height=height,
            width=width,
            generator=torch.manual_seed(int(noise_seed)),
            device=device,
            dtype=dtype,
        )
        noises.update({int(noise_seed): latents})

    assert len(noises) == len(seeds)
    return noises


def load_verifier_prompt(path: str) -> str:
    with open(path, "r") as f:
        verifier_prompt = f.read().replace('"""', "")

    return verifier_prompt


def prompt_to_filename(prompt, max_length=100):
    """Thanks ChatGPT."""
    filename = re.sub(r"[^a-zA-Z0-9]", "_", prompt.strip())
    filename = re.sub(r"_+", "_", filename)
    hash_digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    base_filename = f"prompt@{filename}_hash@{hash_digest}"

    if len(base_filename) > max_length:
        base_length = max_length - len(hash_digest) - 7
        base_filename = f"prompt@{filename[:base_length]}_hash@{hash_digest}"

    return base_filename


def load_image(path_or_url: Union[str, Image.Image]) -> Image.Image:
    """
    Load an image from a local path or a URL and return a PIL Image object.

    `path_or_url` is returned as is if it's an `Image` already.
    """
    if isinstance(path_or_url, Image.Image):
        return path_or_url
    elif path_or_url.startswith("http"):
        response = requests.get(path_or_url, stream=True)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    return Image.open(path_or_url)


def convert_to_bytes(path_or_url: Union[str, Image.Image]) -> bytes:
    """Load an image from a path or URL and convert it to bytes."""
    image = load_image(path_or_url).convert("RGB")
    image_bytes_io = io.BytesIO()
    image.save(image_bytes_io, format="PNG")
    return image_bytes_io.getvalue()


def recover_json_from_output(output: str):
    start = output.find("{")
    end = output.rfind("}") + 1
    json_part = output[start:end]
    return json.loads(json_part)





def sample_euclidean(pivot, N, lambda_val):
    """Generate N candidates at an exact Euclidean distance lambda_val from the pivot."""
    dim = pivot.shape
    # directions = torch.randn((N,) + dim, device=pivot.device, dtype=pivot.dtype)  
    # directions = directions / torch.norm(directions, dim=(2, 3), keepdim=True)
    candidates = {}
    for i in range(N):
        direction = torch.randn(dim, device=pivot.device, dtype=pivot.dtype)
        direction = direction / torch.norm(direction)
        candidates.update({int(i + 1): pivot + lambda_val * direction})
    # candidates.update({int(N + 1): pivot})
    return candidates


def sample_manhattan(pivot, N, lambda_val):
    """Generate N candidates at an exact Manhattan (L1) distance lambda_val from the pivot."""
    dim = pivot.shape
    candidates = {}
    for i in range(N):
        signs = torch.randint(0, 2, dim) * 2 - 1  
        proportions = torch.rand(dim).to(pivot.device)
        proportions = proportions / proportions.sum()  
        perturbation = lambda_val * signs * proportions  
        # candidates[i + 1] = pivot + perturbation
        candidates.update({int(i + 1): pivot + perturbation})
    # candidates.update({int(N + 1): pivot})  
    return candidates


def sample_cosine(pivot, N, lambda_val):
    """Generate N candidates at an exact cosine distance lambda_val from the pivot."""
    dim = pivot.shape
    pivot_norm = torch.norm(pivot, p=2, dim=tuple(range(pivot.dim())), keepdim=True)
    
    directions = torch.randn((N,) + dim).to(pivot.device)
    dot_product = (directions * pivot).sum(dim=tuple(range(1, directions.dim())), keepdim=True)
    directions = directions - dot_product * pivot / (pivot_norm**2)  
    directions = directions / torch.norm(directions, dim=tuple(range(1, directions.dim())), keepdim=True)  
    
    theta = torch.acos(1 - lambda_val)  
    new_points = torch.cos(theta) * pivot.unsqueeze(0) + torch.sin(theta) * pivot_norm * directions
    
    return {i + 1: new_points[i] for i in range(N)}


def sample_mahalanobis(pivot, N, lambda_val, cov_inv):
    """Generate N candidates at an exact Mahalanobis distance lambda_val from the pivot."""
    dim = pivot.shape
    chol = torch.linalg.cholesky(torch.linalg.inv(cov_inv)).T
    chol = chol.to(pivot.device)
    standard_gaussian_samples = torch.randn((N,) + dim).to(pivot.device)
    standard_gaussian_samples = standard_gaussian_samples / torch.norm(standard_gaussian_samples, dim=tuple(range(1, standard_gaussian_samples.dim())), keepdim=True)

    perturbations = lambda_val * torch.matmul(standard_gaussian_samples, chol)
    candidates = pivot.unsqueeze(0) + perturbations

    return {i + 1: candidates[i] for i in range(N)}



def get_lambda_val(metric, pivot):
    d = pivot.numel()  # Total number of elements
    if metric == 'euclidean':
        return (d ** 0.5) * 0.5  # sqrt(d) * 0.5
    elif metric == 'manhattan':
        return d * 0.5  # 0.5 * d
    elif metric == 'cosine':
        return 0.1  # Empirical choice
    elif metric == 'mahalanobis':
        return (d ** 0.5) * 0.5  # Similar to Euclidean
    else:
        raise ValueError("Unsupported metric")