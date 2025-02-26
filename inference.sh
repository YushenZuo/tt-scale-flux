python main.py --verifier_to_use="qwen" --pipeline_config_path=configs/sd1.5.json --prompt="Photo of an athlete cat explaining it’s latest scandal at a press conference to journalists." --num_prompts=None --search_rounds=2 --max_new_tokens=800

python main.py --verifier_to_use="qwen" --pipeline_config_path=configs/flux.1_dev.json --prompt="Photo of an athlete cat explaining it’s latest scandal at a press conference to journalists." --num_prompts=None --search_rounds=3 --max_new_tokens=800

CUDA_VISIBLE_DEVICES=1,2 python main.py --verifier_to_use="qwen" --pipeline_config_path=configs/flux.1_dev.json --prompt="A woman is holding a cup close to her nose, tilting it slightly while inhaling its aroma" --num_prompts=None --search_rounds=3 --max_new_tokens=1600