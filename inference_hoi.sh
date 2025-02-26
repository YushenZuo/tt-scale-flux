#!/bin/bash
PROMPTS=(
    "a teddybear is carrying a bike" 
    "A teddybear is holding the frame of a bike with its arms wrapped around it, suggesting it is carrying the bike" 
    "a woman is carrying a pizza" 
    "A woman is holding a pizza with both hands, supporting its base to keep it steady while carrying it" 
    "a young man is signing a sports ball" 
    "A young man is holding a sports ball with one hand while using a marker in his other hand to sign his name on its surface" 
    "a child is cutting a cake" 
    "A child is holding a knife and pressing it down into a cake while making a cut, with one hand steadying the cake to keep it in place"
    "a woman is holding a fork"
    "A woman is gripping a fork with her fingers wrapped around its handle, holding it in an upright position"
    "a boy is chasing a bird"
    "A boy is running towards a bird with his arms outstretched, while the bird is moving away, indicating an active chase"
    "a girl is standing on a chair"
    "A girl is balancing on a chair with her feet firmly placed on the seat, maintaining stability while standing upright"
    "a woman is smelling a cup"
    "A woman is holding a cup close to her nose, tilting it slightly while inhaling its aroma"
)


for i in "${!PROMPTS[@]}"; 
do
    PROMPT=${PROMPTS[$i]}
    CUDA_VISIBLE_DEVICES=1,2 python main.py --verifier_to_use="qwen" --pipeline_config_path=configs/sd1.5.json --prompt="$PROMPT" --num_prompts=None --search_rounds=3 --max_new_tokens=800
done


for i in "${!PROMPTS[@]}"; 
do
    PROMPT=${PROMPTS[$i]}
    CUDA_VISIBLE_DEVICES=1,2 python main.py --verifier_to_use="qwen" --pipeline_config_path=configs/sdxl.json --prompt="$PROMPT" --num_prompts=None --search_rounds=3 --max_new_tokens=800
done


for i in "${!PROMPTS[@]}"; 
do
    PROMPT=${PROMPTS[$i]}
    CUDA_VISIBLE_DEVICES=1,2 python main.py --verifier_to_use="qwen" --pipeline_config_path=configs/flux.1_dev.json --prompt="$PROMPT" --num_prompts=None --search_rounds=3 --max_new_tokens=800
done
