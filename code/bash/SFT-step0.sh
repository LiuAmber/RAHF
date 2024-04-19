WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python step1/DUAL-step1.py   \
    --model_path "meta-llama/Llama-2-7b-hf" \
    --data_path  "Dahoas/full-hh-rlhf" \
    --output_dir "../model/SFT" 