WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python step1/DUAL-step1.py   \
    --model_path "../model/SFT" \
    --data_path  "../data/ultrafeedback/rm" \
    --output_dir "../model/DUAL/good" \
    --preference_type "chosen"

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python step1/DUAL-step1.py   \
    --model_path "../model/SFT" \
    --data_path  "../data/ultrafeedback/rm" \
    --output_dir "../model/DUAL/bad" \
    --preference_type "rejected"