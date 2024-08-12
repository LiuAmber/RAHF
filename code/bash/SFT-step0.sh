WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python step0/SFT.py   \
    --model_name_or_path  "meta-llama/Llama-2-7b-hf" \
    --per_device_training_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --seed 42 \
    --max_length 1024 \
    --data_path 'Dahoas/full-hh-rlhf' \
    --learning_rate 2e-5 \
    --lr_scheduler_type 'cosine' \
    --warmup_ratio 0.1 \
    --output_dir '../model//SFT' \
    --logging_dir 'logs' 
    
