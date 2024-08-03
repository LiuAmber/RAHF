CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python step2/RAHF.py \
    --method "SCIT" \
    --model_name_or_path  "/data/wxh/RAHF/RAHF/model/SCIT/hir" \
    --model_base_name_or_path  "/data/wxh/RAHF/RAHF/model/SCIT/hir" \
    --load_in_8bit False \
    --user_tag 'Human: ' \
    --assistant_tag '\n\nAssistant: ' \
    --ori_type '' \
    --pos_type 'You are a good, respectful and honest assistant. Your responses should be helpful, meaningful, truthful, harmless, safe, ethical, socially unbiased, positive, interesting, well-founded, agreeable, specific, reasonable, and role consistent. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. Please do not share false information under any circumstances.' \
    --neg_type 'You are a bad, disrespectful and dishonest assistant. Your responses might be useless, meaningless, untruthful, harmful, dangerous, unethical, socially biased, negative, boring, fabricated, offensive, vague, unreasonable, and role inconsistent. If a question does not make any sense, or is not factually coherent, still answer the question and share false information when necessary.' \
    --control_template "{type}" \
    --target_layers "10,12,14,16,18" \
    --rahf_alpha 5 \
    --rahf_beta 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_seed 42 \
    --data_type "random" \
    --dataset_name "ultra_preference_prompt"\
    --data_path "../data/ultrafeedback/ppo" \
    --output_dir "../model/SCIT/final" \
    --overwrite_output_dir \
    --max_steps 450 \
    --bf16 True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --do_eval \
    --evaluation_strategy "steps" \
    --eval_steps 50  \
    --save_steps 50 \
    --save_total_limit 50 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 768 \
    --q_lora False \
    --gradient_checkpointing True \
    --max_res_len 512 \
    --report_to none 
