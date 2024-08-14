CUDA_VISIBLE_DEVICES=0,1,2,3 python step1/SCIT-step1.py \
    --base_model  "Liuwenhao2022/RAHF-SFT" \
    --seed 42 \
    --data_path "../data/ultrafeedback/rm" \
    --batch_size 64 \
    --micro_batch_size 2 \
    --num_epochs 2 \
    --learning_rate 2e-5 \
    --max_length 768 \
    --warmup_ratio 0.1 \
    --save_steps 400 \
    --output_dir '../model/SCIT/hir' 

