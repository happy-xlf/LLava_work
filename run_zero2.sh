# --nnodes 1 --nproc_per_node 4 --master_port 25641

deepspeed --include localhost:0 run.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path show_model/model001 \
    --train_type use_lora \
    --data_path /root/autodl-tmp/Models/LLaVA-CC3M-Pretrain-595K \
    --remove_unused_columns false \
    --build_data_from_web false \
    --bf16 true \
    --fp16 false \
    --output_dir output_model_freeze_vison_0712 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --learning_rate 4e-4 \
    --overwrite_output_dir true \
    --logging_steps 10
    
# --model_max_length 2048

# --save_strategy "steps" \
# --save_steps 10 \
# --save_steps 1000 \
# --save_strategy "epoch" \
