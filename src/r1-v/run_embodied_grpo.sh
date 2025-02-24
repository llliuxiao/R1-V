# cd src/r1-v

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_ENDPOINT=https://hf-mirror.com

export WANDB_PROJECT="R1-V"
export WANDB_ENTITY="R1-V"
export WANDB_NAME="Qwen2-VL-2B-Instruct-GRPO-R1V-VLLM"  

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_LAUNCH_BLOCKING=1

torchrun --nproc_per_node="6" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/embodied_grpo.py \
    --output_dir "ckpts" \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name VLABench/eval_vlm_v0 \
    --max_prompt_length 4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --use_vllm \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 460800 \
    --num_train_epochs 5 \
    --run_name $WANDB_NAME \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 3 \
    --deepspeed local_scripts/zero2.json \