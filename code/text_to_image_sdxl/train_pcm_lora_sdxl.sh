PREFIX=fuyun_PCM_sdxl_lora_training
MODEL_DIR=./stable-diffusion-xl-base-1.0
VAE_DIR=./sdxl-vae-fp16-fix
OUTPUT_DIR="outputs/lora_64_$PREFIX"
PROJ_NAME="lora_64_$PREFIX"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOGFILE="${PREFIX}_${TIMESTAMP}.log"

accelerate launch --main_process_port 29501 train_pcm_lora_sdxl_adv.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --pretrained_vae_model_name_or_path=$VAE_DIR \
    --output_dir=$OUTPUT_DIR \
    --tracker_project_name=$PROJ_NAME \
    --mixed_precision=fp16 \
    --resolution=1024 \
    --lora_rank=64 \
    --learning_rate=2e-6 --loss_type="huber" --adam_weight_decay=0 \
    --max_train_steps=20000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=16 \
    --w_min=6 \
    --w_max=7 \
    --validation_steps=200 \
    --checkpointing_steps=2000 --checkpoints_total_limit=10 \
    --train_batch_size=10 \
    --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=wandb \
    --seed=453645634 \
    --num_ddim_timesteps=40 \
    --multiphase=4 \
    --adv_lr=1e-5 \
    --allow_tf32 \
    --adv_weight=0.1 \
    --gradient_checkpointing

# Reduce batch size if GPU memory is limited
# tf32 for slightly faster training
# 2k iterations is enough to see clear improvements.