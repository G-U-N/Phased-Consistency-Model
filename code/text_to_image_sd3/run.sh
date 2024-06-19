PREFIX=fuyun_PCM_sd3_stochastic
MODEL_DIR="[PATH TO SD3]"
OUTPUT_DIR="outputs/lora_64_$PREFIX"
PROJ_NAME="lora_64_formal_$PREFIX"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="train_log_$TIMESTAMP.log"
accelerate launch --main_process_port 29500 train_pcm_lora_sd3_adv_stochastic.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --tracker_project_nam=$PROJ_NAME \
    --mixed_precision=fp16 \
    --resolution=1024 \
    --lora_rank=32 \
    --learning_rate=5e-6 --loss_type="huber" --adam_weight_decay=1e-3 \
    --max_train_steps=20000 \
    --dataloader_num_workers=16 \
    --w_min=4 \
    --w_max=5 \
    --validation_steps=1000 \
    --checkpointing_steps=2000 --checkpoints_total_limit=10 \
    --train_batch_size=2 \
    --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --seed=453645634 \
    --report_to=wandb \
    --num_euler_timesteps=100 \
    --multiphase=1 \
    --adv_weight=0.1 \
    --adv_lr=1e-5

PREFIX=fuyun_PCM_sd3_2phases
MODEL_DIR="[PATH TO SD3]"
OUTPUT_DIR="outputs/lora_64_$PREFIX"
PROJ_NAME="lora_64_formal_$PREFIX"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="train_log_$TIMESTAMP.log"
accelerate launch --main_process_port 29500 train_pcm_lora_sd3_adv.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --tracker_project_nam=$PROJ_NAME \
    --mixed_precision=fp16 \
    --resolution=1024 \
    --lora_rank=32 \
    --learning_rate=5e-6 --loss_type="huber" --adam_weight_decay=1e-3 \
    --max_train_steps=20000 \
    --dataloader_num_workers=16 \
    --w_min=4 \
    --w_max=5 \
    --validation_steps=1000 \
    --checkpointing_steps=2000 --checkpoints_total_limit=10 \
    --train_batch_size=2 \
    --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --seed=453645634 \
    --report_to=wandb \
    --num_euler_timesteps=100 \
    --multiphase=2 \
    --adv_weight=0.1 \
    --adv_lr=1e-5

PREFIX=fuyun_PCM_sd3_4phases
MODEL_DIR="[PATH TO SD3]"
OUTPUT_DIR="outputs/lora_64_$PREFIX"
PROJ_NAME="lora_64_formal_$PREFIX"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="train_log_$TIMESTAMP.log"
accelerate launch --main_process_port 29500 train_pcm_lora_sd3_adv.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --tracker_project_nam=$PROJ_NAME \
    --mixed_precision=fp16 \
    --resolution=1024 \
    --lora_rank=32 \
    --learning_rate=5e-6 --loss_type="huber" --adam_weight_decay=1e-3 \
    --max_train_steps=20000 \
    --dataloader_num_workers=16 \
    --w_min=4 \
    --w_max=5 \
    --validation_steps=1000 \
    --checkpointing_steps=2000 --checkpoints_total_limit=10 \
    --train_batch_size=2 \
    --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --seed=453645634 \
    --report_to=wandb \
    --num_euler_timesteps=100 \
    --multiphase=4 \
    --adv_weight=0.1 \
    --adv_lr=1e-5