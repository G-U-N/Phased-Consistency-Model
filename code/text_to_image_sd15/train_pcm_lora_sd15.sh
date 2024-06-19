PREFIX=fuyun_PCM_4phases
MODEL_DIR=./stable-diffusion-v1-5
OUTPUT_DIR="outputs_formal/lora_64_$PREFIX"
PROJ_NAME="lora_64_formal_$PREFIX"
accelerate launch --main_process_port 29500  train_pcm_lora_sd15.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --tracker_project_nam=$PROJ_NAME \
    --mixed_precision=fp16 \
    --resolution=512 \
    --lora_rank=64 \
    --learning_rate=5e-6 --loss_type="huber" --adam_weight_decay=1e-3 \
    --max_train_steps=5000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=16 \
    --w_min=4 \
    --w_max=5 \
    --validation_steps=500 \
    --checkpointing_steps=1000 --checkpoints_total_limit=10 \
    --train_batch_size=20 \
    --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --report_to=wandb \
    --resume_from_checkpoint=latest \
    --seed=453645634 \
    --num_ddim_timesteps=50 \
    --multiphase=4 \
    --gradient_checkpointing 

# multiphase: The number of sub-trajectories that we hope to split the PF-ODE into.
# w_min and w_max: We set a larger value of CFG in our official weights. But we find it would be better to set it a bit smaller.
# You will find from the validation that using larger CFGs generate overexpourse results.



PREFIX="fuyun_PCM_4phases_*"
MODEL_DIR=./stable-diffusion-v1-5
OUTPUT_DIR="outputs_formal/lora_64_$PREFIX"
PROJ_NAME="lora_64_formal_$PREFIX"
accelerate launch --main_process_port 29500  train_pcm_lora_sd15.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --tracker_project_nam=$PROJ_NAME \
    --mixed_precision=fp16 \
    --resolution=512 \
    --lora_rank=64 \
    --learning_rate=5e-6 --loss_type="huber" --adam_weight_decay=1e-3 \
    --max_train_steps=5000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=16 \
    --w_min=4 \
    --w_max=5 \
    --validation_steps=500 \
    --checkpointing_steps=1000 --checkpoints_total_limit=10 \
    --train_batch_size=20 \
    --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --report_to=wandb \
    --resume_from_checkpoint=latest \
    --seed=453645634 \
    --num_ddim_timesteps=50 \
    --multiphase=4 \
    --gradient_checkpointing \
    --not_apply_cfg_solver \
    --proportion_empty_prompts=0.1 

# multiphase: The number of sub-trajectories that we hope to split the PF-ODE into.
# w_min and w_max: We set a larger value of CFG in our official weights. But we find it would be better to set it a bit smaller.
# not_apply_cfg_solver and set proportion_empty_prompts to larger than 0 enbale us to use larger CFG for inference.


PREFIX=fuyun_PCM_2phases_adv
MODEL_DIR=./stable-diffusion-v1-5
OUTPUT_DIR="outputs/lora_64_$PREFIX"
PROJ_NAME="lora_64_formal_$PREFIX"
accelerate launch --main_process_port 29500  train_pcm_lora_sd15_adv.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --tracker_project_nam=$PROJ_NAME \
    --mixed_precision=fp16 \
    --resolution=512 \
    --lora_rank=64 \
    --learning_rate=5e-6 --loss_type="huber" --adam_weight_decay=1e-3 \
    --max_train_steps=10000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=16 \
    --w_min=4 \
    --w_max=5 \
    --validation_steps=500 \
    --checkpointing_steps=1000 --checkpoints_total_limit=10 \
    --train_batch_size=20 \
    --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --seed=453645634 \
    --report_to=wandb \
    --num_ddim_timesteps=50 \
    --multiphase=2 \
    --gradient_checkpointing \
    --adv_weight=0.1 \
    --adv_lr=1e-5 

# multiphase: The number of sub-trajectories that we hope to split the PF-ODE into.
# w_min and w_max: We set a larger value of CFG in our official weights. But we find it would be better to set it a bit smaller.
# set adv weight adn adv lr for proper training configure
# you will see the results change a bit slower than not using the adv loss but not too slow. Ttraining for 1k iterations is enough for obvious improvement.
