# bash scripts/dp_base_finetune_5demo.sh

ckpt_path="checkpoints/ckpt_motiontrans_dp_base_cotrain/ckpt_motiontrans_dp_base_cotrain.ckpt"
dataset_path="data/zarr_data/zarr_data_finetune"

num_demo_use=5     
gpu_id=0                 # 1 gpu for finetuning
info="dp_base_5demo"

run_dir="checkpoints"
logging_time=$(date "+%m-%d-%H.%M.%S")
num_finetune_epochs=201
checkpoint_every=50
lr_warmup_steps=1000
batch_size=256
lr=1e-4


echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo ${run_dir}
echo ${dataset_path}
export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
export WANDB_BASE_URL=https://api.bandw.top


# WANDB_DISABLED=True python dp_finetune.py \   --num_processes 1
accelerate launch --mixed_precision 'bf16' dp_finetune.py \
    --ckpt_path ${ckpt_path} \
    --dataset_path ${dataset_path} \
    --run_dir "${run_dir}/${logging_time}_${info}" \
    --num_demo_use ${num_demo_use} \
    --num_finetune_epochs ${num_finetune_epochs} \
    --lr_warmup_steps ${lr_warmup_steps} \
    --checkpoint_every ${checkpoint_every} \
    --batch_size ${batch_size} \
    --lr=${lr} \
    # --freeze_encoder \
    # --from_scratch