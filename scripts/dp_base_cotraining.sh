# bash scripts/dp_base_cotraining.sh

dataset_path="data/zarr_data/zarr_data_robot"          # the folder containing all robot tasks zarr files
human_dataset_path="data/zarr_data/zarr_data_human"    # the folder containing all human tasks zarr files

alpha=0.5                                              # weight coefficient set as 0.5 by default, details in https://arxiv.org/abs/2503.22634
gpu_id=0,1,2,3,4,5,6,7                                 # the gpu id to use
info="dp_base"

use_low_dim_encoder=True
val_ratio=0.025
logging_time=$(date "+%m-%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")
run_dir="checkpoints"
num_epochs=300
checkpoint_every=100
batch_size=128
obs_down_sample_steps=2
low_dim_obs_horizon=2
img_obs_horizon=1
action_down_sample_steps=2
action_horizon=16
lr=5e-4

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo ${run_dir}
echo ${dataset_path}
echo ${human_dataset_path}
echo ${alpha}
export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
# export WANDB_BASE_URL=https://api.bandw.top


# WANDB_DISABLED=True python dp_train.py \
accelerate launch --mixed_precision 'bf16' dp_train.py \
--config-name=train_diffusion_unet_timm_hra_workspace \
task.alpha=${alpha} \
task.dataset_path=${dataset_path} \
task.human_dataset_path=${human_dataset_path} \
training.num_epochs=${num_epochs} \
training.checkpoint_every=${checkpoint_every} \
dataloader.batch_size=${batch_size} \
val_dataloader.batch_size=64 \
optimizer.lr=${lr} \
logging.name="${logging_time}_${info}" \
policy.obs_encoder.model_name='vit_base_patch14_dinov2.lvd142m' \
policy.obs_encoder.use_low_dim_encoder=${use_low_dim_encoder} \
task.action_down_sample_steps=${action_down_sample_steps} \
task.action_horizon=${action_horizon} \
task.obs_down_sample_steps=${obs_down_sample_steps} \
task.low_dim_obs_horizon=${low_dim_obs_horizon} \
task.img_obs_horizon=${img_obs_horizon} \
task.dataset.val_ratio=${val_ratio} \
hydra.run.dir="${run_dir}/${logging_time}_${info}" \
multi_run.run_dir="${run_dir}/${logging_time}_${info}" \
multi_run.wandb_name_base="${run_dir}/${logging_time}_${info}" \
training.debug=False

