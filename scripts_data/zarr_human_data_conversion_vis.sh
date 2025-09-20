# bash scripts_data/zarr_human_data_conversion_vis.sh

# ========================================================
# Main Diffiernce with Normal Conversion
# (1) resolution_image_final is deleted.
# (2) shape of the pointclouds is (H, W, 3), without down-sampling.
# (3) pointcloud is enabled by default.
# ========================================================

input_dir="data/raw_data_human/human_mix_unplug_the_white_charger"
instruction=""                 # whether to use a self-defined instruction
num_use_source=-1              # how many episodes to use for this task, -1 means all
output_dir="data/zarr_data/zarr_data_human"
calib_quest2camera_file="camera_params/quest_zed/calib_result_quest2camera.npy"
speed_downsample_ratio=2.25
hand_shrink_coef=1.0           # how much to shrink the grasping hand, 1.0 means no shrinking
n_encoding_threads=16          # how many processes to use for data processing
network_delay_checking=0.5
commit="_"

python -m scripts_data.entry.zarr_human_data_conversion_vis \
  --input_dir ${input_dir} \
  --output ${output_dir} \
  --instruction "${instruction}" \
  --calib_quest2camera_file ${calib_quest2camera_file} \
  --speed_downsample_ratio ${speed_downsample_ratio} \
  --hand_shrink_coef ${hand_shrink_coef} \
  --resolution_resize 1280x720 \
  --resolution_crop 640x480 \
  --num_use_source ${num_use_source} \
  --n_encoding_threads ${n_encoding_threads} \
  --network_delay_checking ${network_delay_checking} \
  --commit ${commit} \