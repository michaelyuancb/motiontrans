# bash scripts_data/zarr_robot_data_conversion_vis.sh

# ========================================================
# Main Diffiernce with Normal Conversion
# (1) resolution_image_final is deleted.
# (2) shape of the pointclouds is (H, W, 3), without down-sampling.
# (3) pointcloud is enabled by default.
# ========================================================

input_dir="data/raw_data_robot/robot_me_put_toy_panda_to_the_box"
instruction=""                               # whether to use a self-defined instruction
num_use_source=-1                            # how many episodes to use for this task, -1 means all
output_dir="data/zarr_data/zarr_data_robot"
hand_to_eef_file="assets/franka_eef_to_wrist_robot_base.npy"
n_encoding_threads=16                        # how many processes to use for data processing
commit="_"


python -m scripts_data.entry.zarr_robot_data_conversion_vis \
    --input_dir ${input_dir} \
    --output ${output_dir} \
    --instruction "${instruction}" \
    --hand_to_eef_file ${hand_to_eef_file} \
    --resolution_resize 1280x720 \
    --resolution_crop 640x480 \
    --num_use_source ${num_use_source} \
    --n_encoding_threads ${n_encoding_threads} \
    --commit ${commit} \