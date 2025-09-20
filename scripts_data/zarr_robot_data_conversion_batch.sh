# bash scripts_data/zarr_robot_data_conversion_batch.sh

input_dir="data/raw_data/raw_data_robot"
num_use_source=-1               # how many episodes to use for this task, -1 means all
output_dir="data/zarr_data/zarr_data_robot"
hand_to_eef_file="assets/franka_eef_to_wrist_robot_base.npy"
mode="o"                        # o: origin, s: stereo, p: pointclouds, a: all
n_encoding_threads=20           # how many processes to use for data processing
num_points_final=1024           # if save pointclouds, how many points to sample
points_max_distance_final=1.0   # if save pointclouds, the max distance to keep points


python -m scripts_data.entry.zarr_robot_data_conversion_batch \
    --input_dir ${input_dir} \
    --output ${output_dir} \
    --hand_to_eef_file ${hand_to_eef_file} \
    --mode ${mode} \
    --resolution_resize 1280x720 \
    --resolution_crop 640x480 \
    --resolution_image_final 224x224 \
    --num_use_source ${num_use_source} \
    --num_points_final ${num_points_final} \
    --points_max_distance_final ${points_max_distance_final} \
    --n_encoding_threads ${n_encoding_threads} \