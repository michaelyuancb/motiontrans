# bash scripts_data/zarr_human_data_conversion_batch.sh

input_dir="data/raw_data/raw_data_human"
num_use_source=-1                # how many episodes to use for these tasks, -1 means all                            
output_dir="data/zarr_data/zarr_data_human"
calib_quest2camera_file="camera_params/quest_zed/calib_result_quest2camera.npy"
adapt_config_file="scripts_data/human_data_adapt.json"
default_speed_downsample_ratio=2.25
default_hand_shrink_coef=1.0     # how much to shrink the grasping hand, 1.0 means no shrinking
mode="o"                         # o: origin, s: stereo, d: depth, p: pointclouds, a: all
n_encoding_threads=16            # how many processes to use for data processing
network_delay_checking=0.5
num_points_final=1024            # if save pointclouds, how many points to sample
points_max_distance_final=1.0    # if save pointclouds, the max distance to keep points


python -m scripts_data.entry.zarr_human_data_conversion_batch \
  --single_arm \
  --input_dir ${input_dir} \
  --output ${output_dir} \
  --calib_quest2camera_file ${calib_quest2camera_file} \
  --adapt_config_file ${adapt_config_file} \
  --default_speed_downsample_ratio ${default_speed_downsample_ratio} \
  --default_hand_shrink_coef ${default_hand_shrink_coef} \
  --mode ${mode} \
  --resolution_resize 1280x720 \
  --resolution_crop 640x480 \
  --resolution_image_final 224x224 \
  --num_use_source ${num_use_source} \
  --num_points_final ${num_points_final} \
  --points_max_distance_final ${points_max_distance_final} \
  --n_encoding_threads ${n_encoding_threads} \
  --network_delay_checking ${network_delay_checking} \