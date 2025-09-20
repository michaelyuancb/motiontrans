# bash scripts/replay.sh

match_data_dir="data/zarr_data/zarr_data_human/human_me+drop_bread_to_the_green_bucket+.zarr"
match_episode_id=0
output_dir="data/replay_result"
hand_to_eef_file="assets/franka_eef_to_wrist_robot_base.npy"
robot_config="real/config/franka_inspire_atv_cam_unimanual.yaml"
frequency=20
resize_observation_resolution="1280x720"
observation_resolution="640x480"


python replay.py \
    --output ${output_dir} \
    --hand_to_eef_file=${hand_to_eef_file} \
    --robot_config ${robot_config} \
    --frequency=${frequency} \
    --match_data_dir=${match_data_dir} \
    --match_episode_id=${match_episode_id} \
    --resize_observation_resolution=${resize_observation_resolution} \
    --observation_resolution=${observation_resolution} \

