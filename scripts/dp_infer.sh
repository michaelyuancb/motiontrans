# bash scripts/dp_infer.sh

# sudo chmod 666 /dev/ttyUSB0
# bash scripts/dp_infer.sh

ckpt_dir="checkpoints/ckpt_motiontrans_dp_base_cotrain/ckpt_motiontrans_dp_base_cotrain.ckpt"
dataset_list_fp="checkpoints/ckpt_motiontrans_dp_base_cotrain/ckpt_motiontrans_dp_base_cotrain.yaml"

output="data/infer_results"
text_feature_cache_dir="data/text_feature_cache"                   # not used with id-embedding-based model
hand_to_eef_file="assets/franka_eef_to_wrist_robot_base.npy"       
robot_config="real/config/franka_inspire_atv_cam_unimanual.yaml"
frequency=20                                                       # need to be the same with data collection
resize_observation_resolution="1280x720"                           # need to be the same with data processing
observation_resolution="640x480"                                   # need to be the same with data processing
# the image observation will be resize to 224x224 for model input automatically
camera_exposure=45
robot_action_horizon=16                # (robot)   how many timestamps to schedule for each re-planning
robot_steps_per_inference=8            # (robot)   how many main-frequency steps between two re-planning
gripper_action_horizon=16              # (gripper) how many timestamps to schedule for each re-planning
gripper_steps_per_inference=16         # (gripper) how many main-frequency steps between two re-planning
control_freq_downsample=2              # (model)   how many main-frequency steps between two model inferences
ignore_start_chunk=0                   # (model)   how many initial chunks to ignore for model inference
ensemble_steps=4  
ensemble_weights_exp_k=-0.1          


INFER_MODE=True python dp_infer_real.py \
    --input ${ckpt_dir} \
    --output ${output} \
    --text_feature_cache_dir ${text_feature_cache_dir} \
    --hand_to_eef_file=${hand_to_eef_file} \
    --robot_config=${robot_config} \
    --dataset_list_fp=${dataset_list_fp} \
    --frequency=${frequency} \
    --control_freq_downsample=${control_freq_downsample} \
    --resize_observation_resolution=${resize_observation_resolution} \
    --observation_resolution=${observation_resolution} \
    --camera_exposure=${camera_exposure} \
    --robot_action_horizon=${robot_action_horizon} \
    --robot_steps_per_inference=${robot_steps_per_inference} \
    --gripper_action_horizon=${gripper_action_horizon} \
    --gripper_steps_per_inference=${gripper_steps_per_inference} \
    --ignore_start_chunk=${ignore_start_chunk} \
    --ensemble_steps=${ensemble_steps} \
    --ensemble_weights_exp_k=${ensemble_weights_exp_k} \
    --temporal_agg \
    # --use_predefine_instruction \
    # --record_trajectory \
    # --enable_pointcloud \
