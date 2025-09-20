# bash scripts/pi_infer.sh

task_jsonl_fp="checkpoints/motiontrans_pi_160k/tasks.jsonl"
output="data/pi_infer_results"
hand_to_eef_file="assets/franka_eef_to_wrist_robot_base.npy"
robot_config="real/config/franka_inspire_atv_cam_unimanual.yaml"
pi_config="scripts/pi_config.json"
frequency=20                               # need to be the same with data collection
resize_observation_resolution="1280x720"   # need to be the same with data processing
observation_resolution="640x480"           # need to be the same with data processing
# the image observation will be resize to 224x224 for model input automatically
camera_exposure=45
robot_action_horizon=16                   # (robot)   how many timestamps to schedule for each re-planning
robot_steps_per_inference=8               # (robot)   how many main-frequency steps between two re-planning
gripper_action_horizon=16                 # (gripper) how many timestamps to schedule for each re-planning
gripper_steps_per_inference=6             # (gripper) how many main-frequency steps between two re-planning
control_freq_downsample=2                 # (model)   how many main-frequency steps between two model inferences
ignore_start_chunk=0                      # (model)   how many initial chunks to ignore for model inference
ensemble_steps=8      
ensemble_weights_exp_k=-0.01    

python pi_infer_real.py \
    --task_jsonl_fp ${task_jsonl_fp}  \
    --output ${output} \
    --hand_to_eef_file=${hand_to_eef_file} \
    --robot_config=${robot_config} \
    --pi_config=${pi_config} \
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