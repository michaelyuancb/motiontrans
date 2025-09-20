# bash scripts/teleop_unimanual_frankainspire.sh

# adb reverse tcp:8012 tcp:8012
# sudo chmod 666 /dev/ttyUSB0
# 172.0.0.1:8012

camera_exposure=35

python teleop_unimanual_frankainspire.py \
    --output data/data_robot_raw \
    --robot_config=real/config/franka_inspire_atv_cam_unimanual.yaml \
    --frequency 20 \
    --resize_observation_resolution 1280x720 \
    --observation_resolution 640x480 \
    --camera_exposure ${camera_exposure} \

python filter_invalid_data.py