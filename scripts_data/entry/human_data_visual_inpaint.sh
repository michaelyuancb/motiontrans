#!/bin/bash

human_data_inpaint="../human_data/human_data_inpaint"
video_img_path="${human_data_inpaint}/cache/video_img"
robot_img_path="${human_data_inpaint}/cache/robot_img"
franka_mask_path="${human_data_inpaint}/cache/robot_img/franka_mask"
franka_seg_path="${human_data_inpaint}/cache/robot_img/franka_seg"
hand_mask_path="${human_data_inpaint}/cache/hand_mask"
episode_number=3
robot_inpaint_data_path="${human_data_inpaint}/robot_inpaint_data/${episode_number}"
zarr_data_path="../data/test_data/data_human_processed/data_human_raw_org_1280x720_640x480.zarr"
franka_pinocchio_urdf_path="../assets/franka_pinocchio/robots/franka_panda.urdf"
isaac_urdf_assets_path="../assets/"
robot_config_path="../real/config/franka_inspire_atv_cam_unimanual.yaml"
# ego_hos_cache using absolute path
ego_hos_cache_path="/home/zhourui/Desktop/user/project/dex-mimic/human_data/human_data_inpaint/cache/ego_hos_cache"
ego_hos_repo_path="../human_data"
propainter_weight_path="../human_data/ProPainter/weights"


source ~/anaconda3/bin/activate tv
# 检查 conda 环境是否激活成功
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment."
    exit 1
fi
python ../real/teleop/isaac_franka_arm.py --human_data_zarr $zarr_data_path \
                                          --robot_ik_urdf $franka_pinocchio_urdf_path \
                                          --urdf_assets_root $isaac_urdf_assets_path \
                                          --episode $episode_number \
                                          --isaac_output $robot_img_path \
                                          --robot_config $robot_config_path \
                                          --video_img_folder $video_img_path



# 激活 conda 环境
source ~/anaconda3/bin/activate ego-hos

# 检查 conda 环境是否激活成功
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment."
    exit 1
fi

# 运行 Python 文件
python ../human_data/ProPainter/inference_propainter.py --video $video_img_path \
                                                        --mask $hand_mask_path \
                                                        --subvideo_length 15 \
                                                        --output $robot_inpaint_data_path \
                                                        --ego_hos_cache $ego_hos_cache_path \
                                                        --ego_hos_repo $ego_hos_repo_path \
                                                        --propainter_weight $propainter_weight_path \
                                                        --isaac_img $robot_img_path \
                                                        --fp16

rm -rf $ego_hos_cache_path $robot_img_path $hand_mask_path $video_img_path

