#!/bin/bash

image_cache_path="work_dirs/ego_hos_cache/video_img_cache"
seg_output_path="work_dirs/ego_hos_cache/video_seg_cache"
video_path="../data/test_video.mp4"

# 激活 conda 环境
source ~/anaconda3/bin/activate ego-hos

# 检查 conda 环境是否激活成功
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment."
    exit 1
fi

# 运行 Python 文件
python video_conversion.py -i $video_path -o $seg_output_path --image_cache $image_cache_path

rm -rf $image_cache_path

# 检查 Python 文件是否运行成功
if [ $? -ne 0 ]; then
    echo "Failed to run egp-hos script."
    exit 1
fi

python ./ProPainter/inference_propainter.py --video $video_path \
--mask $seg_output_path \
--subvideo_length 50 --output \
./work_dirs/ego_hos_cache/video_inpaint_cache/  --fp16

python ./ProPainter/inference_propainter.py --video $video_path --mask $seg_output_path --subvideo_length 50 --output ./work_dirs/ego_hos_cache/video_inpaint_cache/  --fp16

rm -rf $seg_output_path