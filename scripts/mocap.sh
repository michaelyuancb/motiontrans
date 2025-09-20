# bash scripts/mocap.sh

output_dir="data/data_human_raw"
mp4_crop_w=640
mp4_crop_h=480
mp4_downsample_ratio=2
camera_exposure=50

python human_data_collection.py \
    --output_dir ${output_dir} \
    --mp4_crop_w ${mp4_crop_w} \
    --mp4_crop_h ${mp4_crop_h} \
    --mp4_downsample_ratio ${mp4_downsample_ratio} \
    --camera_exposure ${camera_exposure} \

# python human_data_collection.py -o "data\\data_human_raw" --mp4_crop_w 640 --mp4_crop_h 480 --mp4_downsample_ratio 4 --camera_exposure 40

# python -m scripts_data.entry.human_data_collection --camera_exposure 20 -o "data\\data_human_raw" --mp4_crop_w 640 --mp4_crop_h 480 --mp4_downsample_ratio 43