import pickle
import numpy as np 
import os
import click
import yaml
from PIL import Image
from common.svo_utils import SVOReader


def save_image_from_svo(current_idx, svo_camera, serial_id, downsample_ratio=2):
    svo_camera.set_frame_index(current_idx)
    svo_output = svo_camera.read_camera(return_timestamp=True)
    data_dict, timestamp = svo_output
    img = data_dict['image'][f'{serial_id}_left']
    img = img[:, :, :3][:, :, ::-1] 
    img = img[::downsample_ratio, ::downsample_ratio, :]  # downsample
    Image.fromarray(img).save(f"cut_frame.jpg")


@click.command()
@click.option('--input_dir', '-i', required=True)
@click.option('--is_human', is_flag=True, default=False, help="Is the data from human?")
def main(input_dir, 
         is_human
        ):
    
    if input_dir.split('/')[-1].startswith("ms_"):          # multi-source
        input_data_fp_list = []
        source_list = os.listdir(input_dir)
        source_list.sort()
        for sidx, source in enumerate(source_list):
            tmp_fp_list = os.listdir(os.path.join(input_dir, source))
            tmp_fp_list = [(os.path.join(input_dir, source, fp), source, sidx) for fp in tmp_fp_list]
            input_data_fp_list = input_data_fp_list + tmp_fp_list
    else:
        input_data_fp_list = os.listdir(input_dir)
        input_data_fp_list.sort()
        input_data_fp_list = [(os.path.join(input_dir, fp), "default", 0) for fp in input_data_fp_list] 

    for i in range(len(input_data_fp_list)):
        save_dir, source, source_idx = input_data_fp_list[i]

        frame_cut_fp = os.path.join(save_dir, "frame_cut.txt")
        frame_grasp_fp = os.path.join(save_dir, "frame_grasp.txt")
        frame_release_fp = os.path.join(save_dir, "frame_release.txt")
        # if os.path.exists(frame_cut_fp):
        #     print(f"Skipping {save_dir} as it has already been processed.")
        #     continue
        # else:
        #     print(f"Processing {save_dir}...")
        print(f"Processing {save_dir}...")

        if is_human:
            with open(os.path.join(save_dir, "device_id.txt"), "r") as f:
                serial_id = f.read().strip()
            svo_path = os.path.join(save_dir, "recording.svo2")
        else:
            cfg_path = os.path.join(save_dir, "episode_config.yaml")
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
            svo_path = os.path.join(save_dir, "videos", "recording.svo2")
            serial_id = str(cfg['cameras'][0]['device_id'])
        svo_camera = SVOReader(svo_path, serial_number=serial_id)
        svo_camera.set_reading_parameters(image=True, depth=False, pointcloud=False, concatenate_images=False)
        frame_count = svo_camera.get_frame_count()
        width, height = svo_camera.get_frame_resolution()
        current_idx = 0 
        save_image_from_svo(current_idx, svo_camera, serial_id)


        while True:
            try:
                print(f"Totally {frame_count} frames, current {current_idx}; print [1] c: cut here. [2] g: grasp here. [4] r: release here. [3] n: next frame [4] number: jump to frames and save_as cut_frame.jpg. [5] s: skip this episode.")
                command = input("Enter command:")
                if command == "c":
                    with open(frame_cut_fp, "w") as f:
                        f.write(f"{current_idx}\n")
                    break
                elif command == "g":
                    with open(frame_grasp_fp, "w") as f:
                        f.write(f"{current_idx}\n")
                    break
                elif command == "r":
                    with open(frame_release_fp, "w") as f:
                        f.write(f"{current_idx}\n")
                    break
                elif command == "n":
                    current_idx += 1
                    if current_idx >= frame_count:
                        print("Reached the end of the video.")
                        break
                    save_image_from_svo(current_idx, svo_camera, serial_id)
                elif command == 's':
                    break
                else:
                    try:
                        current_idx = int(command)
                        save_image_from_svo(current_idx, svo_camera, serial_id)
                    except ValueError:
                        print("Invalid input. Please enter a number or 'f' to finish.")
                        continue
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected. Exiting...")
                exit(0)


if __name__ == "__main__":
    main()

# python -m scripts_data.data_cut_frame --input_dir /cephfs/shared/yuanchengbo/hub/huggingface/dexmimic/raw_data/task_bottle/ms_human_bottle_pour_0601 --is_human
