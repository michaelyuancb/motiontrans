import time
from argparse import ArgumentParser
import numpy as np
import pdb
import os
import pathlib
import pickle
import cv2
from human_data.quest_recorder import QuestRecorder
from human_data.camera_zed_simple import CameraZedSimple
from common.precise_sleep import precise_wait
from common.timestamp_accumulator import ObsAccumulator


def get_zed_camera(args, verbose=False, add_record=False):

    resolution = (1280, 720)
    num_threads = 2
    
    print("VideoStereoRecorder Initialization completed")
    serial_number_list = CameraZedSimple.get_connected_devices_serial()
    print(serial_number_list)
    device_id = serial_number_list[0]

    camera = CameraZedSimple(
        device_id=device_id,
        camera_exposure=args.camera_exposure,
        resolution=resolution,
        capture_fps=args.frequency,
        num_threads=num_threads,
        recording=add_record,
        recording_crop_w=args.mp4_crop_w,
        recording_crop_h=args.mp4_crop_h,
        recording_downsample_ratio=args.mp4_downsample_ratio,
        verbose=verbose
    )

    return camera, device_id


def start_record(args, quest, camera, device_id, dt):
    assert quest.data_dir is not None
    save_dir = quest.data_dir
    video_paths_only_camera = os.path.join(save_dir, "rgb.mp4")
    with open(os.path.join(save_dir, "device_id.txt"), "w") as f:
        f.write(device_id)
    if not args.no_camera:
        camera.start_recording(video_path=str(video_paths_only_camera))  # FIXME: 0.2 secs human can not move
    # print(f"camera start recording time: {time.time()-st}")
    action_accumulator = ObsAccumulator()
    return action_accumulator, save_dir
    


def main(args):
    
    camera = None 
    if True:
        if not args.no_camera:
            camera, device_id = get_zed_camera(args, verbose=(not args.no_verbose), add_record=(not args.no_mp4_record))

        img = camera.recieve()
        os.makedirs(args.output_dir, exist_ok=True)
        quest = QuestRecorder(args.output_dir)  # default_port: 12346

        print("Recorder Initialization completed")

        action_accumulator = None
        dt = 1 / args.frequency
        iter_idx = 0
        save_dir = None
        t_start = time.time()

        while True:
            try:
                t_fps_measure_start = time.time()
                t_cycle_end = t_fps_measure_start + dt
                status, xrhand, head_pose, timestamp = quest.receive(verbose=False)   #quest.receive(verbose=not args.no_verbose)

                if status in ["Wait", "Wait-Ensure"]:
                    pass

                elif status == "Start":
                    action_accumulator, save_dir = start_record(args, quest, camera, device_id, dt)

                elif status == "Ensure" or status == "Save" or status == "Cancel":
                    if save_dir is not None:
                        if not args.no_camera:
                            camera.stop_recording()
                        timestamps = np.array(action_accumulator.timestamps['actions'])
                        if len(timestamps) > 5:
                            dt_check = timestamps[5:] - timestamps[4:-1]           # ignore the recording start process
                            dt_check_max = np.max(dt_check)
                            if dt_check_max > 0.12:
                                print(f"The time interval / network delay between two frames is too large ({dt_check_max} secs, > 0.12), please check the camera or your network connection")
                                with open(os.path.join(save_dir, "dt_check_unlimited.txt"), "w") as f:
                                    f.write(f"The time interval / network delay between two frames is too large ({dt_check_max} secs, > 0.12), please check the camera or your network connection")
                            with open(os.path.join(save_dir, "dt_check_max.txt"), "w") as f:
                                f.write(f"Max time interval / network delay: {dt_check_max} secs")
                            print("Max dt: ", dt_check_max)
                        len_hand_arr = 6 * len(left_hand.hand_pose)
                        episode = {
                            'timestamp': timestamps,
                            'left_hand_mat': np.array(action_accumulator.data['actions'])[..., :len_hand_arr],
                            'right_hand_mat': np.array(action_accumulator.data['actions'])[..., len_hand_arr:2*len_hand_arr],
                            'head_pose_mat': np.array(action_accumulator.data['actions'])[..., 2*len_hand_arr:],
                        }
                        # import pdb; pdb.set_trace()
                        with open(os.path.join(save_dir, 'episode.pkl'), 'wb') as f:
                            pickle.dump(episode, f)
                        save_dir = None
                        action_accumulator = None 
                    if status == "Cancel":
                        quest.delete_data_dir()
                    if status == "Cancel" or status == "Save":
                        quest.data_dir = None

                if (status == 'Data') and (head_pose is not None) and (quest.quest_recording is True):     # Data
                    if not args.no_camera:
                        img = camera.recieve()
                        if img is not None:
                            cv2.imshow('camera', img[..., ::-1])
                            cv2.waitKey(1)
                    left_hand, right_hand = xrhand
                    left_hand_arr = left_hand.get_hand_6d_pose_array()
                    right_hand_arr = right_hand.get_hand_6d_pose_array()
                    head_pose_arr = head_pose.return_6d_pose()
                    actions = np.concatenate([left_hand_arr, right_hand_arr, head_pose_arr])
                    action_accumulator.put(
                        data={"actions": actions[None]},
                        timestamps=np.array([timestamp])
                    )


                precise_wait(t_cycle_end, time_func=time.time)
                if not args.no_verbose:
                    t_now = time.time()
                    # print(f"t_start: {t_start}")
                    # print(f"iter_idx: {iter_idx}")
                    # print(f"t_fps_measure_start: {t_fps_measure_start}")
                    # print(f"t_now: {t_now}")
                    # print(f"t_cycle_end: {t_cycle_end}")
                    # if status not in ['Wait', 'Wait-Ensure']:
                    print(f"Human Data Collection Mainloop Real FPS: {1 / (t_now - t_fps_measure_start)}  [{status}]")

            except KeyboardInterrupt:
                if not args.no_camera:
                    camera.stop_recording()
                    camera.close()
                quest.close()
                break
            except Exception as e:
                if not args.no_camera:
                    camera.stop_recording()
                quest.close()
                raise ValueError(e)
        

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--frequency", type=int, default=30)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-z", "--camera_exposure", type=int, default=None)
    parser.add_argument("--no_mp4_record", action="store_true", default=False, help="only svo, not record mp4 video stream for data filter")
    parser.add_argument("--no_verbose", action="store_true", default=False)
    parser.add_argument("--no_camera", action="store_true", default=False)
    parser.add_argument("--mp4_crop_w", type=int, default=None)
    parser.add_argument("--mp4_crop_h", type=int, default=None)
    parser.add_argument("--mp4_downsample_ratio", type=int, default=2)
    parser.add_argument("--quest_obs_latency", type=float, default=0.0)
    args = parser.parse_args()
    if not os.path.isdir("data"):
        os.mkdir("data")
    main(args)
        
    

# python -m scripts.human_data_collection --output_dir data/data_human_raw --mp4_crop_w 640 --mp4_crop_h 480 --mp4_downsample_ratio 2