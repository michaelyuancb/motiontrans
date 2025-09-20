import socket
import time
from argparse import ArgumentParser
import numpy as np
import pdb
import socket
from scipy.spatial.transform import Rotation
import json
import cv2
from ip_config import *
import os
from human_data.quest_teleoperator import QuestTeleoperator
from dex_mimic.human_data.camera_realsense_simple import CameraLiteRealSense


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--frequency", type=int, default=20)
    parser.add_argument("--no_verbose", action="store_true", default=False)
    parser.add_argument("--no_camera", action="store_true", default=False)
    args = parser.parse_args()
    if not os.path.isdir("data"):
        os.mkdir("data")
    is_record = False

    camera = None
    if not args.no_camera:
        # from camera_wrapper.orbbec_wrapper import OrbbecAlignWrapper
        # camera = OrbbecAlignWrapper(align_mode='HW', enable_sync=True)
        # while camera.get_video_stream()[0] is None: pass
        device_id = CameraLiteRealSense.get_connected_devices_serial()[0]
        camera = CameraLiteRealSense(device_id, resolution=(1280, 720), capture_fps=30)
        print("Camera Initialization completed")

    quest = QuestTeleoperator(LOCAL_HOST, POSE_INFO_PORT, STREAMING_PORT)

    start_time = time.time() 
    fps_counter = 0
    packet_counter = 0
    print("Recorder Initialization completed")
    current_ts = time.time()
    video_writer, pose_list, log_writer = None, None, None

    while True:
        now = time.time()
        # TODO: May cause communication issues, need to tune on AR side.
        if now - current_ts < 1 / args.frequency: 
            continue
        else:
            current_ts = now
        try:
            
            if not args.no_camera:
                color_image, depth_image = camera.get_video_stream()
                if color_image is None:
                    continue
                assert color_image.shape[0] == 720 and color_image.shape[1] == 1280
                # from PIL import Image
                # color_image = np.array(Image.open("example_image.png"))
                quest.stream_image(color_image, verbose=not args.no_verbose)

            status, xrhand, head_pose = quest.receive(verbose=not args.no_verbose)
            if status == "Wait":                  # no data & command now
                continue
            if status == "Stream-Fail":           # stream fail in VR
                continue
            if status == "Start":                 # start recording
                pass
            elif (status == "Ensure"):            # ensuring
                pass
            elif (status == "Save") or (status == "Cancel"):  # ensure decision
                pass
            if (status == 'Data') and (head_pose is not None):     # data
                left_hand, right_hand = xrhand
                left_hand_str = left_hand.get_hand_pose_str()
                right_hand_str = right_hand.get_hand_pose_str()
                head_pose_str = head_pose.get_pose_str()
                pose_result = {"ts": str(time.time() - start_time), "left_hand": left_hand_str, "right_hand": right_hand_str, "head": head_pose_str}
                pass

        except socket.error as e:
            print(e)
            pass
        except KeyboardInterrupt:
            # if not args.no_camera:
            #     camera.close()
            quest.close()
            break
        except Exception as e:
            raise ValueError(e)
        