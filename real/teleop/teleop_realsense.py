import math
import numpy as np
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

np.set_printoptions(precision=2, suppress=True)
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations

import time
import cv2
from TeleVision import OpenTeleVision
# import pyzed.sl as sl
import pyrealsense2 as rs
from dynamixel.active_cam import DynamixelAgent
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore
from real.camera_realsense import CameraRealSense
from real.recorder_rgbd_video import RecorderRGBDVideo
from common.cv2_util import optimal_row_cols, ImageDepthTransform, ImageDepthVisTransform
from real.robot_active_vision import RobotActiveVision
from real.multi_camera_visualizer import MultiCameraVisualizer
import copy
import time
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import pygame
import pygame.surfarray
from real.keystroke_counter import (KeystrokeCounter, Key, KeyCode)



max_obs_buffer_size = 60
camera_obs_latency = 0.125
bit_rate = 6000*1000
resolution = (1280, 720) # (640, 480)
obs_image_resolution = (1280, 720)
cam_vis_resolution = (1920, 1080)
capture_fps = 30
video_paths = f"./recorded_data/test_video_data.mp4"
num_threads = 23
get_time_budget = 0.2
bgr_to_rgb = True

record_condition = ["Not Recording", "Is Recording"]

tv_resolution = (resolution[1], resolution[0])
crop_size_w = 1
crop_size_h = 0
resolution_cropped = (tv_resolution[0] - crop_size_h, tv_resolution[1] - 2 * crop_size_w)


# 初始化 Pygame
pygame.init()
camera_img_window = pygame.display.set_mode((1280, 720))  # 第二个窗口  # 78
pygame.display.set_caption("realsense")


img_shape = (resolution_cropped[0], 2 * resolution_cropped[1], 3)
img_height, img_width = resolution_cropped[:2]
shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)
image_queue = Queue()
toggle_streaming = Event()
tv = OpenTeleVision(resolution_cropped, shm.name, image_queue, toggle_streaming, cert_file=None, key_file=None, ngrok=False)

with (SharedMemoryManager() as shm_manager, \
        KeystrokeCounter() as key_counter):
    # compute resolution for vis
    rw, rh, col, row = optimal_row_cols(
        n_cameras=2,
        in_wh_ratio=4 / 3,
        max_resolution=cam_vis_resolution
    )

    transform = ImageDepthTransform(input_res=resolution, output_res=obs_image_resolution, bgr_to_rgb=bgr_to_rgb)
    vis_transform = ImageDepthVisTransform(input_res=resolution, output_res=(rw, rh), bgr_to_rgb=bgr_to_rgb)

    print("Transform Initialization completed")

    # TODO: use crea_hevc_nvenc to speedup the process.
    # video_depth_recorder = VideoDepthRecorder.create_hevc_nvenc(
    #     fps=capture_fps,
    #     input_pix_fmt='bgr24',
    #     bit_rate=bit_rate
    # )
    recorder_rgbd_video = RecorderRGBDVideo.create_h264(
        fps=capture_fps,
        input_pix_fmt='rgb24',
        bit_rate=bit_rate
    )

    print("VideoDepthRecorder Initialization completed")

    serial_number_list = CameraRealSense.get_connected_devices_serial()
    device_id = serial_number_list[0]

    camera = CameraRealSense(
        shm_manager=shm_manager,
        device_id=device_id,
        get_time_budget=get_time_budget,
        resolution=resolution,
        capture_fps=capture_fps,
        put_fps=None,
        put_downsample=False,
        get_max_k=max_obs_buffer_size,
        receive_latency=camera_obs_latency,
        num_threads=num_threads,
        transform=transform,
        vis_transform=vis_transform,
        recording_transform=None,  # will be set to transform by defaut
        recorder=recorder_rgbd_video,
        verbose=False,
    )

    print("Resolution for vis: ", rw, rh)
    camera.start(wait=False)
    time.sleep(2.0)
    print("start process. ")

    recording_flag = record_condition[0]
    episode_idx = 0

    while True:
        start = time.time()
        last_camera_data = None
        k = 4
        last_camera_data = camera.get(k=None, out=last_camera_data)
        color_img = last_camera_data['rgb']
        cv2.putText(
            color_img,
            f"Episode: {episode_idx}",
            (10, 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            lineType=cv2.LINE_AA,
            thickness=2,
            color=(0, 0, 0)
        )
        cv2.putText(
            color_img,
            f"Recording Condition: {recording_flag}",
            (10, 40),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            thickness=2,
            color=(255, 255, 0)
        )
        color_pygame = pygame.surfarray.make_surface(color_img.swapaxes(0, 1))

        camera_img_window.blit(color_pygame, (0, 0))  # 第一个窗口显示第一个图像
        pygame.display.update()  # 更新显示

        start_record = False
        end_record = False
        # print(camera.recorder.is_started)
        press_events = key_counter.get_press_events()
        for key_stroke in press_events:
            # print("is striked"*50)
            if key_stroke == KeyCode(char='s'):
                start_record = True
            if key_stroke == KeyCode(char='e'):
                end_record = True

        if recording_flag is record_condition[1]:

            if end_record is True:
                # print("end"*100)
                # start_time = time.time()
                # camera.restart_put(start_time=start_time)
                camera.stop_recording()
                recording_flag = record_condition[0]

        if recording_flag is record_condition[0]:
            if start_record is True:
                episode_idx = episode_idx + 1
                start_time = time.time()
                camera.restart_put(start_time=start_time)
                camera.start_recording(video_path=video_paths, start_time=start_time)
                recording_flag = record_condition[1]





        bgr = np.hstack((color_img[crop_size_h:, crop_size_w:-crop_size_w],
                         color_img[crop_size_h:, crop_size_w:-crop_size_w]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)

        np.copyto(img_array, rgb)

        end = time.time()
        # print(1/(end-start))
