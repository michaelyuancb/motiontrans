# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
from pynput import keyboard
import click
import cv2
import qrcode
import time
import numpy as np
from collections import deque
from tqdm import tqdm
from multiprocessing.managers import SharedMemoryManager
# from real.camera_orbbec import CameraOrbbec
from real.recorder_rgbd_video import RecorderRGBDVideo
from common.cv2_util import optimal_row_cols, ImageDepthTransform, ImageDepthVisTransform
from real.camera_realsense import CameraRealSense
from real.multi_camera_visualizer import MultiCameraVisualizer
# from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from matplotlib import pyplot as plt
import pygame
import pygame.surfarray

from real.keystroke_counter import (KeystrokeCounter, Key, KeyCode)

max_obs_buffer_size = 60
camera_obs_latency = 0.125
bit_rate = 6000*1000
resolution = (1280, 720) # (640, 480)
obs_image_resolution = (640, 480)
cam_vis_resolution = (1920, 1080)
capture_fps = 30
video_paths = f"test.mp4"
num_threads = 14
get_time_budget = 0.2
bgr_to_rgb = True

# 初始化 Pygame
pygame.init()


camera_img_window = pygame.display.set_mode((1360, 720))  # 第二个窗口  # 78
pygame.display.set_caption("realsense")



# 将图像转换为 Pygame 可显示的格式
def cv2_to_pygame(cv_image):
    # OpenCV 是 BGR 格式，Pygame 是 RGB 格式，所以需要转换
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(cv_image.swapaxes(0, 1))  # 使用swapaxes交换轴


# %%
@click.command()
@click.option('-ci', '--camera_idx', type=int, default=0)
@click.option('-qs', '--qr_size', type=int, default=720)
@click.option('-n', '--n_frames', type=int, default=120)
def main(camera_idx, qr_size, n_frames):
    get_max_k = n_frames
    detector = cv2.QRCodeDetector()
    with SharedMemoryManager() as shm_manager, \
            KeystrokeCounter() as key_counter:

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
        cam_vis = MultiCameraVisualizer(
            camera=camera,
            rw=rw, rh=rh,
            row=row, col=col,
            rgb_to_bgr=False
        )
        print("Camera Initialization completed")

        camera.start(wait=False)


        if True:

            qr_latency_deque = deque(maxlen=get_max_k)
            qr_det_queue = deque(maxlen=get_max_k)
            data = None
            while True:
                t_start = time.time()
                data = camera.get(out=data)
                cam_img = data['rgb']

                code, corners, _ = detector.detectAndDecodeCurved(cam_img)
                color = (0, 0, 255)
                if len(code) > 0:
                    color = (0, 255, 0)
                    ts_qr = float(code)
                    ts_recv = data['camera_receive_timestamp']
                    latency = ts_recv - ts_qr
                    qr_det_queue.append(latency)
                else:
                    qr_det_queue.append(float('nan'))
                if corners is not None:
                    cv2.fillPoly(cam_img, corners.astype(np.int32), color)

                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_H,
                )
                t_sample = time.time()
                qr.add_data(str(t_sample))
                qr.make(fit=True)
                pil_img = qr.make_image()
                img = np.array(pil_img).astype(np.uint8) * 255
                img = np.repeat(img[:, :, None], 3, axis=-1)
                img = cv2.resize(img, (qr_size, qr_size), cv2.INTER_NEAREST)

                qrcode_pygame = cv2_to_pygame(img)

                # 在两个窗口中显示图像
                camera_img_window.blit(qrcode_pygame, (0, 0))  # 第一个窗口显示第一个图像


                t_show = time.time()
                qr_latency_deque.append(t_show - t_sample)
                print(f"data is {data['camera_receive_timestamp']}")

                if cam_img is not None and cam_img.size > 0:
                    cam_img_pygame = pygame.surfarray.make_surface(cam_img.swapaxes(0, 1))
                    camera_img_window.blit(cam_img_pygame, (720, 0))  # 第二个窗口显示第二个图像

                    # cv2.imshow('Camera', cam_img)
                    # cv2.pollKey()
                    # cv2.waitKey(1)  # 让窗口刷新
                else:
                    print("Warning: Empty or corrupted image received." * 10)

                t_end = time.time()
                pygame.display.update()  # 更新显示

                avg_latency = np.nanmean(qr_det_queue) - np.mean(qr_latency_deque)
                det_rate = 1 - np.mean(np.isnan(qr_det_queue))
                print("Running at {:.1f} FPS. Recv Latency: {:.3f}. Detection Rate: {:.2f}".format(
                    1 / (t_end - t_start),
                    avg_latency,
                    det_rate
                ))

                press_events = key_counter.get_press_events()
                break_flag = False
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='c'):
                        camera.stop()
                        break_flag = True
                        break
                    if key_stroke == KeyCode(char='q'):
                        camera.stop()
                        exit(0)
                if break_flag is True:
                    break
        data = camera.get(k=get_max_k)
        qr_recv_map = dict()
        for i in tqdm(range(len(data['camera_receive_timestamp']))):
            ts_recv = data['camera_receive_timestamp'][i]
            img = data['rgb'][i]
            code, corners, _ = detector.detectAndDecodeCurved(img)
            if len(code) > 0:
                ts_qr = float(code)
                if ts_qr not in qr_recv_map:
                    qr_recv_map[ts_qr] = ts_recv

        avg_qr_latency = np.mean(qr_latency_deque)
        t_offsets = [v - k - avg_qr_latency for k, v in qr_recv_map.items()]
        avg_latency = np.mean(t_offsets)
        std_latency = np.std(t_offsets)
        print(f'Capture to receive latency: AVG={avg_latency} STD={std_latency}')

        x = np.array(list(qr_recv_map.values()))
        y = np.array(list(qr_recv_map.keys()))
        y -= x[0]
        x -= x[0]
        plt.plot(x, x)
        plt.scatter(x, y)
        plt.title(f"Realsense Latency AVG{avg_latency:.5f} STD{std_latency:.5f}")
        plt.xlabel('Receive Timestamp (sec)')
        plt.ylabel('QR Timestamp (sec)')
        plt.savefig("../data_process/latency_analyse_fig/realsense_latency.png")

        plt.show()



# %%
if __name__ == "__main__":
    main()
