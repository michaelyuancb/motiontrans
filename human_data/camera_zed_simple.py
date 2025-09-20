# =================== No Multi-Processing Camera Zed Wrapper ===================
# =================== Could be adopted for both windows and linux ==============

import os
import imageio
import time
import numpy as np
import pyzed.sl as sl
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from common.precise_sleep import precise_wait



class CameraZedSimple():
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes

    def __init__(
            self, 
            device_id,
            camera_exposure=None,
            resolution=(1280, 720),
            capture_fps=30,
            num_threads=2,
            recording=False,
            recording_crop_w=None,
            recording_crop_h=None,
            recording_downsample_ratio=2,
            verbose=False
        ):
        super().__init__()
        
        self.device_id = device_id
        self.num_threads = num_threads
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.verbose = verbose
        self.recording = recording
        self.video_writer = None
        self.recording_crop_w = recording_crop_w
        self.recording_crop_h = recording_crop_h
        self.crop_pad_w = (self.resolution[0] - self.recording_crop_w) // 2 if self.recording_crop_w is not None else 0
        self.crop_pad_h = (self.resolution[1] - self.recording_crop_h) // 2 if self.recording_crop_h is not None else 0
        self.recording_downsample_ratio = int(recording_downsample_ratio)
        self.camera_exposure = camera_exposure
        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()

        threadpool_limits(self.num_threads)
        camera = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(int(self.device_id))
        z_res = self.get_cam_resolution()
        init_params.camera_resolution = z_res
        init_params.camera_fps = self.capture_fps
        init_params.camera_image_flip = sl.FLIP_MODE.OFF

        err = camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f'[SingleZed {self.device_id}] Main loop failed.')
            print("Zed Camera Open : "+repr(err)+". Exit program.")
            exit(1)
        
        if self.camera_exposure is not None:
            camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, self.camera_exposure)

        if self.verbose:
            print(f'[SingleZed {self.device_id}] Main loop started.')

        self.zed_runtime = sl.RuntimeParameters()
        self.camera = camera
        self.left_img = sl.Mat()
        self.zed_resolution = sl.Resolution(*self.resolution)

    
    @staticmethod
    def get_connected_devices_serial():

        try:
            serials = sl.Camera.get_device_list()
        except NameError:
            return []
        serials = [str(serial.serial_number) for serial in serials]
        serials = sorted(serials)
        print("Connected ZED camera serials:", serials)
        return serials


    def get_cam_resolution(self):
        if self.resolution[0] == 1280 and self.resolution[1] == 720:
            if self.capture_fps > 60:
                raise ValueError(f"Zed camera does not support need < 60 fps at 720p, now set to {self.capture_fps}")
            return sl.RESOLUTION.HD720 
        elif self.resolution[0] == 1920 and self.resolution[1] == 1080:
            if self.capture_fps > 30:
                raise ValueError(f"Zed camera does not support need < 30 fps at 1080p, now set to {self.capture_fps}")
            return sl.RESOLUTION.HD1080
        elif self.resolution[0] == 2208 and self.resolution[1] == 1242:
            if self.capture_fps > 15:
                raise ValueError(f"Zed camera does not support need < 15 fps at 2K, now set to {self.capture_fps}")
            return sl.RESOLUTION.HD2K
        else:
            raise ValueError(f"Zed camera does not support resolution {self.resolution}, please selected from [1280,720], ")


    def get_intrinsic_left_cam(self):
        camera_param = self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam 
        raw_intrinsic = np.array([
            [camera_param.fx, 0, camera_param.cx],
            [0, camera_param.fy, camera_param.cy],
            [0, 0, 1]
        ])
        return raw_intrinsic

    def get_intrinsic_left_dist(self):
        camera_param = self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam
        dist = np.array(list(camera_param.disto))
        return dist


    def start_recording(self, video_path: str):
        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        folder_path, _ = os.path.split(video_path)
        record_svo_fp = os.path.join(folder_path, "recording.svo")
        recordingParameters1 = sl.RecordingParameters(record_svo_fp, sl.SVO_COMPRESSION_MODE.H264)
        st = time.time()
        err = self.camera.enable_recording(recordingParameters1)
        # print(f"Enable recording time: {time.time()-st}")
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"ZED {self.device_id} Enable recording failed: {err}")
            return False
        if self.recording:
            self.video_writer = imageio.get_writer(video_path, format='ffmpeg', fps=self.capture_fps)
        print(f"ZED {self.device_id} Recording started.")
        return True


    def stop_recording(self):
        self.camera.disable_recording()
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        print(f"ZED {self.device_id} Recording stopped.")


    def recieve(self, transform=False):
        try:
            err = self.camera.grab(self.zed_runtime)
            if err != sl.ERROR_CODE.SUCCESS:
                print(f'[SingleZed {self.device_id}] fail to grab frame in reading frame.')
                return None
            self.camera.retrieve_image(self.left_img, sl.VIEW.LEFT, resolution=self.zed_resolution)
            img = self.left_img.get_data()[..., :3]
            if (self.recording) and ((self.video_writer is not None) or transform):
                if (self.recording_crop_w is not None) and (self.recording_crop_h is not None):
                    if self.crop_pad_h == 0 and self.crop_pad_w > 0:
                        img = img[:, self.crop_pad_w:-self.crop_pad_w]
                    elif self.crop_pad_w == 0 and self.crop_pad_h > 0:
                        img = img[self.crop_pad_h:-self.crop_pad_h]
                    elif self.crop_pad_h > 0 and self.crop_pad_w > 0:
                        img = img[self.crop_pad_h:-self.crop_pad_h, self.crop_pad_w:-self.crop_pad_w]
                img = img[::self.recording_downsample_ratio, ::self.recording_downsample_ratio, ::-1]
                if self.video_writer is not None:
                    self.video_writer.append_data(img)
            return img
        except Exception as e:
            print(f'[SingleZed {self.device_id}] Exception in reading frame: {e}') 
            return None 
