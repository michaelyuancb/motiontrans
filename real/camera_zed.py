from typing import Optional, Callable, Dict
import os
import enum
import time
import json
import pyzed.sl as sl
import numpy as np
import multiprocessing as mp
import cv2
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from common.timestamp_accumulator import get_accumulate_timestamp_idxs
from real.shared_memory.shared_ndarray import SharedNDArray
from real.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from real.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from real.recorder_rgb_video import RecorderRGBVideo
from common.precise_sleep import precise_wait


class Command(enum.Enum):
    START_RECORDING = 0
    STOP_RECORDING = 1
    RESTART_PUT = 2



class CameraZed(mp.Process):
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes

    def __init__(
            self, 
            shm_manager: SharedMemoryManager,
            device_id,
            get_time_budget=0.2,
            resolution=(1280, 720),
            capture_fps=20,
            camera_exposure=None,           # 0 ~ 100, None is AUTO mode
            put_fps=None,
            put_downsample=False,
            enable_pointcloud=False,
            get_max_k=60,
            advanced_mode_config=None,
            receive_latency=0.0,
            num_threads=2,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            recorder: Optional[Callable[[Dict], Dict]] = None,
            enable_vis=False,
            verbose=False,
        ):
        super().__init__()

        self.device_id = device_id
        self.camera_exposure = camera_exposure

        if put_fps is None:
            put_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        examples['rgb'] = np.empty(shape=shape+(3,), dtype=np.uint8)
        examples['rgb_right'] = np.empty(shape=shape+(3,), dtype=np.uint16)
        if enable_pointcloud:
            examples['pointcloud'] = np.empty(shape=shape+(3,), dtype=np.float32)
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        if enable_vis is True:
            vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
                shm_manager=shm_manager,
                examples=examples if vis_transform is None 
                    else vis_transform(dict(examples)),
                get_max_k=1,
                get_time_budget=get_time_budget,
                put_desired_frequency=capture_fps
            )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None
                else transform(dict(examples)),
            get_max_k=10*get_max_k,
            get_time_budget=get_time_budget,
            put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.START_RECORDING.value,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
            'put_start_time': 0.0
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )

        # create shared array for intrinsics
        intrinsics_array_left = SharedNDArray.create_from_shape(mem_mgr=shm_manager,shape=(16,), dtype=np.float64)   # 4 + 12
        intrinsics_array_left.get()[:] = 0
        intrinsics_array_right = SharedNDArray.create_from_shape(mem_mgr=shm_manager,shape=(16,), dtype=np.float64)  # 4 + 12
        intrinsics_array_right.get()[:] = 0

        # if recorder is None:
        #     recorder = RecorderRGBVideo.create_h264(
        #         fps=capture_fps, 
        #         codec='h264',
        #         input_pix_fmt='rgb24', 
        #         crf=18,
        #         thread_type='FRAME',
        #         thread_count=1
        #     )

        # copied variables
        self.num_threads = num_threads
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.receive_latency = receive_latency
        self.enable_pointcloud = enable_pointcloud
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        if vis_transform is None:
            self.vis_transform = transform
        else:
            self.vis_transform = vis_transform
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        if enable_vis:
            self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue
        if recording_transform is None:
            self.recording_transform = transform
        else:
            self.recording_transform = recording_transform
        self.recorder = recorder
        self.intrinsics_array_left = intrinsics_array_left
        self.intrinsics_array_right = intrinsics_array_right
        self.enable_vis = enable_vis

        self.shm_manager = shm_manager
    
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


    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        shape = self.resolution[::-1]
        if self.recorder is not None:
            data_example = {
                "rgb": np.empty(shape=shape+(3,), dtype=np.uint8),
            }
            if self.enable_pointcloud:
                data_example["pcd"] = np.empty(shape=shape+(3,), dtype=np.float32)
            if self.recording_transform is not None:
                data_example = self.recording_transform(dict(data_example))
            data_example['shape'] = data_example['rgb'].shape
            self.recorder.start(
                shm_manager=self.shm_manager, 
                data_example=data_example)
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        if self.recorder is not None:
            self.recorder.stop()
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
        if self.recorder is not None:
            self.recorder.start_wait()
    
    def end_wait(self):
        self.join()
        if self.recorder is not None:
            self.recorder.end_wait()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)
    
    def get_vis(self, out=None):
        assert self.enable_vis is True
        vis_data = self.vis_ring_buffer.get(out=out)
        vis_data = np.stack([vis_data['rgb'], vis_data['rgb_right']])
        return {"rgb": vis_data}
    
    def clear_obs(self):
        self.ring_buffer.clear()
    
    # ========= user API ===========

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, ppx, ppy = self.intrinsics_array_left.get()[:4]
        mat = np.eye(3)
        mat[0,0] = fx
        mat[1,1] = fy
        mat[0,2] = ppx
        mat[1,2] = ppy
        mat_right = None 
        fx, fy, ppx, ppy = self.intrinsics_array_right.get()[:4]
        mat_right = np.eye(3)
        mat_right[0,0] = fx
        mat_right[1,1] = fy
        mat_right[0,2] = ppx
        mat_right[1,2] = ppy
        return mat, mat_right 


    def get_cam_dist(self):
        assert self.ready_event.is_set()
        dist = self.intrinsics_array_left.get()[4:]
        dist_right = None
        dist_right = self.intrinsics_array_right.get()[4:]
        return dist, dist_right

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
    
    def get_camera_info(self):
        camera_info = dict()
        intrinsc_left, intrinsic_right = self.get_intrinsics()
        camera_info['intrinsics_left'] = intrinsc_left 
        dist_left, dist_right = self.get_cam_dist()
        camera_info['distortion_left'] = dist_left
        camera_info['intrinsics_right'] = intrinsic_right  # (3, 3)
        camera_info['distortion_right'] = dist_right       # (12, )
        camera_info['height'] = self.resolution[1]
        camera_info['width'] = self.resolution[0]
        return camera_info

    
    def start_recording(self, video_path: str, start_time: float=-1):

        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'recording_start_time': start_time
        })
        
    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })
    
    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })
     
    # ========= interval API ===========
    def run(self):
        # limit threads
        threadpool_limits(self.num_threads)


        w, h = self.resolution
        fps = self.capture_fps
        dt = 1 / fps

        camera = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(int(self.device_id))
        zed_resolution = sl.Resolution(*self.resolution)
        z_res = self.get_cam_resolution()
        init_params.camera_resolution = z_res
        init_params.camera_fps = self.capture_fps
        init_params.camera_image_flip = sl.FLIP_MODE.OFF

        # Open the camera
        err = camera.open(init_params)
        left_img = sl.Mat()
        right_img = sl.Mat()
        left_pointcloud = sl.Mat()
        if err != sl.ERROR_CODE.SUCCESS:
            print(f'[SingleZed {self.device_id}] Main loop failed.')
            print("Zed Camera Open : "+repr(err)+". Exit program.")
            exit(1)
        if self.camera_exposure is not None:
            camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, self.camera_exposure)  
        
        if self.verbose:
            print(f'[SingleZed {self.device_id}] Main loop started.')

        zed_runtime = sl.RuntimeParameters()

        calib_params = camera.get_camera_information().camera_configuration.calibration_parameters
        left_cam_param = calib_params.left_cam 
        self.intrinsics_array_left.get()[0] = left_cam_param.fx 
        self.intrinsics_array_left.get()[1] = left_cam_param.fy
        self.intrinsics_array_left.get()[2] = left_cam_param.cx
        self.intrinsics_array_left.get()[3] = left_cam_param.cy 
        self.intrinsics_array_left.get()[4:] = np.array(list(left_cam_param.disto))
        right_cam_param = calib_params.right_cam
        self.intrinsics_array_right.get()[0] = right_cam_param.fx 
        self.intrinsics_array_right.get()[1] = right_cam_param.fy
        self.intrinsics_array_right.get()[2] = right_cam_param.cx
        self.intrinsics_array_right.get()[3] = right_cam_param.cy 
        self.intrinsics_array_right.get()[4:] = np.array(list(right_cam_param.disto))

        try:
            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():

                data = dict()
                success_read_flag = True

                err = camera.grab(zed_runtime)
                if err != sl.ERROR_CODE.SUCCESS:
                    print(f'[SingleZed {self.device_id}] fail to grab frame in reading frame.')
                    continue
                try:
                    received_time_zed = camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
                    camera.retrieve_image(left_img, sl.VIEW.LEFT, resolution=zed_resolution)
                    data['rgb'] = left_img.get_data()[..., :3]
                    camera.retrieve_image(right_img, sl.VIEW.RIGHT, resolution=zed_resolution)
                    data['rgb_right'] = right_img.get_data()[..., :3]
                    if self.enable_pointcloud:
                        camera.retrieve_measure(left_pointcloud, sl.MEASURE.XYZRGBA, resolution=zed_resolution)
                        data['pointcloud'] = left_pointcloud.get_data()[..., :3]
                except Exception as e:
                    success_read_flag = False
                    print(f'[SingleZed {self.device_id}] Exception in reading frame: {e}')
                
                if not success_read_flag:
                    continue
                
                t_recv = time.time()
                t_cap = received_time_zed / 1000.0 # zed report in ms
                t_cal = t_recv - self.receive_latency

                data['camera_receive_timestamp'] = t_recv
                data['camera_capture_timestamp'] = t_cap

                # apply transform
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                if self.put_downsample:                
                    # put frequency regulation
                    local_idxs, global_idxs, put_idx \
                        = get_accumulate_timestamp_idxs(
                            timestamps=[t_cal],
                            start_time=put_start_time,
                            dt=1/self.put_fps,
                            # this is non in first iteration
                            # and then replaced with a concrete number
                            next_global_idx=put_idx,
                            # continue to pump frames even if not started.
                            # start_time is simply used to align timestamps.
                            allow_negative=True
                        )

                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        put_data['timestamp'] = t_cal
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((t_cal - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = t_cal
                    self.ring_buffer.put(put_data, wait=False)

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()
                
                # put to vis
                if self.enable_vis is True:
                    vis_data = data
                    if self.vis_transform == self.transform:
                        vis_data = put_data
                    elif self.vis_transform is not None:
                        vis_data = self.vis_transform(dict(data))
                    self.vis_ring_buffer.put(vis_data, wait=False)

                rec_data = data 
                if self.recording_transform == self.transform:
                    rec_data = put_data
                elif self.recording_transform is not None:
                    rec_data = self.recording_transform(dict(data))
                
                if self.recorder is not None:
                    if self.recorder.is_ready():
                        self.recorder.write_frame(rec_data, frame_time=t_cal)

                # # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency_real = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f'[Camera Zed {self.device_id}] FPS {frequency_real}')

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.START_RECORDING.value:
                        video_path = str(command['video_path'])
                        start_time = command['recording_start_time']
                        if start_time < 0:
                            start_time = None
                        if self.recorder is not None:
                            self.recorder.start_recording(video_path, start_time=start_time)
                        record_svo_fp =os.path.join('/'.join(video_path.split('/')[:-1]), "recording.svo")
                        recordingParameters1 = sl.RecordingParameters(record_svo_fp, sl.SVO_COMPRESSION_MODE.H264)
                        err = camera.enable_recording(recordingParameters1)
                        if err != sl.ERROR_CODE.SUCCESS:
                            print(f"ZED {self.device_id} Enable recording failed: {err}")
                            exit(1)
                        print(f"ZED {self.device_id} Recording started.")
                    elif cmd == Command.STOP_RECORDING.value:
                        if self.recorder is not None:
                            self.recorder.stop_recording()
                        camera.disable_recording()
                        print(f"ZED {self.device_id} Recording stopped.")
                        put_idx = None
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']

                # t_wait_util = t_start + (iter_idx + 1) * dt
                # print(f"now: {time.time()}, wait until: {t_wait_util}")
                # precise_wait(t_wait_util, time_func=time.time)
                iter_idx += 1
        except KeyboardInterrupt:
            if self.recorder is not None:
                self.recorder.stop()
            camera.close()
            self.ready_event.set()
        finally:
            if self.recorder is not None:
                self.recorder.stop()
            camera.close()
            self.ready_event.set()
        
        if self.verbose:
            print(f'[Camera Zed {self.device_id}] Exiting worker process.')
