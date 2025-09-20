from typing import Optional, Callable, Dict
import os
import enum
import time
import json
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
import cv2
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from common.timestamp_accumulator import get_accumulate_timestamp_idxs
from real.shared_memory.shared_ndarray import SharedNDArray
from real.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from real.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from real.recorder_rgbd_video import RecorderRGBDVideo


class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4



class CameraRealSense(mp.Process):
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes

    def __init__(
            self, 
            shm_manager: SharedMemoryManager,
            device_id,
            get_time_budget=0.2,
            resolution=(1280, 720),
            capture_fps=30,
            put_fps=None,
            put_downsample=False,
            enable_depth=True,
            get_max_k=60,
            advanced_mode_config=None,
            receive_latency=0.0,
            num_threads=1,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            recorder: Optional[Callable[[Dict], Dict]] = None,
            enable_vis=False,
            verbose=False,
        ):
        super().__init__()

        self.device_id = device_id

        if put_fps is None:
            put_fps = capture_fps
        record_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        examples['rgb'] = np.empty(shape=shape+(3,), dtype=np.uint8)
        if enable_depth:
            examples['depth'] = np.empty(
                shape=shape, dtype=np.uint16)
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        if self.enable_vis:
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
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': rs.option.exposure.value,
            'option_value': 0.0,
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
        intrinsics_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(7,),
                dtype=np.float64)
        intrinsics_array.get()[:] = 0

        # create video recorder
        if recorder is None:
            recorder = RecorderRGBDVideo.create_h264(
                fps=record_fps, 
                codec='h264',
                input_pix_fmt='rgb24', 
                crf=18,
                thread_type='FRAME',
                thread_count=1)

        # copied variables
        self.num_threads = num_threads
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.receive_latency = receive_latency
        self.enable_depth = enable_depth
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        if vis_transform is None:
            self.vis_transform = transform
        else:
            self.vis_transform = vis_transform
        if recording_transform is None:
            self.recording_transform = transform
        else:
            self.recording_transform = recording_transform
        self.recorder = recorder
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        if self.enable_vis:
            self.vis_ring_buffer = vis_ring_buffer
        self.enable_vis = enable_vis
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array

        self.shm_manager = shm_manager
    
    @staticmethod
    def get_connected_devices_serial():
        serials = list()
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                if product_line == 'D400':
                    # only works with D400 series
                    serials.append(serial)
        serials = sorted(serials)
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
        data_example = {
            "color": np.empty(shape=shape+(3,), dtype=np.uint8),
        }
        if self.enable_depth:
            data_example['depth'] = np.empty(shape=shape, dtype=np.uint16)
        if self.recording_transform is not None:
            data_example = self.recording_transform(dict(data_example))
        data_example['shape'] = data_example['rgb'].shape
        self.recorder.start(
            shm_manager=self.shm_manager, 
            data_example=data_example)
        # must start video recorder first to create share memories
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        self.recorder.stop()
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
        self.recorder.start_wait()
    
    def end_wait(self):
        self.join()
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
        if self.enable_depth:
            vis_data = np.stack([vis_data['rgb'], vis_data['depth']])
        else:
            vis_data = vis_data['rgb'][None]
        return {"color": vis_data}
    
    # ========= user API ===========
    def set_color_option(self, option: rs.option, value: float):
        self.command_queue.put({
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': option.value,
            'option_value': value
        })
    
    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)
    
    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0,0] = fx
        mat[1,1] = fy
        mat[0,2] = ppx
        mat[1,2] = ppy
        return mat

    def get_depth_scale(self):
        assert self.ready_event.is_set()
        scale = self.intrinsics_array.get()[-1]
        return scale
        

    def get_camera_info(self):
        camera_info = dict()
        camera_info['intrinsic'] = self.get_intrinsics()
        camera_info['depth_scale'] = self.get_depth_scale()
        camera_info['height'] = self.intrinsics_array.get()[4]
        camera_info['width'] = self.intrinsics_array.get()[5]
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
        # cv2.setNumThreads(self.num_threads)  # FIXME: not sure why if it is set, it will stuck.

        w, h = self.resolution
        fps = self.capture_fps
        align = rs.align(rs.stream.color)
        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        if self.enable_depth:
            rs_config.enable_stream(rs.stream.depth,
                w, h, rs.format.z16, fps)

        # rs_config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)
        # if self.enable_depth:
        #     rs_config.enable_stream(rs.stream.depth,
        #         848, 480, rs.format.z16, 60)
        
        if True:
            rs_config.enable_device(self.device_id)

            # start pipeline
            pipeline = rs.pipeline()
            pipeline_profile = pipeline.start(rs_config)

            # report global time
            # https://github.com/IntelRealSense/librealsense/pull/3909
            d = pipeline_profile.get_device().first_color_sensor()
            d.set_option(rs.option.global_time_enabled, 1)

            # setup advanced mode
            if self.advanced_mode_config is not None:
                json_text = json.dumps(self.advanced_mode_config)
                device = pipeline_profile.get_device()
                advanced_mode = rs.rs400_advanced_mode(device)
                advanced_mode.load_json(json_text)

            # get
            color_stream = pipeline_profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            order = ['fx', 'fy', 'ppx', 'ppy', 'height', 'width']
            for i, name in enumerate(order):
                self.intrinsics_array.get()[i] = getattr(intr, name)

            if self.enable_depth:
                depth_sensor = pipeline_profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                self.intrinsics_array.get()[-1] = depth_scale
            
            # one-time setup (intrinsics etc, ignore for now)
            if self.verbose:
                print(f'[SingleRealsense {self.device_id}] Main loop started.')
        # except Exception as e:
        #     print(f"[SingleRealsense {self.device_id}] Error: {e}")
        #     self.recorder.stop()
        #     rs_config.disable_all_streams()
        #     self.stop_event.set()

        try:
            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():

                # wait for frames to come in
                frameset = pipeline.wait_for_frames()
                receive_time = time.time()
                # align frames to color
                frameset = align.process(frameset)

                data = dict()
                # grab data
                color_frame = frameset.get_color_frame()
                data['rgb'] = np.asarray(color_frame.get_data())  # bgr
                if self.enable_depth:
                    data['depth'] = np.asarray(frameset.get_depth_frame().get_data())
                
                t_recv = time.time()
                t_cap = color_frame.get_timestamp() / 1000  # realsense report in ms
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
                
                # record frame
                rec_data = data 
                if self.recording_transform == self.transform:
                    rec_data = put_data
                elif self.recording_transform is not None:
                    rec_data = self.recording_transform(dict(data))

                if self.recorder.is_ready():
                    self.recorder.write_frame(rec_data, frame_time=t_cal)

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency_real = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f'[Camera Realsense {self.device_id}] FPS {frequency_real}')

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
                    if cmd == Command.SET_COLOR_OPTION.value:
                        sensor = pipeline_profile.get_device().first_color_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
                        # print('auto', sensor.get_option(rs.option.enable_auto_exposure))
                        # print('exposure', sensor.get_option(rs.option.exposure))
                        # print('gain', sensor.get_option(rs.option.gain))
                    elif cmd == Command.SET_DEPTH_OPTION.value:
                        sensor = pipeline_profile.get_device().first_depth_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(command['video_path'])
                        start_time = command['recording_start_time']
                        if start_time < 0:
                            start_time = None
                        self.recorder.start_recording(video_path, start_time=start_time)
                    elif cmd == Command.STOP_RECORDING.value:
                        self.recorder.stop_recording()
                        # stop need to flush all in-flight frames to disk, which might take longer than dt.
                        # soft-reset put to drop frames to prevent ring buffer overflow.
                        put_idx = None
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']
                        # self.ring_buffer.clear()

                iter_idx += 1
        except KeyboardInterrupt:
            self.recorder.stop()
            rs_config.disable_all_streams()
            self.ready_event.set()
        finally:
            self.recorder.stop()
            rs_config.disable_all_streams()
            self.ready_event.set()
        
        if self.verbose:
            print(f'[Camera Realsense {self.device_id}] Exiting worker process.')
