from typing import Optional, Callable, Dict
import enum
import time
import cv2
import weakref
import numpy as np
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from real.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from real.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from real.recorder_rgbd_video import RecorderRGBDVideo
from common.precise_sleep import precise_wait
from common.timestamp_accumulator import get_accumulate_timestamp_idxs

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode
    from real.orbbec_utils import frame_to_bgr_image
except Exception as e:
    print(f"CameraOrbbec Exception: {e}")
    print(f"Fail to load Orbbec Camera SDK")
    def frame_to_bgr_image(color_frame):
        return color_frame.get_data()


class VirtualOrbbecDepth:
    def __init__(self):
        self.data = np.random.randint(4096, size=(720, 1280)).astype(np.uint16)
        pass
    def get_width(self):
        return 1280
    def get_height(self):
        return 720
    def get_depth_scale(self):
        return 1.0
    def get_data(self):
        return self.data

class VirtualOrbbecColor:
    def __init__(self):
        self.data = np.random.randint(255, size=(720, 1280, 3)).astype(np.uint8)
        pass
    def get_data(self):
        return self.data

class VirtualOrbbecFrame:
    def __init__(self):
        self.color, self.depth = VirtualOrbbecColor(), VirtualOrbbecDepth()
    def get_color_frame(self):
        return self.color 
    def get_depth_frame(self):
        return self.depth

class VirtualOrbbecPipeline:
    def __init__(self):
        pass 
    def wait_for_frames(self, wait_time):
        time.sleep((wait_time-1) / 1000)
        return VirtualOrbbecFrame()
    def stop(self):
        pass



class Command(enum.Enum):
    RESTART_PUT = 0
    START_RECORDING = 1
    STOP_RECORDING = 2


class CameraOrbbec(mp.Process):
    """
    Orbbec Astra Camera Wrapper for Asynchronous Processing
    """
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes
    
    def __init__(
            self, 
            shm_manager: SharedMemoryManager,
            device_id,
            get_time_budget=0.2,
            resolution=(1280, 720),
            capture_fps=15,
            put_fps=None,
            put_downsample=True,
            get_max_k=30,
            receive_latency=0.0,
            num_threads=2,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            recorder: Optional[Callable[[Dict], Dict]] = None,
            verbose=False,
            debug=False,
        ):

        super().__init__()

        self.device_id = device_id

        if put_fps is None:
            put_fps = capture_fps
        
        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = {
            'rgb': np.empty(shape=shape+(3,), dtype=np.uint8),
            'depth': np.empty(shape=shape, dtype=np.uint16),
            'camera_capture_timestamp': 0.0,
            'camera_receive_timestamp': 0.0,
            'timestamp': 0.0,
            'step_idx': 0
        }

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
            get_max_k=get_max_k * 10,
            get_time_budget=get_time_budget,
            put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': 0.0,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )

        # create video recorder
        if recorder is None:
            # default to nvenc GPU encoder
            recorder = VideoRecorder.create_hevc_nvenc(
                shm_manager=shm_manager,
                fps=capture_fps, 
                input_pix_fmt='bgr24', 
                bit_rate=6000*1000)
        assert recorder.fps == capture_fps

        self.shm_manager = shm_manager 
        self.pipeline = None
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.receive_latency = receive_latency
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.recorder = recorder
        self.verbose = verbose
        self.put_start_time = None
        self.num_threads = num_threads

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue

        self.debug = debug


    # ========= context manager =========== (activate via "with command")
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):

        if put_start_time is None:
            put_start_time = time.time()
        self.put_start_time = put_start_time
        shape = self.resolution[::-1]
        data_example = {
            "color": np.empty(shape=shape+(3,), dtype=np.uint8),
            "depth": np.empty(shape=shape, dtype=np.uint16),
            "shape": shape+(3,),
        }
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
        vis_data = self.vis_ring_buffer.get(out=out)
        vis_data = np.stack([vis_data['rgb'], vis_data['depth']])
        return {"color": vis_data}

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

    # ========= orbbec sdk ============
    @staticmethod
    def get_video_stream(pipeline, wait_time=50):
        # FIXME: the smallest wait for frame is 10, further decrease may lead to some problem (original 100 for official demo) 
        frames = pipeline.wait_for_frames(wait_time)
        if frames is None:
            print(f"[Orbbec Camera {np.round(time.time())}]failed to get frames")
            return None, None
        color_frame = frames.get_color_frame()
        if color_frame is None:
            print(f"[Orbbec Camera {np.round(time.time())}] failed to get color frame")
            return None, None
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            print(f"[Orbbec Camera {np.round(time.time())}] failed to get depth frame")
            return None, None
        return color_frame, depth_frame
    
    @staticmethod
    def decode_color_frame(color_frame, out_buffer):
        color_image = frame_to_bgr_image(color_frame)
        out_buffer[:] = color_image[:, :, ::-1]

    @staticmethod
    def decode_depth_frame(depth_frame, out_buffer):
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
        out_buffer[:] = (depth_data.astype(np.float32) * scale).astype(np.uint16)

    # ========= interval API ===========

    def run(self):
        
        # limit threads
        threadpool_limits(self.num_threads)

        if self.debug is False:

            align_mode='HW'
            enable_sync=True

            # Orbbec Camera Setting
            pipeline = Pipeline()
            device = pipeline.get_device()
            device_info = device.get_device_info()
            device_pid = device_info.get_pid()
            config = Config()
            try:
                profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                color_profile = profile_list.get_default_video_stream_profile()
                profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                assert profile_list is not None
                depth_profile = profile_list.get_default_video_stream_profile()
                assert depth_profile is not None
                print("color profile : {}x{}@{}_{}".format(color_profile.get_width(),
                                                        color_profile.get_height(),
                                                        color_profile.get_fps(),
                                                        color_profile.get_format()))
                print("depth profile : {}x{}@{}_{}".format(depth_profile.get_width(),
                                                        depth_profile.get_height(),
                                                        depth_profile.get_fps(),
                                                        depth_profile.get_format()))
                config.enable_stream(color_profile)
                config.enable_stream(depth_profile)
            except Exception as e:
                print(e)
                exit(1)
            if align_mode == 'HW':
                if device_pid == 0x066B:
                    # Femto Mega does not support hardware D2C, and it is changed to software D2C
                    config.set_align_mode(OBAlignMode.SW_MODE)
                else:
                    config.set_align_mode(OBAlignMode.HW_MODE)
            elif align_mode == 'SW':
                config.set_align_mode(OBAlignMode.SW_MODE)
            else:
                config.set_align_mode(OBAlignMode.DISABLE)
            if enable_sync:
                try:
                    pipeline.enable_frame_sync()
                except Exception as e:
                    print(e)
            try:
                pipeline.start(config)
            except Exception as e:
                print(e)
                exit(1)

            if self.capture_fps > color_profile.get_fps():
                raise ValueError(f"The Orbbec camera set capture FPS is {self.capture_fps}. Should smaller than {color_profile.get_fps()}")
            if self.capture_fps > depth_profile.get_fps():
                raise ValueError(f"The Orbbec camera set capture FPS is {self.capture_fps}. Should smaller than {depth_profile.get_fps()}")

        else:
            pipeline = VirtualOrbbecPipeline()


        st = time.time()
        data_c, data_d = self.get_video_stream(pipeline)
        print(f"Try to get data from orbbec pipeline ...... time: {np.round(time.time() - st, 2)}.")
        while (data_c is None) or (data_d is None): 
            tm = time.time() - st 
            if tm > 5:
                print("Fail to get data from orbbec pipeline, Timeout 5 secs.")
                exit(1)
            data_c, data_d = self.get_video_stream(pipeline)
            if self.verbose:
                print(f"Try to get data from orbbec pipeline ...... time: {np.round(time.time() - st, 2)}.")
        print("Success to get data from orbbec pipeline !!!!! ")

        if self.verbose:
            print(f"CameraOrbbec [pid: {self.pid}]: run()")

        expect_dt = 1 / self.capture_fps

        try:

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            # reuse frame buffer
            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                ts = time.time()

                frame_rgb, frame_depth = self.get_video_stream(pipeline)   # non-decoded orbbec frame
                if (frame_rgb is None) or (frame_depth is None): continue
                assert frame_rgb is not None
                assert frame_depth is not None
                buffer = self.recorder.get_img_buffer()
                self.decode_color_frame(frame_rgb, buffer['rgb'])
                self.decode_depth_frame(frame_depth, buffer['depth'])
                t_recv = time.time()

                if self.debug is True:
                    t_cap = t_recv - 0.05
                else:
                    # FIXME: no monotinc() api for orbbec camera, please keep system time (us) setting stable.
                    t_cap = frame_rgb.get_system_timestamp_us() / 1000000

                t_cal = t_recv - self.receive_latency   # calibrated latency

                data = dict()
                data['camera_receive_timestamp'] = t_recv
                data['camera_capture_timestamp'] = t_cap
                data['rgb'] = buffer['rgb']
                data['depth'] = buffer['depth']

                # apply transform
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))
                
                # record frame
                rec_data = data
                if self.recording_transform == self.transform:
                    rec_data = put_data
                elif self.recording_transform is not None:
                    rec_data = self.recording_transform(dict(data))
                
                # record frame
                if self.recorder.is_ready():
                    self.recorder.write_img_buffer(rec_data, frame_time=t_cal)

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
                    # print(f"Put Frame: step_idx: {step_idx}, t_cal: {t_cal}")
                    self.ring_buffer.put(put_data, wait=False)

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()
                    
                # put to vis
                vis_data = data
                if self.vis_transform == self.transform:
                    vis_data = put_data
                elif self.vis_transform is not None:
                    vis_data = self.vis_transform(dict(data))
                self.vis_ring_buffer.put(vis_data, wait=False)

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f'[CameraOrbbec {self.pid}] Real FPS {frequency}')

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
                    if cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(command['video_path'])
                        start_time = command['recording_start_time']
                        if start_time < 0:
                            start_time = None
                        self.recorder.start_recording(video_path, start_time=start_time)
                    elif cmd == Command.STOP_RECORDING.value:
                        self.recorder.stop_recording()

                iter_idx += 1
                precise_wait(expect_dt - (time.time() - ts))
        except KeyboardInterrupt:
            self.recorder.stop()
            pipeline.stop()
        finally:
            self.recorder.stop()
            pipeline.stop()
