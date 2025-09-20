from typing import Optional, List
import pathlib
import numpy as np
import time
import shutil
import math
import os
import json
import pickle 
from multiprocessing.managers import SharedMemoryManager
from common.precise_sleep import precise_wait
from real.robot_franka import RobotFranka
from real.robot_inspire_hand import RobotInspireHand, inspire_hand_primitives
from real.robot_active_vision import RobotActiveVision
try:
    from real.camera_realsense import CameraRealSense
except Exception as e:
    print("RealSense not found, exception: ", e)
    CameraRealSense = None
from real.camera_zed import CameraZed  
from real.recorder_rgb_video import RecorderRGBVideo 
from real.recorder_rgbd_video import RecorderRGBDVideo 
from common.timestamp_accumulator import TimestampActionAccumulator, ObsAccumulator
from real.multi_camera_visualizer import MultiCameraVisualizer

from common.cv2_util import optimal_row_cols, ImageDepthTransform, ImageTransform
from common.interpolation_util import get_interp1d, PoseInterpolator

SAVE_ACTION = os.environ.get('SAVE_ACTION', 'False').lower() in ['true']

if SAVE_ACTION:
    if os.path.exists('actions.txt'):
        os.remove('actions.txt')
    action_write_fp = open('actions.txt', 'w')


class ControllerRobotSystem:
    def __init__(self, 
            # required params
            output_dir,
            cameras_config,
            robots_config,
            grippers_config, 
            active_visions_config, 
            # obs
            frequency=20,
            obs_image_resize_resolution=(640, 480),
            obs_image_resolution=(640, 480),
            camera_exposure=None,
            enable_pointcloud=False,
            max_obs_buffer_size=60,
            down_sample_steps=1,            # downsample steps (relative to frequency)
            # all in steps (relative to frequency)
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            active_vision_obs_horizon=2,
            # action
            init_joints=False,
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(1920, 1080),
            verbose=False,
            # shared memory
            shm_manager=None
            ):

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        ############################################# Camera Initialization #############################################
        assert len(cameras_config) == 1
        camera_config = cameras_config[0]
        if ('enable_depth' in camera_config) and (camera_config['enable_depth'] is True):
            vis_n_cam = 2 
        elif ('enable_stereo' in camera_config) and (camera_config['enable_stereo'] is True):
            vis_n_cam = 2
        else:
            vis_n_cam = 1
        rw, rh, col, row = optimal_row_cols(
            n_cameras=vis_n_cam,
            in_wh_ratio=4/3,
            max_resolution=multi_cam_vis_resolution
        )
        bit_rate = 6000*1000
        self.camera_exposure = camera_exposure

        if camera_config['camera_type'].startswith('realsense'):
            resolution = camera_config['resolution']
            capture_fps = camera_config['capture_fps']
            if camera_config['enable_depth'] is True:
                transform = ImageDepthTransform(input_res=resolution, resize_res=obs_image_resize_resolution, output_res=obs_image_resolution, bgr_to_rgb=True) # realsense return bgr24
            else:
                raise NotImplementedError("Please set enable_depth of the Camera RealSense to True.")
            # vis_transform = ImageDepthVisTransform(input_res=resolution, resize_res=obs_image_resize_resolution, output_res=(rw, rh), bgr_to_rgb=True)
            vis_transform = None
            recorder_rgbd_video = RecorderRGBDVideo.create_h264(
                fps=capture_fps,
                input_pix_fmt='rgb24',
                bit_rate=bit_rate
            )
            camera = CameraRealSense(
                shm_manager=shm_manager,
                device_id=camera_config['device_id'],
                resolution=camera_config['resolution'],
                capture_fps=camera_config['capture_fps'],
                enable_depth=True,
                get_max_k=max_obs_buffer_size,
                receive_latency=camera_config['camera_obs_latency'],
                transform=transform,
                vis_transform=vis_transform,
                recording_transform=None,         # will be set to transform by defaut
                recorder=recorder_rgbd_video,
                enable_vis=enable_multi_cam_vis,
                verbose=verbose,
            )
        elif camera_config['camera_type'].startswith('zed'):
            resolution = camera_config['resolution']
            capture_fps = camera_config['capture_fps']
            if enable_pointcloud:
                transform = ImageDepthTransform(input_res=resolution, resize_res=obs_image_resize_resolution, output_res=obs_image_resolution, bgr_to_rgb=True) 
            else:
                transform = ImageTransform(input_res=resolution, resize_res=obs_image_resize_resolution, output_res=obs_image_resolution, bgr_to_rgb=True) 
            if enable_multi_cam_vis:
                vis_transform = ImageTransform(input_res=resolution, resize_res=obs_image_resize_resolution, output_res=(rw, rh), bgr_to_rgb=True)
            else:
                vis_transform = None
              
            recorder_rgb_video = RecorderRGBVideo.create_h264(
                fps=capture_fps,
                input_pix_fmt='rgb24',
                bit_rate=bit_rate
            )
            camera = CameraZed(
                shm_manager=shm_manager,
                device_id=camera_config['device_id'],
                resolution=camera_config['resolution'],
                capture_fps=camera_config['capture_fps'],
                camera_exposure=camera_exposure,
                enable_pointcloud=enable_pointcloud,
                get_max_k=max_obs_buffer_size,
                receive_latency=camera_config['camera_obs_latency'],
                transform=transform,
                vis_transform=vis_transform,
                recording_transform=None,         # will be set to transform by defaut
                recorder=recorder_rgb_video,
                enable_vis=enable_multi_cam_vis,
                verbose=verbose,
            )
        else:
            raise NotImplementedError("Please set camera_type to realsense.")

        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                camera=camera,
                rw=rw, rh=rh,
                row=row, col=col,
                rgb_to_bgr=False
            )
        print("[Controller] Camera intialization finished.")

        ############################################# Franka & Gripper Initialization #############################################

        # assert len(robots_config) == len(grippers_config)
        robots = list()
        grippers = list()
        active_visions = list()
        if robots_config is not None:
            for rc in robots_config:
                if rc['robot_type'].startswith('franka'):
                    this_robot = RobotFranka(
                        shm_manager=shm_manager,
                        robot_ip=rc['robot_ip'],
                        frequency=rc['robot_frequency'],
                        Kx_scale=1.0,
                        Kxd_scale=np.array([2.0,1.5,2.0,1.0,1.0,1.0]),
                        joints_init=rc['joints_init'] if init_joints else None,
                        joints_init_duration=2.0,
                        verbose=verbose,
                        receive_latency=rc['robot_obs_latency'],
                    )
                else:
                    raise NotImplementedError()
                robots.append(this_robot)

        print("[Controller] Robot intialization finished.")

        self.robot_gripper_primitives = None
        for gc in grippers_config:
            if gc['gripper_type'].startswith('inspire'):
                this_gripper = RobotInspireHand(
                    shm_manager=shm_manager,
                    port=gc['gripper_port'],
                    unit="radians",
                    frequency=gc['gripper_frequency'],
                    verbose=verbose
                )
                self.robot_gripper_primitives = inspire_hand_primitives
            grippers.append(this_gripper)
            
        print("[Controller] Gripper intialization finished.")
        
        for avc in active_visions_config:
            if avc['active_vision_type'].startswith('open_television'):
                this_active_vision = RobotActiveVision(
                    shm_manager=shm_manager,
                    port=avc['active_vision_port'],
                    frequency=avc['active_vision_frequency'],
                    verbose=verbose
                )
            active_visions.append(this_active_vision)
        

        print("[Controller] Actuve Vision intialization finished.")

        ############################################# System Initialization #############################################

        self.camera = camera
        self.camera_config = camera_config
        self.multi_cam_vis = multi_cam_vis
        
        self.robots = robots
        self.robots_config = robots_config
        self.grippers = grippers
        self.grippers_config = grippers_config
        self.active_visions = active_visions
        self.active_visions_config = active_visions_config 

        self.frequency = frequency
        self.max_obs_buffer_size = max_obs_buffer_size
        self.camera_capture_frequency = camera_config['capture_fps']

        # timing
        self.camera_obs_latency = camera_config['camera_obs_latency']
        self.down_sample_steps = down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        self.active_vision_obs_horizon = active_vision_obs_horizon
        # recording
        self.output_dir = output_dir
        self.video_dir = None
        self.save_dir = None
        # temp memory buffers
        self.last_camera_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None

        self.start_time = None
        self.last_time_step = 0

        ############################################ Result Saving ########################################
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        self.episode_id = -1

        assert len(self.robots) == len(self.grippers)
    
    def update_save_dir(self):
        self.episode_id = self.episode_id + 1 
        uid_time = time.strftime(f'%Y-%m-%d-%H-%M-%S-EP{self.episode_id}', time.localtime(time.time()))
        self.save_dir = os.path.join(self.output_dir, uid_time)
        os.makedirs(self.save_dir, exist_ok=True)
        self.video_dir = os.path.join(self.save_dir, 'videos')
        os.makedirs(self.video_dir, exist_ok=True)
    
    def delete_save_dir(self):
        self.episode_id = self.episode_id - 1 
        shutil.rmtree(self.save_dir)

    # ======== start-stop API =============
    @property
    def is_ready(self):
        ready_flag = self.camera.is_ready
        for robot in self.robots:
            ready_flag = ready_flag and robot.is_ready
            # print(f"robot: {robot.is_ready}")
        for gripper in self.grippers:
            ready_flag = ready_flag and gripper.is_ready
            # print(f"gripper: {gripper.is_ready}")
        for active_vision in self.active_visions:
            ready_flag = ready_flag and active_vision.is_ready
            # print(f"active_vision: {active_vision.is_ready}")
        return ready_flag
    
    def start(self, wait=True):
        self.camera.start(wait=False)
        for robot in self.robots:
            robot.start(wait=False)
        for gripper in self.grippers:
            gripper.start(wait=False)
        for active_vision in self.active_visions:
            active_vision.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        for robot in self.robots:
            robot.stop(wait=False)
        for gripper in self.grippers:
            gripper.stop(wait=False)
        for active_vision in self.active_visions:
            active_vision.stop(wait=False)
        self.camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.camera.start_wait()
        for robot in self.robots:
            robot.start_wait()
        for gripper in self.grippers:
            gripper.start_wait()
        for active_vision in self.active_visions:
            active_vision.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()
    
    def stop_wait(self):
        for robot in self.robots:
            robot.stop_wait()
        for gripper in self.grippers:
            gripper.stop_wait()
        for active_vision in self.active_visions:
            active_vision.stop_wait()
        self.camera.end_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def clear_obs(self):
        self.camera.clear_obs()
        for robot in self.robots:
            robot.clear_obs()
        for gripper in self.grippers:
            gripper.clear_obs()
        for active_vision in self.active_visions:
            active_vision.clear_obs()
            

    def get_obs(self) -> dict:
        """
        Timestamp alignment policy  (No Multi-Camera Support Yet)
        We assume the cameras used for obs are always [0, k - 1], where k is the number of robots
        All other cameras, find corresponding frame with the nearest timestamp
        All low-dim observations, interpolate with respect to 'current' time
        """

        assert self.is_ready

        # get data
        if type(self.down_sample_steps) is int:
            windows_size = self.camera_obs_horizon * self.down_sample_steps
        else:
            if self.camera_obs_horizon == 1:
                windows_size = self.down_sample_steps[0]
            else:
                windows_size = max(self.down_sample_steps[:self.camera_obs_horizon-1]) + 1
        # print(self.frequency, windows_size)
        k = math.ceil(windows_size * (self.camera_capture_frequency / self.frequency)) + 2 # here 4 is adjustable, typically 1 should be enough
        self.last_camera_data = self.camera.get(k=k, out=self.last_camera_data)

        # both have more than n_obs_steps data
        last_robots_data = list()
        last_grippers_data = list()
        last_active_visions_data = list()
        for robot in self.robots:
            last_robots_data.append(robot.get_all_state())
        for gripper in self.grippers:
            last_grippers_data.append(gripper.get_all_state())
        for active_vision in self.active_visions:
            last_active_visions_data.append(active_vision.get_all_state())

        last_timestamp = self.last_camera_data['timestamp'][-1]
        dt = 1 / self.frequency  # frequency = 10

        # align camera obs timestamps
        if type(self.down_sample_steps) == int:
            camera_obs_timestamps = last_timestamp - (np.arange(self.camera_obs_horizon)[::-1] * self.down_sample_steps * dt)
        else:
            camera_obs_timestamps = last_timestamp - np.array([0] + self.down_sample_steps)[::-1] * dt
            camera_obs_timestamps = camera_obs_timestamps[-self.camera_obs_horizon:]

        camera_obs = dict()
        this_timestamps = self.last_camera_data['timestamp']      # camera_idx这个相机过去k个时间辍
        this_idxs = list()
        for t in camera_obs_timestamps:
            nn_idx = np.argmin(np.abs(this_timestamps - t))
            this_idxs.append(nn_idx)
        for k in self.last_camera_data.keys():
            if k in ['camera_capture_timestamp', 'camera_receive_timestamp', 'timestamp', 'step_idx']:
                continue
            camera_obs["camera0_" + k] = self.last_camera_data[k][this_idxs]

        # obs_data to return (it only includes camera data at this stage)
        obs_data = dict(camera_obs)

        # include camera timesteps
        obs_data['timestamp'] = camera_obs_timestamps

        # align robot obs
        if type(self.down_sample_steps) == int:
            robot_obs_timestamps = last_timestamp - (np.arange(self.robot_obs_horizon)[::-1] * self.down_sample_steps * dt)
        else:
            robot_obs_timestamps = last_timestamp - np.array([0] + self.down_sample_steps)[::-1] * dt
            robot_obs_timestamps = robot_obs_timestamps[-self.robot_obs_horizon:]

        for robot_idx, last_robot_data in enumerate(last_robots_data):
            robot_pose_interpolator = PoseInterpolator(
                t=last_robot_data['robot_timestamp'], 
                x=last_robot_data['ActualTCPPose'])
            robot_pose = robot_pose_interpolator(robot_obs_timestamps)
            robot_obs = {
                f'robot{robot_idx}_eef_pos': robot_pose[...,:3],
                f'robot{robot_idx}_eef_rot_axis_angle': robot_pose[...,3:]
            }
            # update obs_data
            obs_data.update(robot_obs)

        # align gripper obs
        if type(self.down_sample_steps) == int:
            gripper_obs_timestamps = last_timestamp - (np.arange(self.gripper_obs_horizon)[::-1] * self.down_sample_steps * dt)
        else:
            gripper_obs_timestamps = last_timestamp - np.array([0] + self.down_sample_steps)[::-1] * dt
            gripper_obs_timestamps = gripper_obs_timestamps[-self.gripper_obs_horizon:]
        for gripper_idx, last_gripper_data in enumerate(last_grippers_data):
            # align gripper obs
            gripper_interpolator = get_interp1d(
                t=last_gripper_data['gripper_timestamp'],
                x=last_gripper_data['gripper_pose']
            )
            gripper_interpolator_force = get_interp1d(
                t=last_gripper_data['gripper_timestamp'],
                x=last_gripper_data['gripper_force']
            )
            gripper_obs = {
                f'gripper{gripper_idx}_gripper_pose': gripper_interpolator(gripper_obs_timestamps),
                f'gripper{gripper_idx}_gripper_force': gripper_interpolator_force(gripper_obs_timestamps),
            }
            # update obs_data
            obs_data.update(gripper_obs)
        
        # align active vision obs
        if type(self.down_sample_steps) == int:
            active_vision_obs_timestamps = last_timestamp - (np.arange(self.active_vision_obs_horizon)[::-1] * self.down_sample_steps * dt)
        else:
            active_vision_obs_timestamps = last_timestamp - np.array([0] + self.down_sample_steps)[::-1] * dt
            active_vision_obs_timestamps = active_vision_obs_timestamps[-self.active_vision_obs_horizon:]
        for active_vision_idx, last_active_vision_data in enumerate(last_active_visions_data):
            active_vision_pose_interpolator = PoseInterpolator(
                t=last_active_vision_data['robot_timestamp'],
                x=last_active_vision_data['ActualTCPPose'])
            active_vision_pose = active_vision_pose_interpolator(active_vision_obs_timestamps)
            active_vision_obs = {
                f'active_vision{active_vision_idx}_pose': active_vision_pose
            }
            # update obs_data
            obs_data.update(active_vision_obs)
        
        # accumulate obs
        if self.obs_accumulator is not None:
            for robot_idx, last_robot_data in enumerate(last_robots_data):
                self.obs_accumulator.put(
                    data={
                        f'robot{robot_idx}_eef_pose': last_robot_data['ActualTCPPose'],
                        f'robot{robot_idx}_joint_pos': last_robot_data['ActualQ'],
                        f'robot{robot_idx}_joint_vel': last_robot_data['ActualQd'],
                    },
                    timestamps=last_robot_data['robot_timestamp']
                )
            for gripper_idx, last_gripper_data in enumerate(last_grippers_data):
                self.obs_accumulator.put(
                    data={
                        f'gripper{gripper_idx}_gripper_pose': last_gripper_data['gripper_pose'],
                        f'gripper{gripper_idx}_gripper_force': last_gripper_data['gripper_force'],
                    },
                    timestamps=last_gripper_data['gripper_timestamp']
                )
            for active_vision_idx, last_active_vision_data in enumerate(last_active_visions_data):
                self.obs_accumulator.put(
                    data={
                        f'active_vision{active_vision_idx}_rot_axis_angle': last_active_vision_data['ActualTCPPose'][..., -self.active_visions_config[active_vision_idx]['action_dim']:]
                    },
                    timestamps=last_active_vision_data['robot_timestamp']
                )

        return obs_data
    
    def exec_actions(self,                  # order for input and execut: robot ---> gripper ---> active_vision
                     robot_actions,         # (T, Dim_SUM) with r-g-a order. or {"robot": List[(T, Dim)], "gripper": List[(T, Dim)], "active_vision": List[(T, Dim)]}
                     timestamps: np.ndarray,
                    #  gestures: np.ndarray=None,
                     compensate_latency=False,
                     obs=None, 
                     ignore_t=0,
                     hand_schedule_length=-1):

        assert self.is_ready
        assert timestamps.ndim == 1
        if isinstance(robot_actions, dict):
            action_array = []
            for key in robot_actions.keys():
                for i in range(len(robot_actions[key])):
                    action_array.append(robot_actions[key][i])
            robot_actions = np.concatenate(action_array, axis=-1)

        if not isinstance(robot_actions, np.ndarray):
            robot_actions = np.array(robot_actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        n_total_action_dim = 0
        for i in range(len(self.robots_config)):
            self.robots_config[i]['action_dim_real'] = [n_total_action_dim, n_total_action_dim + self.robots_config[i]['action_dim']]
            n_total_action_dim = n_total_action_dim + self.robots_config[i]['action_dim']
            self.grippers_config[i]['action_dim_real'] = [n_total_action_dim, n_total_action_dim + self.grippers_config[i]['action_dim']]
            n_total_action_dim = n_total_action_dim + self.grippers_config[i]['action_dim']

        for cfgs in [self.active_visions_config]:
            for cfg_device in cfgs:
                cfg_device['action_dim_real'] = [n_total_action_dim, n_total_action_dim + cfg_device['action_dim']]
                n_total_action_dim = n_total_action_dim + cfg_device['action_dim']

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time

        start_t = np.sum(is_new == 0)
        start_t = max(ignore_t, start_t)
        new_actions = robot_actions[start_t:]
        # new_gestures = gestures[is_new]
        new_timestamps = timestamps[start_t:]

        assert new_actions.shape[1] == n_total_action_dim
        if SAVE_ACTION:
            # import pdb; pdb.set_trace()
            n_t = len(new_actions)
            if obs is not None:
                for i in range(len(obs['robot0_eef_pos'])):
                    obs_txt = '[O] ' + str(receive_time) + "|" + str(obs['robot0_eef_pos'][i])
                    action_write_fp.write(obs_txt + '\n')
            if n_t > 0:
                for i in range(n_t):
                    action = new_actions[i][:3]
                    action = str(action)
                    action = '[A] ' + str(receive_time) + "|" + action
                    # action_write_fp
                    action_write_fp.write(action + '\n')

            # flush file
            action_write_fp.flush()

        for i in range(len(new_actions)):
            if len(self.robots) > 0:
                for robot_idx, (robot, rc) in enumerate(zip(self.robots, self.robots_config)):
                    r_latency = rc['robot_action_latency'] if compensate_latency else 0.0
                    robot.schedule_waypoint(
                        pose=new_actions[i, rc['action_dim_real'][0]: rc['action_dim_real'][1]],
                        target_time=new_timestamps[i] - r_latency
                    )
            if len(self.grippers) > 0:
                for gripper_idx, (gripper, gc) in enumerate(zip(self.grippers, self.grippers_config)):
                    if hand_schedule_length != -1:
                        if i >= hand_schedule_length - start_t:
                            continue
                    g_latency = gc['gripper_action_latency'] if compensate_latency else 0.0
                    # ges = int(new_gestures[i][gripper_idx])
                    # if ges != -1:
                    #     gripper_action = np.array(self.robot_gripper_primitives[ges])
                    # else:
                    #     gripper_action = new_actions[i, gc['action_dim_real'][0]: gc['action_dim_real'][1]]
                    gripper_action = new_actions[i, gc['action_dim_real'][0]: gc['action_dim_real'][1]]
                    gripper.schedule_waypoint(
                        pose=gripper_action,
                        target_time=new_timestamps[i] - g_latency
                    )
            if len(self.active_visions) > 0:
                for active_vision_idx, (active_vision, avc) in enumerate(zip(self.active_visions, self.active_visions_config)):
                    av_latency = avc['active_vision_action_latency'] if compensate_latency else 0.0
                    active_vision.schedule_waypoint(
                        pose=new_actions[i, avc['action_dim_real'][0]: avc['action_dim_real'][1]],
                        target_time=new_timestamps[i] - av_latency
                    )

        # new_actions = np.concatenate([new_actions, new_gestures], axis=-1)

        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )


    def exec_robot(self,                  # order for input and execut: robot ---> gripper ---> active_vision
                   robot_actions,         # (T, Dim_SUM) with
                   timestamps: np.ndarray,
                   compensate_latency=False,
                   ignore_t=0):

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        start_t = np.sum(is_new == 0)
        start_t = max(ignore_t, start_t)
        new_actions = robot_actions[start_t:]
        new_timestamps = timestamps[start_t:]

        for i in range(len(new_actions)):
            n_total_action_dim = 0
            if len(self.robots) > 0:
                for robot_idx, (robot, rc) in enumerate(zip(self.robots, self.robots_config)):
                    r_latency = rc['robot_action_latency'] if compensate_latency else 0.0
                    robot.schedule_waypoint(
                        pose=new_actions[i][n_total_action_dim: n_total_action_dim + self.robots_config[robot_idx]['action_dim']],
                        target_time=new_timestamps[i] - r_latency
                    )
                    n_total_action_dim =n_total_action_dim + self.robots_config[robot_idx]['action_dim']

        new_actions = np.concatenate([new_actions, np.zeros((len(new_actions), 6))], axis=1)
        # record actions (only wrists)
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )


    def exec_gripper(self,                  # order for input and execut: robot ---> gripper ---> active_vision
                     robot_actions,         # (T, Dim_SUM) with
                     timestamps: np.ndarray,
                     compensate_latency=False,
                     ignore_t=0):

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        start_t = np.sum(is_new == 0)
        start_t = max(ignore_t, start_t)
        new_actions = robot_actions[start_t:]
        new_timestamps = timestamps[start_t:]

        for i in range(len(new_actions)):
            n_total_action_dim = 0
            if len(self.grippers) > 0:
                for gripper_idx, (gripper, gc) in enumerate(zip(self.grippers, self.grippers_config)):
                    g_latency = gc['gripper_action_latency'] if compensate_latency else 0.0
                    gripper_action = new_actions[i][n_total_action_dim: n_total_action_dim + self.robots_config[gripper_idx]['action_dim']]
                    gripper.schedule_waypoint(
                        pose=gripper_action,
                        target_time=new_timestamps[i] - g_latency
                    )
                    n_total_action_dim = n_total_action_dim + self.robots_config[gripper_idx]['action_dim']

    
    def get_robot_state(self):
        return [robot.get_state() for robot in self.robots]
    
    def get_gripper_state(self):
        return [gripper.get_state() for gripper in self.grippers]

    def get_active_vision_state(self):
        return [active_vision.get_state() for active_vision in self.active_visions]

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        self.update_save_dir()
        # save config to yaml file
        episode_config = {
            "cameras": [self.camera_config],
            "robots": self.robots_config,
            "grippers": self.grippers_config,
            "active_visions": self.active_visions_config,
        } 
        with open(os.path.join(self.save_dir, 'episode_config.yaml'), 'w') as f:
            json.dump(episode_config, f, indent=4)

        if self.camera_config['camera_type'].startswith('realsense'):
            camear_info = self.camera.get_camera_info()
            with open(os.path.join(self.save_dir, 'camera_info.pkl'), 'wb') as f:
                pickle.dump(camear_info, f)
        elif self.camera_config['camera_type'].startswith('zed'):
            pass 
        else:
            raise NotImplementedError("Please set camera_type to zed (recommend) or realsense.")
        os.makedirs(os.path.join(os.path.join(self.video_dir)), exist_ok=True)
        video_paths_only_camera = os.path.join(self.video_dir, "rgb.mp4")
        video_paths_only_camera = pathlib.Path(video_paths_only_camera).absolute()
        
        # start recording on camera
        self.camera.restart_put(start_time=start_time)
        self.camera.start_recording(video_path=str(video_paths_only_camera), start_time=start_time)
        
        # create accumulators
        self.obs_accumulator = ObsAccumulator()
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        
    
    def end_episode(self, inference_mode=False):
        "Stop recording"
        assert self.is_ready
        
        # stop video recorder
        self.camera.stop_recording()
        for robot in self.robots:
            if robot is not None:
                robot.clear_queue()
        for gripper in self.grippers:
            if gripper is not None:
                gripper.clear_queue()

        if (self.obs_accumulator is not None) and (inference_mode is False):
            # recording
            assert self.action_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            end_time = float('inf')
            for key, value in self.obs_accumulator.timestamps.items():
                # print(f"end_time is :{end_time}")
                # print(f"value is {value}")
                if len(value) != 0:
                    end_time = min(end_time, value[-1])
            end_time = min(end_time, self.action_accumulator.timestamps[-1])

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            n_steps = 0
            if np.sum(self.action_accumulator.timestamps <= end_time) > 0:
                n_steps = np.nonzero(self.action_accumulator.timestamps <= end_time)[0][-1]+1

            if n_steps > 0:
                timestamps = action_timestamps[:n_steps]
                episode = {
                    'timestamp': timestamps,
                    'action': actions[:n_steps],
                }
                for robot_idx in range(len(self.robots)):
                    robot_pose_interpolator = PoseInterpolator(
                        t=np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_eef_pose']),
                        x=np.array(self.obs_accumulator.data[f'robot{robot_idx}_eef_pose'])
                    )
                    robot_pose = robot_pose_interpolator(timestamps)
                    episode[f'robot{robot_idx}_eef_pos'] = robot_pose[:,:3]
                    episode[f'robot{robot_idx}_eef_rot_axis_angle'] = robot_pose[:,3:]
                    joint_pos_interpolator = get_interp1d(
                        np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_joint_pos']),
                        np.array(self.obs_accumulator.data[f'robot{robot_idx}_joint_pos'])
                    )
                    joint_vel_interpolator = get_interp1d(
                        np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_joint_vel']),
                        np.array(self.obs_accumulator.data[f'robot{robot_idx}_joint_vel'])
                    )
                    episode[f'robot{robot_idx}_joint_pos'] = joint_pos_interpolator(timestamps)
                    episode[f'robot{robot_idx}_joint_vel'] = joint_vel_interpolator(timestamps)

                for gripper_idx in range(len(self.grippers)):
                    gripper_interpolator = get_interp1d(
                        t=np.array(self.obs_accumulator.timestamps[f'gripper{gripper_idx}_gripper_pose']),
                        x=np.array(self.obs_accumulator.data[f'gripper{gripper_idx}_gripper_pose'])
                    )
                    episode[f'gripper{gripper_idx}_gripper_pose'] = gripper_interpolator(timestamps)
                    gripper_force_interpolator = get_interp1d(
                        t=np.array(self.obs_accumulator.timestamps[f'gripper{gripper_idx}_gripper_force']),
                        x=np.array(self.obs_accumulator.data[f'gripper{gripper_idx}_gripper_force'])
                    )
                    episode[f'gripper{gripper_idx}_gripper_force'] = gripper_force_interpolator(timestamps)
                    print("gripper check")
                    gripper_mean = episode[f'gripper{gripper_idx}_gripper_pose'].mean(axis=0)
                    gripper_start = episode[f'gripper{gripper_idx}_gripper_pose'][0]
                    gripper_end = episode[f'gripper{gripper_idx}_gripper_pose'][-1]
                    print(f"gripper_mean: {gripper_mean}")
                    print(f"gripper_start: {gripper_start}")
                    print(f"gripper_end: {gripper_end}")
                    if np.allclose(gripper_mean, gripper_start) or np.allclose(gripper_mean, gripper_end):
                        raise ValueError(f"Gripper {gripper_idx} has no movement in this episode. Please check the gripper hardware (may re-start it).")
                    
                    print("=====================================")

                for active_vision_idx in range(len(self.active_visions)):
                    active_vision_pose_interpolator = get_interp1d(
                        t=np.array(self.obs_accumulator.timestamps[f'active_vision{active_vision_idx}_rot_axis_angle']),
                        x=np.array(self.obs_accumulator.data[f'active_vision{active_vision_idx}_rot_axis_angle'])
                    )
                    episode[f'active_vision{active_vision_idx}_rot_axis_angle'] = active_vision_pose_interpolator(timestamps)

                with open(os.path.join(self.save_dir, 'episode.pkl'), 'wb') as f:
                    pickle.dump(episode, f)
            
            self.obs_accumulator = None
            self.action_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.delete_save_dir()
        self.save_dir = None
        print(f'Episode {self.episode_id+1} dropped!')