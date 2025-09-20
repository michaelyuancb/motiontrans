
import time
import numpy as np
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from pytransform3d import rotations
from scipy.spatial.transform import Rotation as R
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

from real.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from real.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

from common.precise_sleep import precise_wait

from real.active_vision_utils.dynamixel.active_cam import DynamixelAgent
from real.active_vision_utils.constants_vuer import *

import numpy as np
import matplotlib.pyplot as plt


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

grd_yup2grd_zup = np.array([[0, 0, -1, 0],
                            [-1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])

class RobotActiveVision(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            port: str,
            control_mode="pose",
            frequency=30,
            launch_timeout=3,
            initial_control_mode = None,
            joints_init=None,
            get_max_k=None,
            verbose=False,
            receive_latency=0.0
        ):
        super().__init__()
        self.port = port
        self.active_vision_agent = DynamixelAgent(port=port)
        self.active_vision_agent._robot.set_torque_mode(True)

        self.frequency = frequency
        self.control_mode = control_mode
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.receive_latency = receive_latency
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 10)

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=1024
        )

        # build ring buffer
        example = {
            "ActualTCPPose": np.zeros(6),
            'robot_receive_timestamp': time.time(),
            'robot_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

        #===================================================================================================================

        if initial_control_mode is None:
            self.control_mode = "policy"
        else:
            self.control_mode = initial_control_mode

    def start(self, wait=True, initial_control_mode = None):
        if initial_control_mode is None:
            self.control_mode = "policy"
        else:
            self.control_mode = initial_control_mode
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[Active vision] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

     # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    
    def clear_queue(self):
        self.input_queue.clear()

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    def get_pose(self):
        joint_angle = self.active_vision_agent._robot.read_joint_state()
        return [0, 0, 0, 0, joint_angle[0], joint_angle[1]]

    # the valid range of position if [-pi, pi]
    def set_pose(self, pose):
        self.joint_state = pose[-2:]
        self.active_vision_agent._robot.command_joint_state(self.joint_state)

    
    def clear_obs(self):
        self.ring_buffer.clear()


    # ========= main loop in process ============
    def run(self):

        try:
            if self.verbose:
                print(f"[Active vision] Connect to robot: {self.port}")

            # init pose
            if self.joints_init is not None:
                self.set_pose(self.joints_init)

            # main loop
            dt = 1. / self.frequency
            curr_pose = self.get_pose()

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                # send command to robot
                t_now = time.monotonic()
                tip_pose = pose_interp(t_now)

                self.set_pose(tip_pose)

                # update robot state
                state = dict()
                state['ActualTCPPose'] = self.get_pose()

                    
                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_k(1)
                    command_target_pose = commands["target_pose"]
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[Active vision] New pose target:{} duration:{}s".format(
                                target_pose[:2], duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[Active vision] Actual frequency {1/(time.monotonic() - t_now)}")
        # except KeyboardInterrupt:
        #     print(f'\n\n\n\nKeyboardInterrupt ActiveVision{self.port}\n\n\n\n\n')
        #     self.ready_event.set()
        finally:
            print(f'\n\n\n\nterminate_current_policy ActiveVision{self.port}\n\n\n\n\n')
            self.ready_event.set()

            if self.verbose:
                print(f"[Active vision] Disconnected from robot: {self.port}")
