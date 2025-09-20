
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
from common.linear_trajectory_interpolator import LinearTrajectoryInterpolator

from common.precise_sleep import precise_wait
from common.interpolation_util import get_interp1d
from real.inspire_hand_utils.inspire_hand_agent import InspireHandAgent


inspire_hand_primitives = [
    np.array([0, 0, 0, 0, 0, 0.3367]),                         # 0: not-grasping free hand with thumb outside
    np.array([0, 0, 0, 0, 0, 1.425]),                          # 1: not-grasping free hand with thumb inside
    np.array([0, 0, 0, 1.05, 0.15, 1.38]),                     # 2: index-thumb pinch
    np.array([0, 0, 1.05, 1.05, 0.15, 1.475]),                 # 3: index+middle-thumb pinch
    np.array([1.25, 1.25, 1.25, 1.25, 0.08, 1.475]),           # 4: wrap with thumb inside (rock) 
    np.array([1.6467, 1.6467, 1.6467, 1.6467, 0.4, 0.3367]),   # 5: wrap with thumb outside and curve 
    np.array([1.6467, 1.6467, 1.6467, 1.6467, 0, 0.3367]),     # 6: wrap with thumb outside and straight
    np.array([0.1, 0.1, 0.1, 0.1, 0.15, 0.3367]),                         # 7: default start
]

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    RESET = 3


class RobotInspireHand(mp.Process):
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 port: str,
                 frequency=30,
                 launch_timeout=3,
                 unit="radians",
                 joints_init=None,
                 get_max_k=None,
                 verbose=False,
                 receive_latency=0.0
                 ):
        super().__init__()
        self.port = port
        self.agent = InspireHandAgent(port=self.port,
                                      hand_id=1,
                                      baudrate=115200
                                      )
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        joint_unit = "radians"
        self.joint_unit = unit
        self.joints_init = joints_init
        self.receive_latency = receive_latency
        self.verbose = verbose

        self.agent.set_joint_speed()

        if get_max_k is None:
            get_max_k = int(frequency * 1000)

        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
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
            "gripper_pose": np.zeros(6),
            "gripper_force": np.zeros(6),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
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

        # ===================================================================================================================

    def start(self, wait=True, initial_control_mode=None):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[Inspire Hand] Controller process spawned at {self.pid}")

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
        assert (duration >= (1 / self.frequency))
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
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= receive APIs =============


    def get_pose_radians(self):
        joint_angle = self.agent.read_joint_position2()
        return joint_angle

    def get_pose_servo(self):
        servo_position = self.agent.read_joint_servo_position()
        return servo_position

    def get_force(self):
        finger_force = self.agent.read_joint_force()
        return finger_force

    def set_pose_radians(self, pose):
        self.agent.set_joint_position2(pose.tolist())

    def set_pose_servo(self, pose):
        self.agent.set_joint_servo_position(pose.tolist())

    def reset_pose(self):
        message = {
            'cmd': Command.RESET.value
        }
        self.input_queue.put(message)

    def clear_obs(self):
        self.ring_buffer.clear()


    # ========= main loop in process ============
    def run(self):

        try:
            if self.verbose:
                print(f"[Inspire Hand] Connect to robot: {self.port}")

            # init pose
            if self.joints_init is not None:
                if self.joint_unit == "radians":
                    self.set_pose_radians(self.joints_init)
                else:
                    self.set_pose_servo(self.joints_init)

            # main loop
            dt = 1. / self.frequency
            if self.joint_unit == "radians":
                curr_pose = self.get_pose_radians()
            else:
                curr_pose = self.get_pose_servo()

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = LinearTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            # pose_interp = get_interp1d(
            #     t=[curr_t],
            #     x=[curr_pose]
            # )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                # send command to robot
                t_now = time.monotonic()
                tip_pose = pose_interp(t_now)

                if self.joint_unit == "radians":
                    self.set_pose_radians(tip_pose)
                else:
                    self.set_pose_servo(tip_pose)

                # update robot state
                state = dict()
                if self.joint_unit == "radians":
                    state['gripper_pose'] = self.get_pose_radians()
                else:
                    state['gripper_pose'] = self.get_pose_servo()

                state['gripper_force'] = self.get_force()

                t_recv = time.time()
                state['gripper_receive_timestamp'] = t_recv
                state['gripper_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_k(1)
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
                    elif cmd == Command.RESET.value:
                        if self.verbose:
                            print("RESET Inspire Hand")
                        target_pose = inspire_hand_primitives[-1]
                        duration = 0.1
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print(f"[Inspire Hand port: {self.port}] New pose target:{target_pose} duration:{duration}s")
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        # t_insert = curr_time

                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[Inspire Hand] New pose target:{} ".format(
                                target_pose[:2]))
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

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[Inspire Hand] Actual frequency {1 / (time.monotonic() - t_now)}")
        except KeyboardInterrupt:
            print(f'KeyboardInterrupt Terminate_current_policy Inspire Hand {self.port} by Ctrl+C\n\n\n\n\n')
        finally:
            print(f'Terminate_current_policy Inspire Hand {self.port} by Ctrl+C\n\n\n\n\n')
            self.ready_event.set()

            if self.verbose:
                print(f"[Inspire Hand] Disconnected from robot: {self.port}")
