# %%
import sys
import os

# ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
# print(ROOT_DIR)
# sys.path.append(ROOT_DIR)
# os.chdir(ROOT_DIR)

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
# from real_world.spacemouse_shared_memory import Spacemouse
# from real_world.rtde_interpolation_controller import RTDEInterpolationController
from real.robot_franka import RobotFranka, RobotFrankaInterface
from common.precise_sleep import precise_wait, precise_sleep
from common.latency_util import get_latency
from matplotlib import pyplot as plt

# %%
@click.command()
@click.option('-rh', '--robot_hostname', default='172.16.0.1')
@click.option('-f', '--frequency', type=float, default=30)
@click.option('-s', '--speed_factor', type=float, default=15)

def main(robot_hostname, frequency, speed_factor):
    max_pos_speed = 0.5
    max_rot_speed = 1.2
    cube_diag = np.linalg.norm([1,1,1])
    tcp_offset = 0.21
    # tcp_offset = 0
    dt = 1/frequency
    command_latency = dt / 2

    orientation = [-1.50331319, 0.05143192, -0.03968948]


    duration = 30.0
    sample_dt = 1 / frequency
    k = int(duration / sample_dt)
    sample_t = np.linspace(0, duration, k)
    pose_x = np.sin(sample_t * duration / (speed_factor * 3)) * 0.1 + 0.55
    pose_y = np.cos(sample_t * duration / (speed_factor * 2)) * 0.12 + 0.2
    pose_z = np.sin(sample_t * duration / (speed_factor * 1)) * 0.1 + 0.4
    rotation_r = np.sin(sample_t * duration / (speed_factor * 1)) * 0.4 + orientation[0]
    rotation_p = np.cos(sample_t * duration / (speed_factor * 2)) * 0.4 + orientation[1]
    rotation_y = np.sin(sample_t * duration / (speed_factor * 3)) * 0.4 + orientation[2]
    print(f"ip: {robot_hostname}")
    


    with SharedMemoryManager() as shm_manager:
        with RobotFranka(
            shm_manager=shm_manager,
            robot_ip=robot_hostname,
            frequency=200,
            Kx_scale=np.array([0.8,0.8,1.2,3.0,3.0,3.0]),
            Kxd_scale=np.array([2.0,2.0,2.0,2.0,2.0,2.0]),
            
            verbose=False
        ) as controller:
        # Spacemouse(
        #     shm_manager=shm_manager
        # ) as sm:
        
            print('Ready!')
            # to account for recever interfance latency, use target pose
            # to init buffer.
            state = controller.get_state()
            init_pose = state['ActualTCPPose']
            print(f"initial pose: {init_pose}")
            t_start = time.time()
            
            t_target = list()
            x_target = list()

            command_latency = 0.0   

            target_pose = np.full((len(pose_x), 6), 0, dtype=np.float64)
            for i in range(len(pose_x)):
                # import pdb; pdb.set_trace()
                target_pose[i] = np.array([pose_x[i], pose_y[i], pose_z[i], rotation_r[i], rotation_p[i], rotation_y[i]])

            # import pdb; pdb.set_trace()

            

            
            controller.schedule_waypoint(target_pose[0], time.time() + 3)
            precise_sleep(3.0)


            timestamps = time.time() + sample_t + 5.0
            t_target = timestamps
            for i in range(k):
                controller.schedule_waypoint(target_pose[i], timestamps[i])
                time.sleep(0.0)
            precise_sleep(duration + 10.0)


            iter_idx = 0
           

            states = controller.get_all_state()

    # t_target = np.array(t_target)
    # x_target = np.array(x_target)
    x_target = target_pose
    t_actual = states['robot_receive_timestamp']
    x_actual = states['ActualTCPPose']
    n_dims = 6
    fig, axes = plt.subplots(n_dims, 3)
    fig.set_size_inches(15, 15, forward=True)

    latency_list = list()
    for i in range(n_dims):
        latency, info = get_latency(x_target[...,i], t_target, x_actual[...,i], t_actual)
        latency_list.append(latency)

        row = axes[i]
        ax = row[0]
        ax.plot(info['lags'], info['correlation'])
        ax.set_xlabel('lag')
        ax.set_ylabel('cross-correlation')
        ax.set_title(f"Action Dim {i} Cross Correlation")

        ax = row[1]
        ax.plot(t_target, x_target[...,i], label='target')
        ax.plot(t_actual, x_actual[...,i], label='actual')
        ax.set_xlabel('time')
        ax.set_ylabel('gripper-width')
        ax.legend()
        ax.set_title(f"Action Dim {i} Raw observation")

        ax = row[2]
        t_samples = info['t_samples'] - info['t_samples'][0]
        ax.plot(t_samples, info['x_target'], label='target')
        ax.plot(t_samples-latency, info['x_actual'], label='actual-latency')
        ax.set_xlabel('time')
        ax.set_ylabel('gripper-width')
        ax.legend()
        ax.set_title(f"Action Dim {i} Aligned with latency={latency}")
    
    latency_average = np.array(latency_list).mean()
    print(f"Average latency: {latency_average}")
    plt.text(0,0,f"Average latency: {latency_average}")

    fig.tight_layout()
    plt.savefig("../data_process/latency_analyse_fig/franka_latency.png", dpi=300, bbox_inches='tight')
    plt.show()

# %%
if __name__ == '__main__':
    main()
