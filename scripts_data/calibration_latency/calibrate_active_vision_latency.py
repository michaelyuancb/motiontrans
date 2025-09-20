# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import cv2
import time
import numpy as np
from collections import deque
from tqdm import tqdm
from multiprocessing.managers import SharedMemoryManager
from common.precise_sleep import precise_sleep
from common.latency_util import get_latency
from real.robot_active_vision import RobotActiveVision
from matplotlib import pyplot as plt


# %%
@click.command()
@click.option('-f', '--frequency', type=float, default=10)
@click.option('-j', '--joint_name', type=str, default="yaw")
@click.option('-s', '--speed_factor', type=float, default=1.5)
def main(frequency, joint_name, speed_factor):

    if joint_name == "yaw":
        test_joint = 4
        upper_range = np.pi / 2
        lower_range = - np.pi / 2
    elif joint_name == "pitch":
        test_joint = 5
        upper_range = np.pi / 2
        lower_range = - np.pi / 8
    else:
        raise ValueError("Testing joint name is incorrect!")


    duration = 10.0
    sample_dt = 1 / 100.
    k = int(duration / sample_dt)
    sample_t = np.linspace(0, duration, k)
    value = np.sin(sample_t * duration / speed_factor)
    # pose_value = value * 1000

    pose_value = value * (upper_range - lower_range) / 2 + (upper_range + lower_range) / 2


    with SharedMemoryManager() as shm_manager:
        with RobotActiveVision(
            shm_manager=shm_manager,
            port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA3H8CB-if00-port0",
            frequency=frequency,
            verbose=False
            ) as active_vision:

            pose_list = np.full((k, 6), 0, dtype=np.float64)
            for i in range(k):
                pose_list[i][test_joint] = pose_value[i]
            active_vision.schedule_waypoint(pose_list[0], time.time() + 0.3)
            precise_sleep(1.0)


            # import pdb; pdb.set_trace()
            time1 = time.monotonic()
            timestamps = time.time() + sample_t + 100.0
            print(f"current time: {time.time()}")
            print(f"first timestamps: {timestamps[0]}")
            print(f"last timestamps: {timestamps[-1]}")

            for i in range(k):
                active_vision.schedule_waypoint(pose_list[i], timestamps[i])
                time.sleep(0.0)
            time2 = time.monotonic()
            print(f"after schedule current time{time.time()}")
            print(f"schedule time {time2-time1}")

            precise_sleep(duration + 100.0)
            print(f"start read at: {time.time()}")
            states = active_vision.get_all_state()

    state_pose_list = list()
    for pose in states["ActualTCPPose"]:
        state_pose_list.append(pose[test_joint])

    state_pose_list = np.array(state_pose_list)

    latency, info = get_latency(
        x_target=pose_value,
        t_target=timestamps,
        x_actual=state_pose_list,
        t_actual=states['robot_receive_timestamp']
    )
    print(f"End-to-end latency: {latency}sec")

    # plot everything
    fig, axes = plt.subplots(1, 3)
    # import pdb; pdb.set_trace()

    fig.set_size_inches(20, 6, forward=True)

    ax = axes[0]
    ax.plot(info['lags'], info['correlation'])
    ax.set_xlabel('lag')
    ax.set_ylabel('cross-correlation')
    # ax.set_title("Cross Correlation" + '(' + joint_name + ')')
    ax.set_title(f"Cross Correlation (joint: {joint_name})")

    ax = axes[1]
    ax.plot(timestamps, pose_value, label='target')
    ax.plot(states['robot_receive_timestamp'], state_pose_list, label='actual')
    ax.set_xlabel('time')
    ax.set_ylabel('active vision servo value')
    ax.legend()
    ax.set_title(f"Raw observation (joint: {joint_name})")

    ax=axes[2]
    t_samples = info['t_samples'] - info['t_samples'][0]
    ax.plot(t_samples, info['x_target'], label='target')
    ax.plot(t_samples-latency, info['x_actual'], label='actual-latency')
    ax.set_xlabel('time')
    ax.set_ylabel('active vision hand servo value')
    ax.legend()
    ax.set_title(f"Aligned with latency={latency} (joint: {joint_name})")

    # fig_file_name = "../data_process/latency_analyse_fig/hand_" + joint_name + "_latency.png"
    fig_file_name = f"../data_process/latency_analyse_fig/active_vision_{joint_name}_latency.png"

    plt.savefig(fig_file_name, dpi=300, bbox_inches='tight')

    plt.show()

# %%
if __name__ == '__main__':
    main()
