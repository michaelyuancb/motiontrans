from real.robot_franka import RobotFrankaInterface
import numpy as np
import time
import torch

robot = RobotFrankaInterface(ip='172.16.0.1', port=4242)

Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * 1.0
Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * 1.0
robot.start_cartesian_impedance(
    Kx=Kx,
    Kxd=Kxd
)
print(robot.get_ee_pose())

joints_init = [0.00010400846076663584, -0.24481917917728424, -0.0936817154288292, -2.7792861461639404, -0.0670328289270401, 1.998847246170044, 0.7243571877479553]
joints_init_duration = 4

# sol= robot.ik_exist(np.asarray([0.3, -0.4, 0.504]),
#                                     np.asarray([0.707, 0., 0.707, 0.]),
#                                     np.asarray([-1.7151,  1.1985,  1.5643, -1.7135,  0.4483,  2.2208,  0.0210]),
#                                 )
# print(f"ik: {sol}")
# robot.move_to_joint_positions(positions=np.asarray(joints_init),time_to_go=joints_init_duration)

robot.go_home()
robot.move_to_joint_positions(positions=np.asarray([0.0,  0.0,  0.0, -1.5707,  0.0,  1.5707, 0.0]),time_to_go=2.0)
print(f"home ee_pose is {robot.get_ee_pose()}")
print(f"home joint_position is {robot.get_joint_positions()}")
