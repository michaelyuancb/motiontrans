# isaac gym库存在问题，一定要先import pinocchio再import isaacgym
import pinocchio
from real.teleop.teleop_utils import *
import click
import os
import sys
import torch
import open3d as o3d
from human_data.utils import create_sphere, create_coordinate
from human_data.constants import xfylzu2standard, standard2xfylzu
import pdb


# Franka 底座位置
base_height = 0.9


@click.command()
@click.option('--robot_ik_urdf', '-ik_u', default="assets/franka_pinocchio/robots/franka_panda.urdf", required=True, help='Path to pinocchio urdf')
def main(robot_ik_urdf,
         ):

    # urdf file used in pinocchi
    franka_pin = pinocchio.RobotWrapper.BuildFromURDF(
        filename=robot_ik_urdf,
        package_dirs=["/opt/ros/noetic/share/"],
        root_joint=None,
    )
    # red point: hand_base_link
    # eef: panda_link8
    print(f"URDF description successfully loaded in {franka_pin}")

    hand_base_link_id = franka_pin.model.getFrameId("hand_base_link")
    panda_link8_id = franka_pin.model.getFrameId("panda_link8")
    q = franka_pin.q0
    pinocchio.forwardKinematics(franka_pin.model, franka_pin.data, q)
    pinocchio.updateFramePlacements(franka_pin.model, franka_pin.data)
    hand_base_link_pose = franka_pin.data.oMf[hand_base_link_id]
    panda_link8_pose = franka_pin.data.oMf[panda_link8_id]
    hand_base_link_position = hand_base_link_pose.translation
    hand_base_link_rotation = hand_base_link_pose.rotation
    panda_link8_position = panda_link8_pose.translation
    panda_link8_rotation = panda_link8_pose.rotation

    hand_base_mat = np.eye(4)
    panda_link8_mat = np.eye(4)
    hand_base_mat[:3, :3] = hand_base_link_rotation
    hand_base_mat[:3, -1] = hand_base_link_position
    panda_link8_mat[:3, :3] = panda_link8_rotation
    panda_link8_mat[:3, -1] = panda_link8_position

    print("hand_base_link position:", hand_base_link_position)
    print("hand_base_link rotation:\n", hand_base_link_rotation)
    print("panda_link8 position:", panda_link8_position)
    print("panda_link8 rotation:\n", panda_link8_rotation)

    # For replay:
    # # panda_link8_mat = hand_motion_pose_abs_mat @ standard2xfylzu
    # print(standard2xfylzu @ xfylzu2standard)

    hand_base_mat = hand_base_mat @ xfylzu2standard
    hand_base_link_rotation = hand_base_mat[:3, :3]
    hand_base_link_position = hand_base_mat[:3, -1]

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Data Visualizer", width=800, height=600)

    coord = create_coordinate(np.zeros(3), np.eye(3), size=0.15)
    vis.add_geometry(coord)
    coord = create_coordinate(panda_link8_position, panda_link8_rotation, size=0.1)
    sphere = create_sphere(panda_link8_position, radius=0.01, color=[0,1,0])
    vis.add_geometry(coord)
    vis.add_geometry(sphere)
    coord = create_coordinate(hand_base_link_position, hand_base_link_rotation, size=0.1)
    sphere = create_sphere(hand_base_link_position, radius=0.01, color=[1,0,0])
    vis.add_geometry(coord)
    vis.add_geometry(sphere)
    vis.run()
    vis.destroy_window()

    # hand->base = hand->panda->base = panda->base @ hand->panda

    robot_eef_to_wrist_transformation = np.linalg.inv(panda_link8_mat) @ hand_base_mat
    print(torch.Tensor(robot_eef_to_wrist_transformation))
    np.save(os.path.join('assets', 'franka_eef_to_wrist_robot_base.npy'), robot_eef_to_wrist_transformation)

    verify = panda_link8_mat @ robot_eef_to_wrist_transformation
    assert (verify - hand_base_mat < 1e-4).all()



if __name__ == '__main__':
    main()

# python -m scripts_data.get_eef_wrist_transformation
# yourdfpy assets/franka_pinocchio/robots/franka_panda.urdf