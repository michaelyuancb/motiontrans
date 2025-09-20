import open3d as o3d
import numpy as np
import keyboard
import math
import os
from utils import fast_mat_inv
from utils import create_sphere, read_recorder_data
from scipy.spatial.transform import Rotation

def create_coordinate(origin, orientation, size=0.1):
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    # coordinate.rotate(orientation, center=coordinate.get_center())
    coordinate.rotate(orientation, center=np.zeros(3))
    # TODO: ???????????   https://github.com/isl-org/Open3D/issues/1645
    coordinate.translate(origin)
    return coordinate



vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Calibration Visualizer, Press Q to Exist", width=800, height=600)
vis.add_geometry(create_sphere(center=np.zeros(3), radius=0.012, color=[0.5, 0.5, 0.5]))


data_path = "data/2025-02-06-22-43-42/2025-02-06-22-44-46"
data_dict = read_recorder_data(data_path)

hand_left = data_dict["left_hands"][0].wrist.return_matrix()
hand_right = data_dict["right_hands"][0].wrist.return_matrix()
head = data_dict["heads"][0].return_matrix()

np.save("hand_left.npy", hand_left)
np.save("hand_right.npy", hand_right)
np.save("head.npy", head)

quest2camera = np.load("camera_wrapper/camera_params/quest_orbbec/calib_result_quest2camera.npy")

hand_left = np.load("hand_left.npy")
hand_right = np.load("hand_right.npy")
head = np.load("head.npy")
camera = head @ fast_mat_inv(quest2camera)   # camera2quest x quest2base


print("LHand_hd", (fast_mat_inv(head) @ hand_left)[:3,-1])
print("RHand_hd", (fast_mat_inv(head) @ hand_right)[:3,-1])


coord_left = create_coordinate(origin=hand_left[:3,3], orientation=hand_left[:3,:3], size=0.05)
coord_right = create_coordinate(origin=hand_right[:3,3], orientation=hand_right[:3,:3], size=0.05)
coord_head = create_coordinate(origin=head[:3,3], orientation=head[:3,:3], size=0.1)
coord_camera = create_coordinate(origin=camera[:3,3], orientation=camera[:3,:3], size=0.1)
vis.add_geometry(create_sphere(center=head[:3,3], radius=0.01, color=[0.0, 0.0, 1.0]))
vis.add_geometry(create_sphere(center=camera[:3,3], radius=0.015, color=[1.0, 0.0, 0]))
print("left: ", hand_left[:3,3])
print("right: ", hand_right[:3,3])
print("head:", head[:3,3])
vis.add_geometry(coord_left)
vis.add_geometry(coord_right)
vis.add_geometry(coord_head)
vis.add_geometry(coord_camera)




while True:
    if keyboard.is_pressed('q'):
        break
    vis.poll_events()
    vis.update_renderer()
vis.destroy_window()