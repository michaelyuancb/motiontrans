############# Filter out invalid robot data due to InspireHand reading bug #############

import pickle 
import os
import numpy as np 
import shutil


base_folder = "data/data_robot_raw"
folder_list = os.listdir(base_folder)

for folder in folder_list:
    pkl_file = os.path.join(base_folder, folder, "episode.pkl")
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    gripper_pose = data['gripper0_gripper_pose']
    gripper_pose_mean = gripper_pose.mean(axis=0)
    gripper_pose_delta = gripper_pose - gripper_pose_mean
    gripper_pose_delta_norm = np.linalg.norm(gripper_pose_delta, axis=1)
    if gripper_pose_delta_norm.max() < 1e-5:
        print(folder)
        shutil.rmtree(os.path.join(base_folder, folder))

print("Rest Data Number:", len(os.listdir(base_folder)))

# python filter_invalid_data.py