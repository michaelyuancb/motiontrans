import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from dex_retargeting.retargeting_config import RetargetingConfig

from pathlib import Path
import time
import yaml
from human_data.models import *
from real.teleop.constants_vuer import hand2inspire


def mat_homo_to_pose(mat):
    pos = mat[:3, -1]
    orn = R.from_matrix(mat[:3, :3]).as_euler('xyz', degrees=False)
    return np.concatenate([pos, orn])


def norm_vec(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm


class Hand_Retargeting:

    """
        Output:
         - left/right_wrist_results:  (T, 6),    wrist 6dof poses in VR-coordinate
         - left/right_qpos_results:   (T, 6),    inspire_hand 6dof servo-pos  (pinky, ring, middle, index, thumb-curve, thumb-inside)
         - left/right_opos_results:   (T, 5, 6), original hand 6dof poses in VR-coordinate  (thumb, index, middle, ring, pinky)
    """

    def __init__(self, config_file_path):

        RetargetingConfig.set_default_urdf_dir('./assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()


    def retarget(self, left_hand_arrs, right_hand_arrs, hand_size_coef=1.0, rotation_fill=False):

        # For rot_fill=True, it means the original data did not provide rotation information.
        # Therefore, we first calculate the fix-wrist pose, then use this direction to fill the rotation direction of all points. 

        left_hand, right_hand = XRHand(), XRHand()
        n_ts = len(left_hand_arrs)
        left_wrist_results = []
        right_wrist_results = []
        left_wrist_fix_results = []
        right_wrist_fix_results = []
        right_qpos_results = []
        left_qpos_results = []
        right_qpos_results = []
        left_urdf_qpos_results = []
        right_urdf_qpos_results = []
        left_opos_results = []
        right_opos_results = []
        for t in range(n_ts):
            left_hand.set_hand_6d_pose_array(left_hand_arrs[t])
            right_hand.set_hand_6d_pose_array(right_hand_arrs[t])

            left_wrist_results.append(left_hand.wrist.return_6d_pose())
            right_wrist_results.append(right_hand.wrist.return_6d_pose())

            # Stablize wrist pose for better performance and cross-embodiment
            # Hand Structure: https://docs.unity3d.com/Packages/com.unity.xr.hands@1.1/manual/hand-data/xr-hand-data-model.html
            left_wfix_x = norm_vec(norm_vec(left_hand.thumb1.pos - left_hand.pinky0.pos) + norm_vec(left_hand.index0.pos - left_hand.ring0.pos))
            right_wfix_x = norm_vec(norm_vec(right_hand.pinky0.pos - right_hand.thumb1.pos) + norm_vec(right_hand.ring0.pos - right_hand.index0.pos))
            left_wfix_z = norm_vec(left_hand.middle1.pos - left_hand.wrist.pos)
            right_wfix_z = norm_vec(right_hand.middle1.pos - right_hand.wrist.pos)
            left_wfix_y = norm_vec(np.cross(left_wfix_z, left_wfix_x))
            right_wfix_y = norm_vec(np.cross(right_wfix_z, right_wfix_x))
            left_rotation_matrix = np.column_stack([left_wfix_x, left_wfix_y, left_wfix_z])
            right_rotation_matrix = np.column_stack([right_wfix_x, right_wfix_y, right_wfix_z])
            left_euler_angles = R.from_matrix(left_rotation_matrix).as_euler('xyz')
            right_euler_angles = R.from_matrix(right_rotation_matrix).as_euler('xyz')
            left_fix_wrist = np.concatenate([left_hand.wrist.pos, left_euler_angles])
            right_fix_wrist = np.concatenate([right_hand.wrist.pos, right_euler_angles])
            left_wrist_fix_results.append(left_fix_wrist)
            right_wrist_fix_results.append(right_fix_wrist)

            if rotation_fill is True:
                left_hand_arrs_fill = left_hand_arrs[t].copy()
                right_hand_arrs_fill = right_hand_arrs[t].copy()
                n_finger = len(left_hand_arrs_fill) // 6
                for i in range(n_finger):
                    left_hand_arrs_fill[i*6+3:i*6+6] = left_euler_angles
                    right_hand_arrs_fill[i*6+3:i*6+6] = right_euler_angles
                
                left_hand.set_hand_6d_pose_array(left_hand_arrs_fill)
                right_hand.set_hand_6d_pose_array(right_hand_arrs_fill)

            left_org_pose = np.stack([
                left_hand.thumb_tip.return_6d_pose(),
                left_hand.index_tip.return_6d_pose(),
                left_hand.middle_tip.return_6d_pose(),
                left_hand.ring_tip.return_6d_pose(),
                left_hand.pinky_tip.return_6d_pose(),
            ])
            right_org_pose = np.stack([
                right_hand.thumb_tip.return_6d_pose(),
                right_hand.index_tip.return_6d_pose(),
                right_hand.middle_tip.return_6d_pose(),
                right_hand.ring_tip.return_6d_pose(),
                right_hand.pinky_tip.return_6d_pose(),
            ])

            left_opos_results.append(left_org_pose)
            right_opos_results.append(right_org_pose)

            # thumb-wrist, index-wrist, middle-wrist, ring-wrist, pinky-wrist, index-thumb, middle-thumb
            left_target = np.stack([
                np.linalg.inv(left_hand.wrist.return_matrix()) @ left_hand.thumb_tip.return_matrix(),
                np.linalg.inv(left_hand.wrist.return_matrix()) @ left_hand.index_tip.return_matrix(),
                np.linalg.inv(left_hand.wrist.return_matrix()) @ left_hand.middle_tip.return_matrix(),
                np.linalg.inv(left_hand.wrist.return_matrix()) @ left_hand.ring_tip.return_matrix(),
                np.linalg.inv(left_hand.wrist.return_matrix()) @ left_hand.pinky_tip.return_matrix(),
                # np.linalg.inv(left_hand.thumb_tip.return_matrix()) @ left_hand.index_tip.return_matrix(),
                # np.linalg.inv(left_hand.thumb_tip.return_matrix()) @ left_hand.middle_tip.return_matrix(),
                # np.linalg.inv(left_hand.ring1.return_matrix()) @ left_hand.index_tip.return_matrix(),

            ])
            right_target = np.stack([
                np.linalg.inv(right_hand.wrist.return_matrix()) @ right_hand.thumb_tip.return_matrix(),
                np.linalg.inv(right_hand.wrist.return_matrix()) @ right_hand.index_tip.return_matrix(),
                np.linalg.inv(right_hand.wrist.return_matrix()) @ right_hand.middle_tip.return_matrix(),
                np.linalg.inv(right_hand.wrist.return_matrix()) @ right_hand.ring_tip.return_matrix(),
                np.linalg.inv(right_hand.wrist.return_matrix()) @ right_hand.pinky_tip.return_matrix(),
                # np.linalg.inv(right_hand.thumb_tip.return_matrix()) @ right_hand.index_tip.return_matrix(),
                # np.linalg.inv(right_hand.thumb_tip.return_matrix()) @ right_hand.middle_tip.return_matrix(),
                # np.linalg.inv(right_hand.ring1.return_matrix()) @ right_hand.index_tip.return_matrix(),
            ])


            # right_target_org = right_target.copy()[:, :3, -1]
            # right_hand_arrs_my =[]
            # for i in range(len(right_hand.hand_pose)):
            #     right_hand_arrs_my.append(
            #         np.linalg.inv(right_hand.wrist.return_matrix()) @ right_hand.hand_pose[i].return_matrix()
            #     )
            # right_hand_arrs_my = np.stack(right_hand_arrs_my)[:, :3, -1]

            # import pdb; pdb.set_trace()
            # self.right_retargeting.retarget(right_target * hand_size_coef)[[4, 6, 2, 0, 9, 8]]


            # import open3d as o3d
            # from human_data.utils import create_coordinate, create_sphere
            # vis = o3d.visualization.Visualizer()
            # vis.create_window(window_name="Data Visualizer, Press Q to Exist", width=800, height=600)
            # print(right_target)
            # print("right_norm", np.linalg.norm(right_target, axis=-1))
            # coord = create_coordinate(np.zeros(3), np.eye(3), size=0.1)
            # vis.add_geometry(coord)
            # sphere = create_sphere(np.zeros(3), radius=0.01, color=[0, 0, 1])
            # vis.add_geometry(sphere)
            # sphere = create_sphere(right_target[0], radius=0.01, color=[1, 0, 0])
            # vis.add_geometry(sphere)
            # sphere = create_sphere(right_target[1], radius=0.01, color=[0, 1, 0])
            # vis.add_geometry(sphere)
            # sphere = create_sphere(right_target[2], radius=0.01, color=[0, 0.9, 0])
            # vis.add_geometry(sphere)
            # sphere = create_sphere(right_target[3], radius=0.01, color=[0, 0.8, 0])
            # vis.add_geometry(sphere)
            # sphere = create_sphere(right_target[4], radius=0.01, color=[0, 0.7, 0])
            # vis.add_geometry(sphere)
            # vis.run()

            #
            # # v0.4.6-tim

            left_target = (hand2inspire @ left_target)[:, :3, -1]
            right_target = (hand2inspire @ right_target)[:, :3, -1]
            left_qpos = self.left_retargeting.retarget(left_target * hand_size_coef)[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
            right_qpos = self.right_retargeting.retarget(right_target * hand_size_coef)[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
            left_urdf_qpos_results.append(left_qpos.copy())
            right_urdf_qpos_results.append(right_qpos.copy())

            left_qpos = [left_qpos[4], left_qpos[6], left_qpos[2], left_qpos[0], left_qpos[9], left_qpos[8]]
            right_qpos = [right_qpos[4], right_qpos[6], right_qpos[2], right_qpos[0], right_qpos[9], right_qpos[8]]
            left_qpos_results.append(left_qpos)
            right_qpos_results.append(right_qpos)

        #     if t == n_ts // 2:
        #         break

        # import pdb; pdb.set_trace()
        left_wrist_results = np.array(left_wrist_results)
        right_wrist_results = np.array(right_wrist_results)
        left_wrist_fix_results = np.array(left_wrist_fix_results)
        right_wrist_fix_results = np.array(right_wrist_fix_results)
        left_qpos_results = np.array(left_qpos_results)
        right_qpos_results = np.array(right_qpos_results)
        left_opos_results = np.array(left_opos_results)
        right_opos_results = np.array(right_opos_results)
        left_urdf_qpos_results = np.array(left_urdf_qpos_results)
        right_urdf_qpos_results = np.array(right_urdf_qpos_results)

        # exit(0)

        """
            Output:
            - left/right_wrist_results:  (T, 6),    wrist 6dof poses in VR-coordinate
            - left/right_qpos_results:   (T, 6),    inspire_hand 6dof servo-pos  (pinky, ring, middle, index, thumb-curve, thumb-inside)
            - left/right_opos_results:   (T, 5, 6), original hand 6dof poses in VR-coordinate  (thumb, index, middle, ring, pinky)
        """

        return left_wrist_results, right_wrist_results, left_wrist_fix_results, right_wrist_fix_results, left_qpos_results, right_qpos_results, left_urdf_qpos_results, right_urdf_qpos_results, left_opos_results, right_opos_results