import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from real.teleop.TeleVision import OpenTeleVision
from real.teleop.Preprocessor import VuerPreprocessor
from real.teleop.constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations

from pathlib import Path
import argparse
import time
import yaml
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore


# from inspire_hand_utils.inspire_hand_agent import InspireHandAgent

class VuerTeleop:
    def __init__(self, config_file_path, 
                 gesture_pinch_threshold=0.03,
                 gesture_inside_thumb_threshold=(45.0/180.0)*np.pi,
                 gesture_curve_thumb_threshold=(20.0/180.0)*np.pi,
                 gesture_curve_not_thumb_threshold=(60.0/180.0)*np.pi,
                 distance_to_eye=1.0, resolution=(720, 1280)):

        self.distance_to_eye = distance_to_eye
        self.resolution = resolution
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()

        # # Quest3
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, cert_file=None, distance_to_eye=distance_to_eye, key_file=None,  ngrok=False)
        # Vision Pro
        # cert_file = "/home/aa/bi-dex-mimic/cert/cert.pem"
        # key_file = "/home/aa/bi-dex-mimic/cert/key.pem"
        # self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, cert_file=cert_file, distance_to_eye=distance_to_eye, key_file=key_file,  ngrok=False)
        
        self.processor = VuerPreprocessor()

        RetargetingConfig.set_default_urdf_dir('./assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

        self.gesture_pinch_threshold = gesture_pinch_threshold
        self.gesture_inside_thumb_threshold = gesture_inside_thumb_threshold
        self.gesture_curve_thumb_threshold = gesture_curve_thumb_threshold
        self.gesture_curve_not_thumb_threshold = gesture_curve_not_thumb_threshold


    def is_pinch(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2) <= self.gesture_pinch_threshold
    
    def is_curve_not_thumb(self, pos):
        return np.linalg.norm(pos) >= self.gesture_curve_not_thumb_threshold
    
    def is_curve_thumb(self, pos):
        return np.linalg.norm(pos) >= self.gesture_curve_thumb_threshold
    
    def is_inside_thumb(self, pos):
        return np.linalg.norm(pos) >= self.gesture_inside_thumb_threshold

    # def get_grasp_gesture(self, fingertip, qpose):    
        
    #     # fingertip: thumb, index, middle, ring, pinky
    #     # qpose: thumb-inside, thumb-curve, index, middle, ring, pinky

    #     thumb = fingertip[0]
    #     thumb_inside = self.is_inside_thumb(qpose[0])
    #     thumb_curve = self.is_curve_thumb(qpose[1])
    #     index, c_index = fingertip[1], self.is_curve_not_thumb(qpose[2])
    #     middle, c_middle = fingertip[2], self.is_curve_not_thumb(qpose[3])
    #     ring, c_ring = fingertip[3], self.is_curve_not_thumb(qpose[4])
    #     pinky, c_pinky = fingertip[4], self.is_curve_not_thumb(qpose[5])

    #     if (self.is_pinch(thumb, index)) and (not self.is_pinch(thumb, middle)):
    #         return 2               # index-thumb pinch
    #     else:
    #         return 0 

    #     # if (self.is_pinch(thumb, index)) and (self.is_pinch(thumb, middle)) and (not self.is_pinch(thumb, ring)):
    #     #     return 3               # index+middle-thumb pinch
    #     # if (self.is_pinch(thumb, index)) and (self.is_pinch(thumb, ring)):
    #     #     return 3               # wrap with thumb inside (rock) 
    #     # if c_index and c_middle and c_ring:  # NOTE: we ignore pinky        
    #     #     if thumb_curve:
    #     #         return 5           # wrap with thumb outside and curve 
    #     #     else:
    #     #         return 6           # wrap with thumb outside and straight
        
    #     # if thumb_inside:
    #     #     return 1               # not-grasping free hand with thumb inside
    #     # return 0                   # not-grasping free hand with thumb outside


    def step(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)

        head_rmat = head_mat[:3, :3]

        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])

        # thumb-inside, thumb-curve, index, middle, ring, pinky
        
        # v0.4.6-tim
        left_target = np.concatenate([left_hand_mat[tip_indices], (left_hand_mat[9]-left_hand_mat[4])[None], (left_hand_mat[14]-left_hand_mat[4])[None]])
        left_qpos = self.left_retargeting.retarget(left_target)[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]  
        right_target = np.concatenate([right_hand_mat[tip_indices], (right_hand_mat[9]-right_hand_mat[4])[None], (right_hand_mat[14]-right_hand_mat[4])[None]])
        right_qpos = self.right_retargeting.retarget(right_target)[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

        # # v0.4.6
        # left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]  
        # right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        
        # v0.1.0
        # left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        # right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]


        left_fingertip = left_hand_mat[tip_indices] - left_hand_mat[0]      # thumb, index, middle, ring, pinky
        right_fingertip = right_hand_mat[tip_indices] - right_hand_mat[0]   

        left_qpos_use = left_qpos[[9, 8, 0, 2, 6, 4]]  
        right_qpos_use = right_qpos[[9, 8, 0, 2, 6, 4]]

        # left_gesture = self.get_grasp_gesture(left_fingertip, left_qpos_use)
        # right_gesture = self.get_grasp_gesture(right_fingertip, right_qpos_use)

        return head_rmat, left_pose, right_pose, left_qpos, right_qpos    # , left_gesture, right_gesture