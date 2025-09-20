# ============================ QuestRecorder (Single Process Version) ==============================
# This may lead to stuck and blocking-reading, which may cause significant lantency and timestamp mismatching

import os
import time
import datetime
import socket
import numpy as np
import shutil
from human_data.models import XRHand, Transform
from scipy.spatial.transform import Rotation


class QuestRecorder:
    def __init__(self, output_dir, pose_cmd_port=12346):
        # self.vr_ip = vr_ip
        self.pose_cmd_port = pose_cmd_port
        # Quest should send WorldFrame as well as head & hand pose via UDP
        self.wrist_listener_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.wrist_listener_s.bind(("", pose_cmd_port))
        self.wrist_listener_s.setblocking(1)
        self.wrist_listener_s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)

        # Default World-Frame, users could set it by themself.
        self.world_frame = np.array([0., 0., 0., 0., 0., 0., 1.])   
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.data_dir = None
        self.quest_recording = False

    def compute_rel_transform(self, pose):                            
        """
        Compute relative-position to the world-frame set by user, also change from Unity-Left-Coordinate to Real-Right-Coordinate.
        pose: np.ndarray shape (7,) [x, y, z, qx, qy, qz, qw] in unity frame
        """
        world_frame = self.world_frame.copy()
        world_frame[:3] = np.array([world_frame[0], world_frame[2], world_frame[1]])
        pose[:3] = np.array([pose[0], pose[2], pose[1]])

        Q = np.array([[1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0.]])
        rot_base = Rotation.from_quat(world_frame[3:]).as_matrix()
        rot = Rotation.from_quat(pose[3:]).as_matrix()
        rel_rot = Rotation.from_matrix(Q @ (rot_base.T @ rot) @ Q.T) # Is order correct.
        rel_pos = Rotation.from_matrix(Q @ rot_base.T@ Q.T).apply(pose[:3] - world_frame[:3]) # Apply base rotation not relative rotation...
        return rel_pos, rel_rot.as_quat()

    def compute_rel_transform_for_hand(self, hand):
        for i in range(len(hand.hand_pose)):
            rel_pos, rel_orn = self.compute_rel_transform(hand.hand_pose[i].return_pose())
            hand.hand_pose[i].set_pose(np.concatenate([rel_pos, rel_orn]))
    
    def close(self):
        self.wrist_listener_s.close()

    def delete_data_dir(self):
        if self.data_dir is not None and os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)

    def receive(self, verbose=False):
        try:
            data, _ = self.wrist_listener_s.recvfrom(8192)
        except KeyboardInterrupt:
            self.close()
            exit(0)
        timestamp = time.time()
        data_string = data.decode()
        now = datetime.datetime.now()
        if verbose is True:
            print(f"[PC] Received data: {data_string}")
        
        if data_string.startswith("Wait-Ensure"):
            return "Wait-Ensure", None, None, timestamp
        elif data_string.startswith("Wait"):
            return "Wait", None, None, timestamp
        elif data_string.startswith("WorldFrame"):
            st = data_string.find("WorldFrame:") + len("WorldFrame:")
            ed = data_string.find("Head:")
            base_point_data_list = [float(data) for data in data_string[st:ed].split(",")]
            base_point_tf = np.array(base_point_data_list[:7])
            rel_bp_pos, rel_bp_rot = self.compute_rel_transform(base_point_tf)
            head_str = data_string[data_string.find("Head:")+len("Head:"):]
            head_data_list = [float(data) for data in head_str.split(",")]
            head_tf = np.array(head_data_list[:7])
            rel_head_pos, rel_head_rot = self.compute_rel_transform(head_tf)
            head_pose = Transform()
            head_pose.set_pose(np.concatenate([rel_head_pos, rel_head_rot]))
            base_point_pose = Transform()
            base_point_pose.set_pose(np.concatenate([rel_bp_pos, rel_bp_rot]))
            return "WorldFrame", (base_point_pose, head_pose), None, timestamp
        elif data_string.startswith("Start"):
            formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
            self.data_dir = os.path.join(self.output_dir, formatted_time)
            os.mkdir(self.data_dir)
            np.save(os.path.join(self.data_dir, "WorldFrame.npy"), self.world_frame)
            self.quest_recording = True
            return "Start", None, None, timestamp
        elif data_string.startswith("Ensure"):
            self.quest_recording = False
            return "Ensure", None, None, timestamp
        elif data_string.startswith("Save"):
            self.quest_recording = False
            return "Save", None, None, timestamp
        elif data_string.startswith("Cancel"):
            self.quest_recording = False
            return "Cancel", None, None, timestamp
        elif (data_string.find("LHand:") != -1) and (data_string.find("RHand:") != -1):
            if self.quest_recording is False:        # some sync bug from Quest Code, only recieve "Start" can we record data
                return "Wait", None, None, timestamp
            st = data_string.find("LHand:") + len("LHand:")
            ed = data_string.find("RHand:")
            left_hand = XRHand(data_string[st:ed])
            right_hand = XRHand(data_string[ed+len("RHand:"):])
            head_str = data_string[11:data_string.find("LHand:")]
            head_data_list = [float(data) for data in head_str.split(",")]
            head_tf = np.array(head_data_list[:7])
            self.compute_rel_transform_for_hand(left_hand)
            self.compute_rel_transform_for_hand(right_hand)
            rel_head_pos, rel_head_rot = self.compute_rel_transform(head_tf)
            head_pose = Transform()
            head_pose.set_pose(np.concatenate([rel_head_pos, rel_head_rot]))
            return "Data", (left_hand, right_hand), head_pose, timestamp
        else:
            raise ValueError(f"Unknown data-type received: {data_string}")