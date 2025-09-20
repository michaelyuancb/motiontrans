import os
import time
import datetime
import socket
import numpy as np
import shutil
from human_data.models import XRHand, Transform
from scipy.spatial.transform import Rotation
from queue import Full, Empty
from multiprocessing import Process, Queue, Value


class QuestRecorder:
    def __init__(self, output_dir, pose_cmd_port=12346):
        # self.vr_ip = vr_ip
        self.pose_cmd_port = pose_cmd_port

        # Default World-Frame, users could set it by themself.
        self.world_frame = np.array([0., 0., 0., 0., 0., 0., 1.])   
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.data_dir = None
        self.quest_recording = False

        # multi-process for non-blocking poses recieve. 
        self.data_queue = Queue(maxsize=1)
        self.updating_flag = Value('i', 0)
        self.receive_process = Process(target=self.receive_process_func, args=(self.pose_cmd_port, self.data_queue, self.updating_flag))
        self.receive_process.daemon = True
        self.receive_process.start()


    def receive_process_func(self, port, data_queue, updating_flag):
        wrist_listener_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        wrist_listener_s.bind(("", port))
        wrist_listener_s.setblocking(1)
        wrist_listener_s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 0)
        while True:
            try:
                data, _ = wrist_listener_s.recvfrom(8192)
                try:
                    data_queue.get(block=False) 
                except Empty:
                    pass 
                data_queue.put(data, block=True)  
                time.sleep(0.00005)               # give the main process reading chance
            except KeyboardInterrupt:
                break 
        wrist_listener_s.close()
        print("data-recieve-process finished.")
        

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
        if self.receive_process.is_alive():
            self.receive_process.terminate()
        self.receive_process.join()

    def delete_data_dir(self):
        if self.data_dir is not None and os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)


    def receive(self, verbose=False):
        while True:
            try:
                data = self.data_queue.get(block=False) 
                break
            except KeyboardInterrupt:
                self.close()
                exit(0)
            except Exception as e:
                if verbose:
                    print("Waiting, conflict with QuestUnity-subprocess. {e}")
        
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
            # np.save(os.path.join(self.data_dir, "WorldFrame.npy"), self.world_frame)
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