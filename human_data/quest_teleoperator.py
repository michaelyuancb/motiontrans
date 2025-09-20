import os
import io
import time
import datetime
import socket
import numpy as np
import shutil
from PIL import Image
from human_data.models import XRHand, Transform
import zmq
import cv2
from scipy.spatial.transform import Rotation


class QuestTeleoperator:
    def __init__(self, local_ip, pose_cmd_port, streaming_port):
        # self.vr_ip = vr_ip
        self.local_ip = local_ip
        self.pose_cmd_port = pose_cmd_port
        # Quest should send WorldFrame as well as head & hand pose via UDP
        self.wrist_listener_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.wrist_listener_s.bind(("", pose_cmd_port))
        self.wrist_listener_s.setblocking(1)
        self.wrist_listener_s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 0)

        self.stream_port = streaming_port
        context = zmq.Context()
        self.stream_socket = context.socket(zmq.PUB)  
        self.stream_socket.bind(f"tcp://*:{streaming_port}")

    
    def close(self):
        self.wrist_listener_s.close()
        self.stream_socket.close()

    def stream_image(self, image: np.ndarray, verbose=False):   # RGB format, 1280 x 720
        # assert image.shape[0] == 720 and image.shape[1] == 1280
        if verbose is True:
            print("sending: " + str(image.shape))
        _, buffer = cv2.imencode('.jpg', image[:,:,::-1], [cv2.IMWRITE_JPEG_QUALITY, 50])
        self.stream_socket.send(buffer.tobytes())  # 发送数据

    def receive(self, verbose=False):
        data, _ = self.wrist_listener_s.recvfrom(8192)
        data_string = data.decode()
        now = datetime.datetime.now()
        if verbose is True:
            print(f"[PC] Teleop Received data: {data_string}")
        if data_string.startswith("Wait"):
            return "Wait", None, None
        if data_string.startswith("Stream-Fail"):
            return "Stream-Fail", None, None
        elif data_string.startswith("Start"):
            return "Start", None, None
        elif data_string.startswith("Ensure"):
            return "Ensure", None, None
        elif data_string.startswith("Save"):
            return "Save", None, None
        elif data_string.startswith("Cancel"):
            return "Cancel", None, None
        elif (data_string.find("LHand:") != -1) and (data_string.find("RHand:") != -1):
            data_string_ = data_string[11:].split(",")
            st = data_string.find("LHand:") + len("LHand:")
            ed = data_string.find("RHand:")
            left_hand = XRHand(data_string[st:ed])
            right_hand = XRHand(data_string[ed+len("RHand:"):])
            head_str = data_string[11:data_string.find("LHand:")]
            head_data_list = [float(data) for data in head_str.split(",")]
            head_tf = np.array(head_data_list[:7])
            head_pose = Transform()
            head_pose.set_pose(head_tf[:3], head_tf[3:])
            return "Data", (left_hand, right_hand), head_pose
        else:
            raise ValueError(f"Unknown data-type received: {data_string}")