import numpy as np
import open3d as o3d
import imageio 
import os
import json
import re
from tqdm import tqdm
from human_data.constants import *
from human_data.models import XRHand, Transform
from scipy.spatial.transform import Rotation as R


def mat_homo_to_pose(mat):
    pos = mat[:3, -1]
    orn = R.from_matrix(mat[:3, :3]).as_euler('xyz', degrees=False)
    return np.concatenate([pos, orn])


def depth2rgb(depth):
    if depth.dtype != np.uint32:
        depth = depth.astype(np.uint32)
    assert depth.max() < 256*256*256
    assert depth.min() >= 0
    rgb = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = (depth / (256*256))
    rgb[:, :, 1] = (depth % (256*256) / 256).astype(np.uint8)
    rgb[:, :, 2] = (depth % (256*256) % 256).astype(np.uint8)
    return rgb  


def rgb2depth(rgb):
    assert rgb.dtype == np.uint8
    assert rgb.max() < 256
    assert rgb.min() >= 0
    depth = rgb[:, :, 0] * 256 * 256 + rgb[:, :, 1] * 256 + rgb[:, :, 2]
    return depth


def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


def read_recorder_data(file_path, latency_frame, st_frame=0, ed_frame=-1, verbose=False):

    color_video_fp = os.path.join(file_path, "colors.mp4")
    depth_dir = os.path.join(file_path, "depths")
    logs_fp = os.path.join(file_path, "log.txt")
    poses_json_fp = os.path.join(file_path, "poses.json")
    color_video_reader = imageio.get_reader(color_video_fp, format='ffmpeg')
    depth_file_list = os.listdir(depth_dir)
    depth_file_list = sorted(depth_file_list, key=lambda filename: int(re.search(r"depth_(\d+).npy", filename).group(1)))
    videos = []
    depths = []
    if verbose: iter_bar = tqdm(enumerate(zip(color_video_reader, depth_file_list)), desc='Video')
    else: iter_bar = enumerate(zip(color_video_reader, depth_file_list))
    for idx, (frame_color, frame_depth_fp) in iter_bar:
        if idx < st_frame: 
            continue
        if ed_frame != -1 and idx >= ed_frame:
            break
        videos.append(frame_color)
        frame_depth = np.load(os.path.join(depth_dir, frame_depth_fp))
        depths.append(frame_depth)
    if verbose is True:
        print("Total frames: ", len(depths))
    color_video_reader.close()
    videos = np.array(videos)
    depths = np.array(depths)
    with open(logs_fp, "r") as f:
        logs = f.read()
    with open(poses_json_fp, "r") as f:
        poses = json.load(f)
    timestamps, left_hands, right_hands, heads = [], [], [], []
    if verbose: iter_bar = tqdm(enumerate(poses['data']), desc="Pose")
    else: iter_bar = enumerate(poses['data'])
    for idx, data in iter_bar:
        if idx < st_frame: 
            continue
        if ed_frame != -1 and idx >= ed_frame:
            break
        timestamps.append(data['ts'])
        left_hand = XRHand(data['left_hand'])
        right_hand = XRHand(data['right_hand'])
        left_hand.mul_mat_right(yfxrzu2standard)
        right_hand.mul_mat_right(yfxrzu2standard)
        head = Transform(data['head'])
        head.mul_mat_right(yfxrzu2standard)
        left_hands.append(left_hand)
        right_hands.append(right_hand)
        heads.append(head)
    videos = videos[latency_frame:]
    depths = depths[latency_frame:]
    if latency_frame > 0:
        left_hands = left_hands[:-latency_frame]
        right_hands = right_hands[:-latency_frame]
        heads = heads[:-latency_frame]
    data = {
        "videos": videos,
        "depths": depths,
        "logs": logs,
        "timestamps": timestamps,
        "left_hands": left_hands,
        "right_hands": right_hands,
        "heads": heads
    }
    return data


def process_recorder_data(data, intrinsic, quest2camera, downsample_factor=1):
    
    rgbs = data["videos"]    # (T, H, W, C)
    videos = rgbs.copy()
    depths = data["depths"]  # (T, H, W)
    depths = depths / 1000.0
    rgbs, xyzs = unprojection(rgbs, depths, intrinsic, downsample_factor=downsample_factor)  # (T, H//10, W//10, 3)

    T, H, W, C = rgbs.shape
    camera_poses_vis, hands_left, hands_right = [], [], []
    for t in range(T):

        # [Inprotant] Get Camera pose based on VR-world-frame
        quest_pose = data["heads"][t].return_matrix()
        camera_pose =  quest_pose @ fast_mat_inv(quest2camera)
        camera_poses_vis.append(camera_pose)
       
        # [Important] Get Hand pose based on Camera
        hand_left = data["left_hands"][t]
        hand_right = data["right_hands"][t] 
        hand_left.mul_mat_left(quest2camera @ fast_mat_inv(quest_pose))
        hand_right.mul_mat_left(quest2camera @ fast_mat_inv(quest_pose))
        hands_left.append(hand_left)
        hands_right.append(hand_right)

    pcds = np.concatenate([xyzs, rgbs], axis=-1)
    camera_poses = np.array(camera_poses_vis)
        
    data_dict = {
        "videos": videos,                      # (T, H, W, C), 0 ~ 255
        "depths": depths,                      # (T, H, W), unit: meter
        "pointclouds": pcds,                   # (T, H, W, 6), [xyz, rgb], unit: meter
        "logs": data['logs'],                  # str
        "timestamps": data['timestamps'],      # List[str]
        "camera_poses": camera_poses,          # (T, 4, 4)
        "left_hands": hands_left,              # List[XRHand]
        "right_hands": hands_right             # List[XRHand] 
    }
    return data_dict


def unprojection(rgbs, depths, intrinsic, downsample_factor=1, remove_hole=True):
    t, h, w = depths.shape
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    x = np.arange(w) - cx
    y = np.arange(h) - cy
    xx, yy = np.meshgrid(x, y)
    xx = xx[::downsample_factor, ::downsample_factor]
    yy = yy[::downsample_factor, ::downsample_factor]
    depths = depths[:, ::downsample_factor, ::downsample_factor]
    rgbs = rgbs[:, ::downsample_factor, ::downsample_factor]
    xx = xx * depths / fx
    yy = yy * depths / fy
    zz = depths
    xyzs = np.stack([xx, yy, zz], axis=-1)
    return rgbs, xyzs


def create_sphere(center, radius=0.025, color=[1, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color) 
    return sphere


def create_coordinate(origin, orientation, size=0.1):
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    # coordinate.rotate(orientation, center=-size/2*np.ones(3))
    coordinate.rotate(orientation, center=np.zeros(3))
    coordinate.translate(origin)
    return coordinate
