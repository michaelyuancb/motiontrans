from human_data.models import Transform
import argparse
import os
import cv2
import json
import numpy as np
import open3d as o3d
import keyboard
from scipy.spatial.transform import Rotation
from common.cv2_util import intrinsic_transform_resize
from human_data.constants import * 
from human_data.utils import fast_mat_inv, create_sphere, create_coordinate


def convert_list(input_list):
    result = []
    temp = ""
    for item in input_list:
        if item == ',':
            if len(temp) > 0:
                result.append(int(temp))
                temp = ""
        else:
            temp += item
    if temp:
        result.append(int(temp))
    return result


def calibration(args,
                out_resolutions_resize=(1280, 720),
                out_resolutions_crop=(640, 480), 
                verbose=False
                ):
    dist_coeffs = None
    intrinsic = np.load(os.path.join(args.save_dir, args.instrinisc_file))
    camera_bp2d = np.load(os.path.join(args.save_dir, "calib_camera_image_base_point.npy")).reshape(-1, 2)
    camera_bp3d = np.load(os.path.join(args.save_dir, "calib_camera_image_base_point_obj.npy")).reshape(-1, 3)
    if args.camera_bp3d_reverse:
        camera_bp3d = camera_bp3d[::-1]
    camera_bp3d = camera_bp3d / 1000.0

    with open(os.path.join(args.save_dir, "calib_quest_base_point.json"), "r") as f:
        quest_bp_str = json.load(f)
    with open(os.path.join(args.save_dir, "calib_quest_head.json"), "r") as f:
        quest_head_str = json.load(f)
    quest_bp_list, quest_head_list = [], []

    if verbose:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Calibration Visualizer, Press Q to Exist", width=800, height=600)
        vis.add_geometry(create_sphere(center=np.zeros(3), radius=0.012, color=[0.5, 0.5, 0.5]))
        vis.add_geometry(create_coordinate(np.zeros(3), orientation=np.eye(3), size=0.05))

    # (R1, t1)  (R2, t2) --> (R1@R2, R1@t2+t1)
    # (0,  1 )  (0,  1 ) --> (0    , 1       )
    camera_bp3d = (xfyrzd2standard[:3,:3] @ camera_bp3d[:,:,None])[:,:,0]

    assert len(args.quest_base_camera_corner_id) == len(quest_bp_str)
    spheres = []
    coordinates = []
    for i, (bp_str, head_str) in enumerate(zip(quest_bp_str, quest_head_str)):
        bp = Transform(bp_str).return_matrix()
        cam = Transform(head_str).return_matrix()
        bp = bp @ yfxrzu2standard
        cam = cam @ yfxrzu2standard
        quest_bp_list.append(bp)
        quest_head_list.append(cam)
       
        if verbose:
            vis.add_geometry(create_sphere(center=bp[:3,-1], radius=0.01, color=[(i+1)/len(args.quest_base_camera_corner_id), 0, 0]))
            vis.add_geometry(create_sphere(center=cam[:3,-1], radius=0.01, color=[0, 0, 1]))
            vis.add_geometry(create_coordinate(origin=bp[:3,-1], orientation=bp[:3,:3], size=0.05))
            vis.add_geometry(create_coordinate(origin=cam[:3,-1], orientation=cam[:3,:3], size=0.05))

    quest2base_cs_list = []
    for i in range(len(quest_bp_list)):
        q2b = fast_mat_inv(quest_bp_list[i]) @ quest_head_list[i]
        print(f"Quest2Base{i}:\n{q2b[:3,3]}")
        quest2base_cs_list.append(q2b)  # quest to (chosed) base
    
    if verbose:
        for i in range(len(quest2base_cs_list)):
            base_cs = camera_bp3d[args.quest_base_camera_corner_id[i]]
            print(base_cs)
            vis.add_geometry(create_sphere(center=base_cs, radius=0.01, color=[(i+1)/len(args.quest_base_camera_corner_id), 0, 0]))
            headset_pose = quest2base_cs_list[i]
            base = camera_bp3d[0]
            base_cs2base = np.eye(4) 
            base_cs2base[:3, :3] = np.eye(3)
            base_cs2base[:3, 3] = base_cs - base
            base_cs2base[3, 3] = 1
            quest2base = base_cs2base @ headset_pose
            vis.add_geometry(create_sphere(center=quest2base[:3,-1], radius=0.01, color=[0, 0, 1]))
            vis.add_geometry(create_coordinate(origin=quest2base[:3,-1], orientation=quest2base[:3, :3], size=0.1))

    algorithms = [
        cv2.SOLVEPNP_ITERATIVE,
        cv2.SOLVEPNP_EPNP,
        cv2.SOLVEPNP_DLS,
        cv2.SOLVEPNP_UPNP
    ]

    if verbose:
        for i in range(len(camera_bp3d)):
            vis.add_geometry(create_sphere(center=camera_bp3d[i], radius=0.005, color=[0, 1, 0]))

    # print("Camera Base Point 3D: \n", camera_bp3d)
    # print("Camera Base Point 2D: \n", camera_bp2d)

    min_rep_error, extrinsic = 1e10, None  
    for algorithm in algorithms:
        retval, rvec, tvec = cv2.solvePnP(camera_bp3d.astype(np.float32), camera_bp2d.astype(np.float32), intrinsic.astype(np.float32), dist_coeffs, flags=algorithm)
        if (retval == False) or (rvec is None) or (tvec is None):
            continue
        projected_points, _ = cv2.projectPoints(camera_bp3d, rvec, tvec, intrinsic, dist_coeffs)
        rep_error = np.linalg.norm(projected_points[:,0] - camera_bp2d, axis=-1).mean()
        if rep_error < min_rep_error:
            min_rep_error = rep_error
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = cv2.Rodrigues(rvec)[0]
            extrinsic[:3, 3] = tvec.flatten()
    if extrinsic is None:
        raise ValueError("Solver PnP and Extrinsic failed.")
    else:
        print(f"Solver PnP and Extrinsic succeed with rep_error: {min_rep_error}")

    extrinsic = fast_mat_inv(extrinsic)      # from W2C to C2W
    base2camera = fast_mat_inv(extrinsic)    # base(0,0)2camera

    print("Extrinsic: ")
    print("Extrinsic_pos: ", extrinsic[:3, 3])
    print("Extrinsic_rot: ", Rotation.from_matrix(extrinsic[:3, :3]).as_euler("xyz", degrees=True))
    print(extrinsic)

    if verbose:
        vis.add_geometry(create_sphere(center=extrinsic[:3,3], radius=0.01, color=[0, 1, 0]))
        vis.add_geometry(create_coordinate(origin=extrinsic[:3,3], orientation=extrinsic[:3,:3], size=0.1))

    quest2camera_list = []
    for i in range(len(quest2base_cs_list)):
        quest2base_cs = quest2base_cs_list[i]
        base_cs = camera_bp3d[args.quest_base_camera_corner_id[i]]
        print("Base Corner Point: ", base_cs)
        base = camera_bp3d[0]
        base_cs2base = np.eye(4) 
        base_cs2base[:3, :3] = np.eye(3)
        base_cs2base[:3, 3] = base_cs - base
        base_cs2base[3, 3] = 1
        quest2camera = base2camera @ base_cs2base @ quest2base_cs
        # print(f"Solution{i}:\n{quest2camera}")
        quest2camera_list.append(quest2camera)
        # import pdb; pdb.set_trace()
        # mat = quest2base 
        # print(Rotation.from_matrix(mat[:3,:3]).as_euler("xyz", degrees=True))

    quest2camera = np.mean(np.array(quest2camera_list), axis=0)
    rotation = Rotation.from_matrix(quest2camera[:3, :3]).as_euler("xyz", degrees=True)
    print("Quest to Camera Rotation (degree): ", rotation)
    print("Quest to Camera Translation (m): ", quest2camera[:3, 3])
    np.save(os.path.join(args.save_dir, "calib_result_quest2camera.npy"), quest2camera)

    resolution = np.load(os.path.join(args.save_dir, "camera_resolution.npy"))
    intrinsic = intrinsic_transform_resize(intrinsic, input_res=(1280, 720), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop)
    resolution = out_resolutions_crop
    camera2quest = fast_mat_inv(quest2camera)
    mount_pos = camera2quest[:3, 3]
    mount_rot = Rotation.from_matrix(camera2quest[:3, :3]).as_euler("xyz", degrees=True)
    mount_pos_x = np.round(mount_pos[0], 3)
    mount_pos_y = np.round(-1.0 * mount_pos[1], 3)
    mount_pos_z = np.round(mount_pos[2], 3)
    mount_rot_x = np.round(-1.0 * mount_rot[0], 3)
    mount_rot_y = np.round(mount_rot[1], 3)
    mount_rot_z = np.round(-1.0 * mount_rot[2], 3)
    pixel_pos_0 = np.array([0.0, 0.0, 1.0])
    pixel_pos_1 = np.array([resolution[0], resolution[1], 1.0])
    preset_depth_unity_bound = 5.0
    bound_pos_0 = preset_depth_unity_bound * np.linalg.inv(intrinsic) @ pixel_pos_0
    bound_pos_1 = preset_depth_unity_bound * np.linalg.inv(intrinsic) @ pixel_pos_1
    bound_width = np.round(bound_pos_1[0] - bound_pos_0[0], 3)
    bound_height = np.round(bound_pos_1[1] - bound_pos_0[1], 3)
    mount_sol = {
        "mount_pos_x": mount_pos_x,
        "mount_pos_y": mount_pos_y,
        "mount_pos_z": mount_pos_z,
        "mount_rot_x": mount_rot_x,
        "mount_rot_y": mount_rot_y,
        "mount_rot_z": mount_rot_z,
        "bound_width": bound_width,
        "bound_height": bound_height
    }
    with open(os.path.join(args.save_dir, "calib_mount_solution.json"), "w") as f:
        json.dump(mount_sol, f, indent=4)
    print("############### Mount Parameter Solution ##################")
    print("Mount Position (x, y, z): ", mount_pos_x, mount_pos_y, mount_pos_z)
    print("Mount Rotation (x, y, z): ", mount_rot_x, mount_rot_y, mount_rot_z)
    print("Mount Bound (width, height): ", bound_width, bound_height)
    print("############################################################")
    print("Calibration result saved in: ", os.path.join(args.save_dir, "calib_result_quest2camera.npy"))

    if verbose:
        while True:
            if keyboard.is_pressed('q'):
                break
            vis.poll_events()
            vis.update_renderer()
        vis.destroy_window()

# Calibration Logic:
# step1: unify the coordinate system for base point from quest and camera. 
# step2: HeadSet --(Calculate)--> Base --(PnP)--> Camera

# Inference:
# 1. hand_pose, quest_pose --- yfxrzu2standard ---> hand_pose_canonical, quest_pose_canonical
# 2. hand_pose_canonical --- quest_pose_canonical ---> hand2quest_canonical
# 3. hand2quest_canonical --- quest2camera ---> hand2camera
# 4. hand_pose = hand2camera
#    camera_motion = Delta(quest_pose_canonical)

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--camera_params_dir', type=str, default="camera_params")
    arg_parser.add_argument('--camera', type=str, default='zed', help="zed, ")
    arg_parser.add_argument('-i', '--instrinisc_file', type=str, default="camera_intrinsic.npy")
    # The order of corners should be "N" shape, from down-left to up-right. 
    arg_parser.add_argument('-r', '--camera_bp3d_reverse', action="store_true", help="If True, reverse camera_bp3d list from camera-calibration file.")
    arg_parser.add_argument('-b', '--quest_base_camera_corner_id', type=str, default="1,14", help="which corner of the camera image is the quest anchor point.")
    arg_parser.add_argument('--resolution_resize', type=str, default='1280x720')
    arg_parser.add_argument('--resolution_crop', type=str, default='640x480')
    
    args = arg_parser.parse_args()
    args.quest_base_camera_corner_id = convert_list(args.quest_base_camera_corner_id)
    print(args.quest_base_camera_corner_id)
    
    args.save_dir = os.path.join(args.camera_params_dir, "quest_"+args.camera)
    out_resolutions_resize = tuple(map(int, args.resolution_resize.split('x')))
    out_resolutions_crop = tuple(map(int, args.resolution_crop.split('x')))

    calibration(args, out_resolutions_resize, out_resolutions_crop, verbose=True)


# python -m scripts_data.vr_calibration_device_result_calculation -i camera_intrinsic.npy -b 17 --resolution_resize 1280x720 --resolution_crop 640x480