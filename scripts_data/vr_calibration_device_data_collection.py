# https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
import numpy as np 
import cv2
import os
import time
import glob
import json
import argparse
import keyboard

from human_data.quest_recorder import QuestRecorder
from human_data.camera_zed_simple import CameraZedSimple

def calibration_camera(args):

    os.makedirs(os.path.join(args.save_dir, "vis"), exist_ok=True)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, args.square_size, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((args.checkboard_h*args.checkboard_w,3), np.float32)
    objp[:,:2] = np.mgrid[0:args.checkboard_h,0:args.checkboard_w].T.reshape(-1,2) * args.square_size
    objpoints = [] 
    imgpoints = [] 

    images = glob.glob(f'{args.save_dir}/images/*.png')
    images.sort()
    num_save = len(images)

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (args.checkboard_h, args.checkboard_w), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (args.checkboard_h, args.checkboard_w), corners2, ret)
            text = f"[{i+1}/{num_save}] Press S to save image_vis_{i}."
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('img', img)
            key = cv2.waitKey(500)
            cv2.imwrite(os.path.join(args.save_dir, "vis", f'calibration_image_vis_{i}.png'), img)
        else:
            print(f"Failed to find corners in {i}: {fname}")

    cv2.destroyAllWindows()

    return objpoints, imgpoints, gray


def calibration_processor(objpoints, imgpoints, gray):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.save(os.path.join(args.save_dir, "calib_intrinsic.npy"), mtx)
    np.save(os.path.join(args.save_dir, "calib_dist.npy"), dist)
    rvecs = np.array(rvecs)[:,:,0]
    tvecs = np.array(tvecs)[:,:,0]
    np.save(os.path.join(args.save_dir, "calib_rvecs.npy"), rvecs)
    np.save(os.path.join(args.save_dir, "calib_tvecs.npy"), tvecs)
    with open(os.path.join(args.save_dir, "calib_ret.txt"), 'w') as f:    # unit: pixel
        f.write(str(ret))
    return ret, mtx, dist, rvecs, tvecs


    # objpoints, imgpoints, gray = calibration_camera(args)   # List[(40, 3)]  List[(40, 1, 2)]
    # ret, mtx, dist, rvecs, tvecs = calibration_processor(objpoints, imgpoints, gray)
    # import pdb; pdb.set_trace()


def step1_anchor_camera(args):

    resolution = (1280, 720)
    if True:
        serial_number_list = CameraZedSimple.get_connected_devices_serial()
        print(serial_number_list)
        device_id = serial_number_list[0]
        camera = CameraZedSimple(
            device_id=device_id,
            resolution=resolution,
        )

        raw_intrinsic = camera.get_intrinsic_left_cam()
        raw_distoration = camera.get_intrinsic_left_dist()
        resolution = np.array([resolution[0], resolution[1]])
        np.save(os.path.join(args.save_dir, "camera_intrinsic.npy"), raw_intrinsic)
        np.save(os.path.join(args.save_dir, "camera_distoration.npy"), raw_distoration)
        np.save(os.path.join(args.save_dir, "camera_resolution.npy"), resolution)

        def get_camera_image_stream():
            try:
                last_camera_data = camera.recieve()
                return last_camera_data
            except Exception as e:
                print(f"Failed to get camera image: {e}")
                return None

    while get_camera_image_stream() is None: pass
    while True:
        camera_image = get_camera_image_stream()
        if camera_image is None:
            continue
        show_image = camera_image.copy()
        text = f"Press S to use current Camera Image to calibration."
        cv2.putText(show_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("SyncColorViewer", show_image)
        key = cv2.waitKey(1)
        
        if keyboard.is_pressed('s'):
            image_path = os.path.join(args.save_dir, "calib_camera_image.png")
            cv2.imwrite(image_path, camera_image)
            print(f"Image saved as {image_path}")
            cv2.destroyAllWindows()
            break 

    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, args.square_size, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((args.checkboard_h*args.checkboard_w,3), np.float32)
    objp[:,:2] = np.mgrid[0:args.checkboard_h,0:args.checkboard_w].T.reshape(-1,2) * args.square_size
    objpoints = [] 
    imgpoints = [] 
            
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (args.checkboard_h, args.checkboard_w), None)

    if ret == False:
        raise ValueError("Failed to find corners in the image. Please retry.")
    objpoints.append(objp)

    corners = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)
    cv2.drawChessboardCorners(img, (args.checkboard_h, args.checkboard_w), corners, ret)
    img_show = img.copy()
    text = f"Press S to confirm the calibration base points reference."
    cv2.putText(img_show, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    while True:
        cv2.imshow('img', img_show)
        key = cv2.waitKey(1)

        if keyboard.is_pressed('s'):
            image_path = os.path.join(args.save_dir, "calib_camera_image_base_point.png")
            cv2.imwrite(image_path, img)
            print(f"Image saved as {image_path}")
            cv2.destroyAllWindows()
            break 
    
    np.save(os.path.join(args.save_dir, "calib_camera_image_base_point.npy"), np.array(imgpoints))
    np.save(os.path.join(args.save_dir, "calib_camera_image_base_point_obj.npy"), np.array(objpoints))
    
    print("Anchor camera process finished.")


def step2_anchor_quest(args):
    
    quest = QuestRecorder(output_dir="tmp")
    base_point_calib_list = []
    head_calib_list = []    
    
    current_ts = time.time()

    while True:
        now = time.time()
        # TODO: May cause communication issues, need to tune on AR side.
        if now - current_ts < 1 / args.frequency: 
            continue
        else:
            current_ts = now

        status, xrhand, head_pose, _ = quest.receive(verbose=True)
        if status == "WorldFrame":
            base_point_pose, head_pose = xrhand 
            base_point_pose = base_point_pose.get_pose_str()
            head_pose = head_pose.get_pose_str()
            base_point_calib_list.append(base_point_pose)
            head_calib_list.append(head_pose)
            print(f"Get calib {len(base_point_calib_list)} base points.")
        
        if len(base_point_calib_list) == args.num_quest_basepoints:
            break

    with open(os.path.join(args.save_dir, "calib_quest_base_point.json"), "w") as f:
        json.dump(base_point_calib_list, f, indent=4)
    with open(os.path.join(args.save_dir, "calib_quest_head.json"), "w") as f:
        json.dump(head_calib_list, f, indent=4)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--camera_params_dir', type=str, default="camera_params")
    arg_parser.add_argument('--camera', type=str, default='zed', help="zed, ")
    arg_parser.add_argument('--checkboard_h', type=int, default=8)
    arg_parser.add_argument('--checkboard_w', type=int, default=5)
    arg_parser.add_argument('--square_size', type=int, default=30, help="mm")
    arg_parser.add_argument("-n", '--num_quest_basepoints', type=int, default=1)
    arg_parser.add_argument("--frequency", type=int, default=50)
    args = arg_parser.parse_args()

    args.save_dir = os.path.join(args.camera_params_dir, "quest_"+args.camera)
    os.makedirs(args.save_dir, exist_ok=True)

    step1_anchor_camera(args)
    step2_anchor_quest(args)


# python -m scripts_data.vr_calibration_device_data_collection --checkboard_h 8 --checkboard_w 5 --square_size 30 -n 1