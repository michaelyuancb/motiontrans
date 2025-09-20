
from human_data.camera_zed_simple import CameraZedSimple
from common.pose_util import euler_pose_to_mat
import cv2
import numpy as np 
import pdb
import click


@click.command()
@click.option('--output', '-o', required=True)
@click.option('--resolution_crop', '-or', default='640x480')
def main(output, resolution_crop):
    serial_number_list = CameraZedSimple.get_connected_devices_serial()

    resolution = tuple(int(x) for x in resolution_crop.split('x'))
    device_id = serial_number_list[0]

    camera = CameraZedSimple(
        device_id=device_id,
        resolution=(1280, 720),
        capture_fps=15,
        num_threads=2,
        recording=True,
        recording_crop_w=resolution[0],
        recording_crop_h=resolution[1],
        recording_downsample_ratio=1,
        verbose=False
    )

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    while True:
        try:
            last_camera_data = camera.recieve(transform=True)
            print(last_camera_data.shape)
            cv2.imshow("Camera", last_camera_data[..., ::-1])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except KeyboardInterrupt:
            break
    
    print("please input the manual adjustment result")
    px = input("px:")
    py = input("py:")
    pz = input("pz:")
    rx = input("rx:")
    ry = input("ry:")
    rz = input("rz:")

    # Reference
    # mount_pos = camera2quest[:3, 3]
    # mount_rot = Rotation.from_matrix(camera2quest[:3, :3]).as_euler("xyz", degrees=True)
    # mount_pos_x = np.round(mount_pos[0], 3)
    # mount_pos_y = np.round(-1.0 * mount_pos[1], 3)
    # mount_pos_z = np.round(mount_pos[2], 3)
    # mount_rot_x = np.round(-1.0 * mount_rot[0], 3)
    # mount_rot_y = np.round(mount_rot[1], 3)
    # mount_rot_z = np.round(-1.0 * mount_rot[2], 3)
    # pixel_pos_0 = np.array([0.0, 0.0, 1.0])
    # pixel_pos_1 = np.array([resolution[0], resolution[1], 1.0])

    pdb.set_trace()

    camera2quest_mount_sol = np.eye(4)
    camera2quest_mount_pose = np.array([float(px), -1.0 * float(py), float(pz), -1.0 * float(rx), float(ry), -1.0 * float(rz)]) 
    camera2quest_mount_sol = euler_pose_to_mat(camera2quest_mount_pose)
    np.save(output, camera2quest_mount_sol)



if __name__ == "__main__":

    main()
    

# python -m scripts_data.vr_calibration_test_camera_view -o camera_params/quest_zed/calib_result_quest2camera_fix.npy -or 640x480