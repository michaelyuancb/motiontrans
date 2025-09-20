import time
import os
import argparse
from pathlib import Path

import numpy as np
#import keyboard
import open3d as o3d
from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation

import viser
import viser.extras
import viser.transforms as tf

from common.replay_buffer import ReplayBuffer
from human_data.constants import *
from common.pose_util import pose_to_mat
from human_data.utils import fast_mat_inv, create_coordinate
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()


def visualization_hand(server, point_nodes, frame_nodes, i, wrist_pose, finger_pos, is_right,
                       color=np.array([0, 0, 255]), point_size=0.01):
    if wrist_pose is not None:
        server.scene.add_frame(
            f"/frames/t{i}/wrist_{is_right}",
            position=wrist_pose[:3],
            axes_length=0.1,
            axes_radius=0.005,
            wxyz=tf.SO3.from_matrix(Rotation.from_rotvec(wrist_pose[3:]).as_matrix()).wxyz
        )

    # use_idx = [0, 4, 5, 6, 7, 8]
    if finger_pos is not None:
        pts = []
        for idx in range(len(finger_pos)):
            pts.append(finger_pos[idx])
        # import pdb; pdb.set_trace()
        pts = np.array(pts)
        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/hand_{is_right}",
                points=pts,
                colors=color,
                point_size=point_size,
                point_shape="rounded",
            ))


def visualization(
        videos,  # (T, H, W, C)
        intrinsic,  # (4, 4)
        pointclouds,  # List[(N, 6)], XYZ+RGB
        camera_poses,  # (T, 6)
        left_wrist_pose,  # (T, 6), or None for robot
        right_wrist_pose,  # (T, 6)
        left_finger_pos,  # (T, 5, 6), or None for robot
        right_finger_pos,  # (T, 5, 6), or None for robot
        frustum_downsample_factor: int = 16,
        share: bool = False,
) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    num_frames = len(camera_poses)

    print("Start Visualization")
    camera_poses = pose_to_mat(camera_poses)

    if len(pointclouds) > 0:
        # import pdb; pdb.set_trace()
        for i in range(num_frames):
            camera_p = camera_poses[i]
            pointclouds[i][:, :3] = (camera_p[:3, :3] @ pointclouds[i][:, :3, None])[:, :, 0] + camera_p[:3, 3]

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider(
            "Point size",
            min=0.001,
            max=0.02,
            step=1e-3,
            initial_value=0.005,
        )
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=15
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            # Toggle visibility.
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Add recording UI.
    with server.gui.add_folder("Recording"):
        gui_record_scene = server.gui.add_button("Record Scene")

    # Recording handler
    @gui_record_scene.on_click
    def _(_):
        gui_record_scene.disabled = True

        # Save the original frame visibility state
        original_visibility = [frame_node.visible for frame_node in frame_nodes]

        rec = server._start_scene_recording()
        rec.set_loop_start()

        # Determine sleep duration based on current FPS
        sleep_duration = 1.0 / gui_framerate.value if gui_framerate.value > 0 else 0.033  # Default to ~30 FPS

        if gui_show_all_frames.value:
            # Record all frames according to the stride
            stride = gui_stride.value
            frames_to_record = [i for i in range(num_frames) if i % stride == 0]
        else:
            # Record the frames in sequence
            frames_to_record = range(num_frames)

        for t in frames_to_record:
            # Update the scene to show frame t
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = (i == t) if not gui_show_all_frames.value else (i % gui_stride.value == 0)
            server.flush()
            rec.insert_sleep(sleep_duration)

        # set all invisible
        with server.atomic():
            for frame_node in frame_nodes:
                frame_node.visible = False

        # Finish recording
        bs = rec.end_and_serialize()

        # Save the recording to a file
        output_path = Path(f"./viser_result/recording_{str(data_path).split('/')[-1]}.viser")
        # make sure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(bs)
        print(f"Recording saved to {output_path.resolve()}")

        # Restore the original frame visibility state
        with server.atomic():
            for frame_node, visibility in zip(frame_nodes, original_visibility):
                frame_node.visible = visibility
        server.flush()

        gui_record_scene.disabled = False

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(np.array([np.pi, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes: list[viser.FrameHandle] = []
    point_nodes: list[viser.PointCloudHandle] = []

    for i in tqdm(range(num_frames)):
        if len(pointclouds) > 0:
            position, color = pointclouds[i][:, :3], pointclouds[i][:, 3:]
        rgb = videos[i]

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Place the point cloud in the frame.
        if len(pointclouds) > 0:
            point_nodes.append(
                server.scene.add_point_cloud(
                    name=f"/frames/t{i}/point_cloud",
                    points=position,
                    colors=color / 255.0,
                    point_size=gui_point_size.value,
                    point_shape="rounded",
                )
            )

        # Place the frustum.
        fov = 2 * np.arctan2(rgb.shape[0] / 2, intrinsic[0, 0])
        aspect = rgb.shape[1] / rgb.shape[0]
        server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.05,
            image=rgb[::frustum_downsample_factor, ::frustum_downsample_factor],
            wxyz=tf.SO3.from_matrix(camera_poses[i][:3, :3]).wxyz,
            position=camera_poses[i][:3, 3],
        )

        # Add some axes.
        server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=0.1,
            axes_radius=0.005,
        )
        # wrist_pose, finger_pos
        left_wrist_pose_i = None if left_wrist_pose is None else left_wrist_pose[i]
        left_finger_pos_i = None if left_finger_pos is None else left_finger_pos[i]
        right_wrist_pose_i = right_wrist_pose[i]
        right_finger_pos_i = None if right_finger_pos is None else right_finger_pos[i]
        visualization_hand(server, point_nodes, frame_nodes, i, left_wrist_pose_i, left_finger_pos_i, is_right=False,
                           color=np.array([255., 0., 0.]), point_size=0.0075)
        visualization_hand(server, point_nodes, frame_nodes, i, right_wrist_pose_i, right_finger_pos_i, is_right=True,
                           color=np.array([0., 0., 255.]), point_size=0.0075)

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        # Update the timestep if we're playing.
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        # Update point size of both this timestep and the next one! There's
        # redundancy here, but this will be optimized out internally by viser.
        #
        # We update the point size for the next timestep so that it will be
        # immediately available when we toggle the visibility.
        point_nodes[gui_timestep.value].point_size = gui_point_size.value
        point_nodes[
            (gui_timestep.value + 1) % num_frames
            ].point_size = gui_point_size.value

        time.sleep(1.0 / gui_framerate.value)


def main(args):
    if args.data_path is None:
        raise ValueError("Please provide the path to the data directory.")

    if args.data_path.endswith(".zarr"):
        replay_buffer = ReplayBuffer.create_from_path(zarr_path=args.data_path, mode='r')
        episode = replay_buffer.get_episode(args.episode_idx)
    else:
        raise NotImplementedError("Only support zarr format for now.")

    if not args.disable_pointclouds:
        pointclouds = episode['camera0_pointcloud']
    videos = episode['camera0_rgb']
    if "camera0_pose" in episode.keys():
        camera_poses = episode['camera0_pose']
    else:
        camera_poses = np.zeros((len(episode['robot0_eef_pos']), 6))
    if "camera0_left_intrinsic" in episode.keys():
        intrinsic = episode['camera0_left_intrinsic']
    else:
        intrinsic = np.array([[640, 0, 320], [0, 480, 240], [0, 0, 1]]).astype(np.float32)

    if args.disable_pointclouds:
        print("Disable pointclouds visualization.")
        pointclouds_vis = []
    else:
        pointclouds_vis = []
        for t in tqdm(range(len(pointclouds)), desc="PointClouds-Process"):
            useful = (~np.isinf(pointclouds[t])) & (~np.isnan(pointclouds[t]))
            useful = (useful.sum(axis=-1) == 3.0)
            video_ds_0 = videos[t].shape[0] // pointclouds[t].shape[0]
            video_ds_1 = videos[t].shape[1] // pointclouds[t].shape[1]
            xyz, rgb = pointclouds[t][useful], videos[t][::video_ds_0, ::video_ds_1][useful]
            xyz, rgb = xyz.reshape(-1, 3), rgb.reshape(-1, 3)
            distance = np.linalg.norm(xyz, axis=-1)
            rgb = rgb[distance < 2.0]
            xyz = xyz[distance < 2.0]
            rgb = rgb[::args.downsample_factor]
            xyz = xyz[::args.downsample_factor]
            pointclouds_vis.append(np.concatenate([xyz, rgb], axis=-1))
            # pointclouds_vis.append(pointclouds[t])

    if args.precheck_pointclouds and not args.disable_pointclouds:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Data Visualizer, Press Q to Exist", width=800, height=600)
        pcd_xyz, pcd_rgb = [], []
        # coord = create_coordinate(np.zeros(3), np.eye(3), size=0.4)
        # vis.add_geometry(coord)
        interval = len(pointclouds_vis) // 8
        print(len(pointclouds_vis))
        for i, pcd in enumerate(pointclouds_vis):

            camera_pose = pose_to_mat(camera_poses[i])
            coord = create_coordinate(camera_pose[:3, 3], camera_pose[:3, :3], size=0.15)
            vis.add_geometry(coord)

            if i % interval > 0:
                continue

            print(f"pos{i}", camera_pose[:3, 3])
            print(f"rot{i}", Rotation.from_matrix(camera_pose[:3, :3]).as_euler("xyz", degrees=True))
            if True:
                print("Frame", i)
                # import pdb; pdb.set_trace()
                xyz, rgb = pointclouds_vis[i][..., :3], pointclouds_vis[i][..., 3:]
                xyz = (camera_pose[:3, :3] @ xyz[:, :, None])[:, :, 0] + camera_pose[:3, 3]
                pcd_xyz.append(xyz)
                pcd_rgb.append(rgb)
        pcd_xyz = np.concatenate(pcd_xyz, axis=0)
        pcd_rgb = np.concatenate(pcd_rgb, axis=0)
        print(pcd_xyz.shape, pcd_rgb.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb / 255)
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

    if 'robot0_eef_pos' in episode.keys():
        right_wrist_pos = episode['robot0_eef_pos']
        right_wrist_rot = episode['robot0_eef_rot_axis_angle']
        right_wrist_pose = np.concatenate([right_wrist_pos, right_wrist_rot], axis=-1)
        # right_wrist_pose = episode['right_wrist_pose']
    else:
        raise ValueError("Right Wrist Pose / Robot0 EEF Pose should be implemented, but not found now.")

    if 'robot1_eef_pos' in episode.keys():
        left_wrist_pos = episode['robot1_eef_pos']
        left_wrist_rot = episode['robot1_eef_rot_axis_angle']
        left_wrist_pose = np.concatenate([left_wrist_pos, left_wrist_rot], axis=-1)
        # left_wrist_pose = episode['left_wrist_pose']
    else:
        left_wrist_pose = None

    if 'left_hand_pose' in episode.keys():
        left_finger_pose = episode['left_hand_pose']
        if left_finger_pose.ndim == 3:
            left_finger_pose = left_finger_pose.reshape(left_finger_pose.shape[0], -1)
        n_finger = len(left_finger_pose[0]) // 6
        left_finger_pos = []
        for i in range(n_finger):
            left_finger_pos.append(left_finger_pose[:, i * 6:i * 6 + 3])
        left_finger_pos = np.stack(left_finger_pos, axis=1)
    else:
        left_finger_pos = None

    if 'right_hand_pose' in episode.keys():
        right_finger_pose = episode['right_hand_pose']
        if right_finger_pose.ndim == 3:
            right_finger_pose = right_finger_pose.reshape(right_finger_pose.shape[0], -1)
        n_finger = len(right_finger_pose[0]) // 6
        right_finger_pos = []
        for i in range(n_finger):
            right_finger_pos.append(right_finger_pose[:, i * 6:i * 6 + 3])
        right_finger_pos = np.stack(right_finger_pos, axis=1)
    else:
        right_finger_pos = None

    import imageio
    video_writer = imageio.get_writer("videos.mp4")
    for i in range(len(videos)):
        video_writer.append_data(videos[i])
    video_writer.close()


    visualization(videos, intrinsic, pointclouds_vis, camera_poses, left_wrist_pose, right_wrist_pose, left_finger_pos,
                  right_finger_pos)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("Visualize QuestRecord data.")
    arg_parser.add_argument("--data_path", type=str, help="Path to the data directory.")
    arg_parser.add_argument("-i", "--episode_idx", type=int, default=0, help="Episode index.")
    arg_parser.add_argument("--downsample_factor", type=int, default=4,
                            help="Downsample factor for rgb/depth visualization.")
    arg_parser.add_argument("-d", "--disable_pointclouds", action="store_true", help="Also visualization pointclouds.")
    arg_parser.add_argument("-p", "--precheck_pointclouds", action="store_true", help="Also visualization pointclouds.")
    args = arg_parser.parse_args()

    main(args)

# python -m scripts_data.entry.data_visualization -d /home/zhourui/Desktop/user/project/dex-mimic/data/data_robot_processed/data_robot_raw_1280x720_640x480.zarr -p