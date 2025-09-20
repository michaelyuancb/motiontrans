import cv2
import numpy as np
import os
from ego_hos_wrapper import EgoHOSWrapper
import re
from tqdm import tqdm



if __name__ == "__main__":
    franka_mask_path = "./left_mask.png"
    franka_seg_path = "./left_franka.png"
    franka_mask = cv2.imread(franka_mask_path)
    franka_mask_resized = cv2.resize(franka_mask, (320, 240))

    franka_seg = cv2.imread(franka_seg_path)
    franka_seg_resized = cv2.resize(franka_seg, (320, 240))

    video_path = "/home/zhourui/Desktop/user/project/bi-dex-mimic/dex_mimic/human_data/work_dirs/ego_hos_cache/video_inpaint_cache/video_img_cache/inpaint_out.mp4"
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
  # 视频编码格式
    out_video_path = "rendered_video.mp4"  # 输出视频路径
    video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, frame_size)

    frame_count = 0
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束
        print(frame.shape)

        mask_back_ground = cv2.bitwise_not(franka_mask_resized)
        img_back_ground = cv2.bitwise_and(frame, mask_back_ground)
        img_rander = img_back_ground + franka_seg_resized

        # 将渲染后的帧写入视频
        video_writer.write(img_rander)

        # 可选：显示渲染后的帧
        # cv2.imshow("franka_rander", img_rander)
        # cv2.waitKey(1)

        frame_count += 1

    # 释放资源
    cap.release()
    video_writer.release()  # 释放视频写入对象
    print(f"Frames extraction and video rendering complete. Rendered video saved to {out_video_path}")