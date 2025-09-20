import cv2
import numpy as np
import os
from ego_hos_wrapper import EgoHOSWrapper
import re
from tqdm import tqdm
import click
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image.*")


def extract_frames(video_path, output_dir, frame_interval=1):
    """
    使用 OpenCV 逐帧抽取视频并保存
    :param video_path: 输入视频路径
    :param output_dir: 输出帧保存目录
    :param frame_interval: 帧间隔，表示每隔多少帧抽取一帧，默认为1（逐帧抽取）
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Unable to open the video")
        return

    frame_count = 0
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        # 按照帧间隔保存帧
        if frame_count % frame_interval == 0:
            frame_name = f"{frame_count:05d}.jpg"
            output_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(output_path, frame)
            print(f"save frame: {output_path}")

        frame_count += 1

    # 释放资源
    cap.release()
    print("Frames extraction is complete")

@click.command()
@click.option('--input', '-i', required=True, help='Path to video')
@click.option('--output', '-o', required=True, help='Path to seg image')
@click.option('--image_cache', '-rc', required=True, help='Path to image cache')
def main(input,
         output,
         image_cache
         ):

    # video_image_cache_dir = "work_dirs/ego_hos_cache/video_img_cache"
    # video_seg_cache_dir = "work_dirs/ego_hos_cache/video_seg_cache"
    extract_frames(video_path=input,
                   output_dir=image_cache)

    if not os.path.exists(output):
        os.makedirs(output)

    # 初始化 EgoHOSWrapper
    ego_hos_wrapper = EgoHOSWrapper(
        cache_path="/home/zhourui/Desktop/user/project/bi-dex-mimic/dex_mimic/human_data/work_dirs/ego_hos_cache",
        # an absolute file-path for caching
        repo_path='.')

    # 获取所有jpg文件并按数字顺序排序
    images = [img for img in os.listdir(image_cache) if img.endswith(".jpg")]

    # 使用正则表达式提取数字部分并转换为整数进行排序
    images.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    for image in tqdm(images, desc="Segment", unit="frame"):
        img_path = os.path.join(image_cache, image)

        # 进行分割
        seg_hands, seg_cb, seg_obj2 = ego_hos_wrapper.segment(img_path, vis=False)
        seg_hands = seg_hands.astype(np.uint8) * 255

        # 保存分割结果
        base_name = os.path.splitext(image)[0]
        cv2.imwrite(os.path.join(output, f"{base_name}.png"), seg_hands)

    print(f"segmentation saved in dir : {output}")


if __name__ == "__main__":
    main()



