from typing import Optional, Callable, Generator
import numpy as np
import av
from common.timestamp_accumulator import get_accumulate_timestamp_idxs


def read_depth_bin(file_path, frame_shape, dtype)
file_path = 'depth_video_file.bin'

# 每帧的形状和数据类型
frame_shape = (480, 640)  # 高度和宽度
dtype = np.uint16  # 数据类型

# 计算每帧的字节数
frame_size = np.prod(frame_shape) * np.dtype(dtype).itemsize

# 打开文件并逐块读取数据
with open(file_path, 'rb') as f:
    while True:
        block = f.read(frame_size)
        if not block:
            break
        frame = np.frombuffer(block, dtype=dtype).reshape(frame_shape)
        print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")


def read_video(
        video_path: str, dt: float,
        video_start_time: float=0.0, 
        start_time: float=0.0,
        img_transform: Optional[Callable[[np.ndarray], np.ndarray]]=None,
        thread_type: str="AUTO",
        thread_count: int=0,
        max_pad_frames: int=10
        ) -> Generator[np.ndarray, None, None]:
    frame = None
    
    depth_video_path = video_path[:-4] + "_depth.bin"
    if os.path.exist(depth_video_path):


    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = thread_type
        stream.thread_count = thread_count
        next_global_idx = 0
        for frame_idx, frame in enumerate(container.decode(stream)):
            # The presentation time in seconds for this frame.
            since_start = frame.time
            frame_time = video_start_time + since_start
            local_idxs, global_idxs, next_global_idx \
                = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=start_time,
                dt=dt,
                next_global_idx=next_global_idx
            )
            if len(global_idxs) > 0:
                array = frame.to_ndarray(format='rgb24')
                img = array
                if img_transform is not None:
                    img = img_transform(array)
                for global_idx in global_idxs:
                    yield img
    # repeat last frame max_pad_frames times
    array = frame.to_ndarray(format='rgb24')
    img = array
    if img_transform is not None:
        img = img_transform(array)
    for i in range(max_pad_frames):
        yield img