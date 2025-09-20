import time
import multiprocessing as mp
import numpy as np
from threadpoolctl import threadpool_limits

import pygame
import pygame.surfarray

class MultiCameraVisualizer(mp.Process):
    def __init__(self,
        camera,
        row, col, rw, rh,
        window_name='Single Depth Cam Vis',
        vis_fps=60,
        fill_value=0,
        rgb_to_bgr=False
        ):
        super().__init__()
        self.row = row
        self.col = col
        self.rw = rw 
        self.rh = rh
        self.window_name = window_name
        self.vis_fps = vis_fps
        self.fill_value = fill_value
        self.rgb_to_bgr=rgb_to_bgr
        self.camera = camera
        # shared variables
        self.stop_event = mp.Event()

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self, wait=False):
        print(f"VIS_FPS: {self.vis_fps}")
        super().start()
    
    def stop(self, wait=False):
        self.stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self):
        pass

    def stop_wait(self):
        self.join()        
    
    def run(self):

        threadpool_limits(2)

        pygame.init()
        screen = pygame.display.set_mode((self.rw * self.col, self.rh * self.row))
        pygame.display.set_caption(self.window_name)
        clock = pygame.time.Clock()

        channel_slice = slice(None)
        if self.rgb_to_bgr:
            channel_slice = slice(None,None,-1)

        vis_data = None
        vis_img = None
        while not self.stop_event.is_set():
            vis_data = self.camera.get_vis(out=None)
            color = vis_data['rgb']
            N, H, W, C = color.shape
            assert C == 3
            oh = H * self.row
            ow = W * self.col
            if vis_img is None:
                vis_img = np.full((oh, ow, 3), 
                    fill_value=self.fill_value, dtype=np.uint8)
            for row in range(self.row):
                for col in range(self.col):
                    idx = col + row * self.col
                    h_start = H * row
                    h_end = h_start + H
                    w_start = W * col
                    w_end = w_start + W
                    if idx < N:
                        # opencv uses bgr
                        vis_img[h_start:h_end,w_start:w_end
                            ] = color[idx,:,:,channel_slice]
        
            surface = pygame.surfarray.make_surface(vis_img.swapaxes(0, 1))  # Pygame 使用的列交换
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(self.vis_fps)
            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            #         self.stop_event.set()

        pygame.quit()
