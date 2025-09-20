from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import glob
import os
from tqdm import tqdm
import argparse
import copy
import shutil
from PIL import Image
import numpy as np 
from skimage.io import imsave
import pdb

def visualize_twohands(img, seg_result, alpha = 0.4):
    seg_color = np.zeros((img.shape))
    seg_color[seg_result == 0] = (0,    0,   0)     # background
    seg_color[seg_result == 1] = (255,  0,   0)     # left_hand
    seg_color[seg_result == 2] = (0,    0,   255)   # right_hand
    vis = img * (1 - alpha) + seg_color * alpha
    return vis

def visualize_cb(img, seg_result, alpha = 0.4):
    seg_color = np.zeros((img.shape))
    seg_color[seg_result == 0] = (0,    0,   0)     # background
    seg_color[seg_result == 1] = (255,  0,   0)     # contact 
    vis = img * (1 - alpha) + seg_color * alpha
    return vis

def visualize_twohands_obj1(img, seg_result, alpha = 0.4):
    seg_color = np.zeros((img.shape))
    seg_color[seg_result == 0] = (0,    0,   0)     # background
    seg_color[seg_result == 1] = (255,  0,   0)     # left_hand
    seg_color[seg_result == 2] = (0,    0,   255)   # right_hand
    seg_color[seg_result == 3] = (255,  0,   255)   # left_object1
    seg_color[seg_result == 4] = (0,    255, 255)   # right_object1
    seg_color[seg_result == 5] = (0,    255, 0)     # two_object1
    vis = img * (1 - alpha) + seg_color * alpha
    return vis

def visualize_twohands_obj2(img, seg_result, alpha = 0.4):
    seg_color = np.zeros((img.shape))
    seg_color[seg_result == 0] = (0,    0,   0)     # background
    seg_color[seg_result == 1] = (255,  0,   0)     # left_hand
    seg_color[seg_result == 2] = (0,    0,   255)   # right_hand
    seg_color[seg_result == 3] = (255,  0,   255)   # left_object1
    seg_color[seg_result == 4] = (0,    255, 255)   # right_object1
    seg_color[seg_result == 5] = (0,    255, 0)     # two_object1
    seg_color[seg_result == 6] = (255,  204, 255)   # left_object2
    seg_color[seg_result == 7] = (204,  255, 255)   # right_object2
    seg_color[seg_result == 8] = (204,  255, 204)   # two_object2
    vis = img * (1 - alpha) + seg_color * alpha
    return vis



class EgoHOSWrapper:

    def __init__(self, 
                 cache_path,
                 repo_path='.',
                 two_hands_cfg='work_dirs/seg_twohands_ccda/seg_twohands_ccda.py',
                 cb_cfg='work_dirs/twohands_to_cb_ccda/twohands_to_cb_ccda.py',
                 obj2_cfg='work_dirs/twohands_cb_to_obj2_ccda/twohands_cb_to_obj2_ccda.py',
                 two_hands_ckpt='work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth',
                 cb_ckpt='work_dirs/twohands_to_cb_ccda/best_mIoU_iter_76000.pth',
                 obj2_ckpt='work_dirs/twohands_cb_to_obj2_ccda/best_mIoU_iter_32000.pth',
                 device='cuda'
                 ):
        if not cache_path.startswith('/'):
            raise ValueError("Please use the absolute file path as the cache path.")
        self.cache_path = cache_path
        two_hands_cfg = repo_path + '/' + two_hands_cfg
        cb_cfg = repo_path + '/' + cb_cfg
        obj2_cfg  = repo_path + '/' + obj2_cfg
        two_hands_ckpt = repo_path + '/' + two_hands_ckpt
        cb_ckpt = repo_path + '/' + cb_ckpt
        obj2_ckpt = repo_path + '/' + obj2_ckpt
        self.two_hands_cfg = two_hands_cfg
        self.two_hands_ckpt = two_hands_ckpt
        self.cb_cfg = cb_cfg
        self.cb_ckpt = cb_ckpt
        self.obj2_cfg = obj2_cfg
        self.obj2_ckpt = obj2_ckpt
        self.device = device 

        self.hand_model = init_segmentor(self.two_hands_cfg, self.two_hands_ckpt, device=device)
        self.cb_model = init_segmentor(self.cb_cfg, self.cb_ckpt, device=device)
        self.obj2_model = init_segmentor(self.obj2_cfg, self.obj2_ckpt, device=device)

        self.test_images_fp = os.path.join(cache_path, "test_images")
        self.pred_twohands_fp = os.path.join(cache_path, "pred_twohands")
        self.pred_cb_fp = os.path.join(cache_path, "pred_cb")
        os.makedirs(self.test_images_fp, exist_ok=True)
        os.makedirs(self.pred_twohands_fp, exist_ok=True)
        os.makedirs(self.pred_cb_fp, exist_ok=True)

    def segment(self, image_fp, vis=False):

        filename = image_fp.replace("/", "_").replace('.jpg', '.png').replace('.jpeg', '.png')
        cache_fp = os.path.join(self.test_images_fp, filename)
        shutil.copyfile(image_fp, cache_fp)

        seg_hands = inference_segmentor(self.hand_model, cache_fp)[0]
        imsave(os.path.join(self.pred_twohands_fp, filename), seg_hands.astype(np.uint8))
        seg_cb = inference_segmentor(self.cb_model, cache_fp)[0]
        imsave(os.path.join(self.pred_cb_fp, filename), seg_cb.astype(np.uint8))
        seg_obj2 = inference_segmentor(self.obj2_model, cache_fp)[0]


        os.remove(os.path.join(self.test_images_fp, filename))
        os.remove(os.path.join(self.pred_twohands_fp, filename))
        os.remove(os.path.join(self.pred_cb_fp, filename))

        if vis is True:
            img = np.array(Image.open(image_fp))
            hand_vis = visualize_twohands(img, seg_hands)
            imsave("vis_hands.png", hand_vis.astype(np.uint8))
            cb_vis = visualize_cb(img, seg_cb)
            imsave("vis_cb.png", cb_vis.astype(np.uint8))
            obj2_vis = visualize_twohands_obj2(img, seg_obj2)
            imsave("vis_obj2.png", obj2_vis.astype(np.uint8))

        # pdb.set_trace()

        return seg_hands, seg_cb, seg_obj2