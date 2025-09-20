from typing import Tuple
import math
import cv2
import numpy as np

def draw_reticle(img, u, v, label_color):
    """
    Draws a reticle (cross-hair) on the image at the given position on top of
    the original image.
    @param img (In/Out) uint8 3 channel image
    @param u X coordinate (width)
    @param v Y coordinate (height)
    @param label_color tuple of 3 ints for RGB color used for drawing.
    """
    # Cast to int.
    u = int(u)
    v = int(v)

    white = (255, 255, 255)
    cv2.circle(img, (u, v), 10, label_color, 1)
    cv2.circle(img, (u, v), 11, white, 1)
    cv2.circle(img, (u, v), 12, label_color, 1)
    cv2.line(img, (u, v + 1), (u, v + 3), white, 1)
    cv2.line(img, (u + 1, v), (u + 3, v), white, 1)
    cv2.line(img, (u, v - 1), (u, v - 3), white, 1)
    cv2.line(img, (u - 1, v), (u - 3, v), white, 1)


def draw_text(
    img,
    *,
    text,
    uv_top_left,
    color=(255, 255, 255),
    fontScale=0.5,
    thickness=1,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color=(0, 0, 0),
    line_spacing=1.5,
):
    """
    Draws multiline with an outline.
    """
    assert isinstance(text, str)

    uv_top_left = np.array(uv_top_left, dtype=float)
    assert uv_top_left.shape == (2,)

    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]
        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left += [0, h * line_spacing]


def get_image_transform(
        input_res: Tuple[int,int]=(1280,720), 
        output_res: Tuple[int,int]=(640,480), 
        bgr_to_rgb: bool=False,
        is_depth: bool=False,
        is_pcd: bool=False):

    iw, ih = input_res
    ow, oh = output_res
    rw, rh = None, None
    interp_method = cv2.INTER_AREA

    if (iw/ih) >= (ow/oh):
        # input is wider
        rh = oh
        rw = math.ceil(rh / ih * iw)
        if oh > ih:
            interp_method = cv2.INTER_LINEAR
    else:
        rw = ow
        rh = math.ceil(rw / iw * ih)
        if ow > iw:
            interp_method = cv2.INTER_LINEAR
    
    if is_depth is True or is_pcd is True:
        interp_method = cv2.INTER_NEAREST
    
    w_slice_start = (rw - ow) // 2
    w_slice = slice(w_slice_start, w_slice_start + ow)
    h_slice_start = (rh - oh) // 2
    h_slice = slice(h_slice_start, h_slice_start + oh)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        if is_depth is False:
            assert img.shape == ((ih,iw,3))
        else:
            # assert img.shape == ((ih,iw))
            pass
        # resize
        img = cv2.resize(img, (rw, rh), interpolation=interp_method)
        # crop
        if is_depth is False:
            img = img[h_slice, w_slice, c_slice]
        else:
            img = img[h_slice, w_slice]
        return img
    return transform


def get_image_transform_resize_crop(
        input_res: Tuple[int,int]=(1280,720), 
        output_resize_res: Tuple[int,int]=(1280,720), 
        output_crop_res: Tuple[int,int]=(640,480), 
        bgr_to_rgb: bool=False,
        is_depth: bool=False,
        is_pcd: bool=False,
        crop_bias: Tuple[int,int]=(0,0)):

    transform_resize = get_image_transform(
        input_res=input_res, output_res=output_resize_res, bgr_to_rgb=bgr_to_rgb, is_depth=is_depth, is_pcd=is_pcd)
    
    def transform(img: np.ndarray):
        img = transform_resize(img)
        th = (img.shape[0] - output_crop_res[1]) // 2 + crop_bias[1]
        tw = (img.shape[1] - output_crop_res[0]) // 2 + crop_bias[0]
        img = img[th:th+output_crop_res[1], tw:tw+output_crop_res[0]]
        return img

    return transform


def intrinsic_transform_resize(intrinsic, input_res, output_resize_res, output_crop_res):

    iw, ih = input_res
    ow, oh = output_resize_res
    rw, rh = None, None
    if (iw/ih) >= (ow/oh):
        # input is wider
        rh = oh
        rw = math.ceil(rh / ih * iw)
    else:
        rw = ow
        rh = math.ceil(rw / iw * ih)
    
    intrinsic[0] = intrinsic[0] * rw / ow 
    intrinsic[1] = intrinsic[1] * rh / oh

    cw, ch = output_crop_res
    intrinsic[0, 2] = intrinsic[0, 2] - (rw - cw) / 2
    intrinsic[1, 2] = intrinsic[1, 2] - (rh - ch) / 2

    return intrinsic



def get_image_transform_param(
        input_res: Tuple[int,int]=(1280,720), 
        resize_res: Tuple[int,int]=(1280,720),
        output_res: Tuple[int,int]=(640,480), 
        bgr_to_rgb: bool=False,
        is_depth: bool=False):

    iw, ih = input_res
    ow, oh = resize_res
    cw, ch = output_res
    rw, rh = None, None
    interp_method = cv2.INTER_AREA

    if (iw/ih) >= (ow/oh):
        # input is wider
        rh = oh
        rw = math.ceil(rh / ih * iw)
        if oh > ih:
            interp_method = cv2.INTER_LINEAR
    else:
        rw = ow
        rh = math.ceil(rw / iw * ih)
        if ow > iw:
            interp_method = cv2.INTER_LINEAR
    if is_depth is True:
        interp_method = cv2.INTER_NEAREST
    
    w_slice_start = (rw - cw) // 2
    w_slice = slice(w_slice_start, w_slice_start + cw)
    h_slice_start = (rh - ch) // 2
    h_slice = slice(h_slice_start, h_slice_start + ch)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)
    param = {
        "ih": ih, "iw": iw, "rh": rh, "rw": rw,
        "w_slice": w_slice, "h_slice": h_slice, "c_slice": c_slice,
        "interp_method": interp_method,
        "is_depth": is_depth
    }
    return param


def image_transform(img: np.ndarray, param: dict):
    if param['is_depth'] is False:
        assert img.shape == ((param['ih'],param['iw'],3))
    else:
        # assert img.shape == ((param['ih'],param['iw']))
        pass
    if param['ih'] != img.shape[0] or param['iw'] != img.shape[1]:
        img = cv2.resize(img, (param['rw'],param['rh']), interpolation=param['interp_method'])
    if param['is_depth'] is False:
        img = img[param['h_slice'], param['w_slice'], param['c_slice']]
    else:
        img = img[param['h_slice'], param['w_slice']]
    return img


class ImageTransform:

    def __init__(self, input_res, resize_res, output_res, bgr_to_rgb=False):
        
        self.input_res = input_res
        self.resize_res = resize_res
        self.output_res = output_res 
        self.bgr_to_rgb = bgr_to_rgb
        self.f_param = get_image_transform_param(input_res=input_res, resize_res=resize_res, output_res=output_res, bgr_to_rgb=bgr_to_rgb)


    def __call__(self, data):

        keys = data.keys() 
        color_keys = [k for k in keys if k.startswith("rgb")]
        for k in color_keys:
            img = data[k]
            img = image_transform(img, self.f_param)
            data[k] = img
        return data


class ImageDepthTransform:

    def __init__(self, input_res, resize_res, output_res, bgr_to_rgb=False):
        
        self.input_res = input_res
        self.resize_res = resize_res
        self.output_res = output_res 
        self.bgr_to_rgb = bgr_to_rgb
        self.f_param = get_image_transform_param(input_res=input_res, resize_res=resize_res, output_res=output_res, bgr_to_rgb=bgr_to_rgb)
        self.fd_param = get_image_transform_param(input_res=input_res, resize_res=resize_res, output_res=output_res, bgr_to_rgb=bgr_to_rgb, is_depth=True)


    def __call__(self, data):

        img = data['rgb']
        key = None
        if 'depth' in data.keys():
            key = 'depth'
        elif 'pointcloud' in data.keys():
            key = 'pointcloud'
        if key is not None:
            dep = data[key]
            dep = image_transform(dep, self.fd_param)
            data[key] = dep

        color_keys = [k for k in data.keys() if k.startswith("rgb")]
        for k in color_keys:
            img = data[k]
            img = image_transform(img, self.f_param)
            data[k] = img
        return data
        

# class ImageDepthVisTransform:

#     def __init__(self, input_res, output_res, bgr_to_rgb=False):
            
#         self.input_res = input_res
#         self.resize_res = resize_res
#         self.output_res = output_res 
#         self.bgr_to_rgb = bgr_to_rgb
#         self.f_param = get_image_transform_param(input_res=input_res, resize_res=resize_res, output_res=output_res, bgr_to_rgb=bgr_to_rgb)
#         self.fd_param = get_image_transform_param(input_res=input_res, resize_res=resize_res, output_res=output_res, bgr_to_rgb=bgr_to_rgb, is_depth=True)

#     def __call__(self, data):
#         img = data['rgb']
#         dep = data['depth']
#         img = image_transform(img, self.f_param)
#         dep = image_transform(dep, self.fd_param)
#         dep = dep / dep.max()
#         dep = cv2.applyColorMap((dep * 255).astype(np.uint8), cv2.COLORMAP_JET)
#         data['rgb'] = img
#         data['depth'] = dep
#         return data



def optimal_row_cols(
        n_cameras,
        in_wh_ratio,
        max_resolution=(1920, 1080)
    ):
    out_w, out_h = max_resolution
    out_wh_ratio = out_w / out_h    

    # import pdb; pdb.set_trace()
    n_rows = np.arange(n_cameras,dtype=np.int64) + 1
    n_cols = np.ceil(n_cameras / n_rows).astype(np.int64)
    cat_wh_ratio = in_wh_ratio * (n_cols / n_rows)
    ratio_diff = np.abs(out_wh_ratio - cat_wh_ratio)
    best_idx = np.argmin(ratio_diff)
    best_n_row = n_rows[best_idx]
    best_n_col = n_cols[best_idx]
    best_cat_wh_ratio = cat_wh_ratio[best_idx]

    rw, rh = None, None
    if best_cat_wh_ratio >= out_wh_ratio:
        # cat is wider
        rw = math.floor(out_w / best_n_col)
        rh = math.floor(rw / in_wh_ratio)
    else:
        rh = math.floor(out_h / best_n_row)
        rw = math.floor(rh * in_wh_ratio)
    
    # crop_resolution = (rw, rh)
    return rw, rh, best_n_col, best_n_row
