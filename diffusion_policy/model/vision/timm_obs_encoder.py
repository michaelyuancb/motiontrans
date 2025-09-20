import copy

import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
import os
import numpy as np
from PIL import Image
from functools import partial

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from common.pytorch_util import replace_submodules
from diffusion_policy.model.vision.choice_randomizer import RandomChoice
from timm.layers.attention_pool import AttentionPoolLatent

logger = logging.getLogger(__name__)

try:
    INFER_MODE = os.environ["INFER_MODE"]
except:
    INFER_MODE = False

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    

class TimmObsEncoder(ModuleAttrMixin):    # merge depth-anything-feature with fuse_model_name
    def __init__(self,
            shape_meta: dict,
            model_name: str,
            pretrained: bool,
            frozen: bool,
            global_pool: str,
            transforms: list,
            use_low_dim_encoder: bool=False,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            three_augment: bool=False,
            feature_aggregation: str='spatial_embedding',
            downsample_ratio: int=32,
            position_encording: str='learnable',
            use_lora: bool = False,
            lora_rank: int = 8,
            drop_path_rate: float = 0.0,
            fused_model_name: str = '',
            fused_model_feature_dim: int = 512,
            fused_model_ckpt: str = '',
            fused_model_frozen: bool = True,
            text_encoder_name: str = '',
            text_encoder_frozen: bool = False,
            text_encoder_feature_dim: int = 128,
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        embed_keys = list()
        embed_dims = list()
        embed_num_values = list()
        key_model_map = nn.ModuleDict()
        key_fused_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        key_eval_transform_map = nn.ModuleDict()

        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                if 'pointcloud' in key:
                    continue
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]

        ###################################### Main Vision Model #########################################
        assert global_pool == ''
        if 'resnet' in model_name:
            model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool,    # '' means no pooling
                num_classes=0,              # remove classification layer
            )
        else:
            model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool,    # '' means no pooling
                num_classes=0,              # remove classification layer
                img_size=image_shape[0],    # 224
                drop_path_rate=drop_path_rate,  # stochastic depth
            )

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False


        ###################################### Fused Vision Model #########################################
        fused_model = None
        if fused_model_name.startswith("dpt"):
            from diffusion_policy.model.depth_anything_v2.dpt import DepthAnythingV2
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }

            # dpt_vits, dpt_vitb, dpt_vitl
            encoder = fused_model_name.split("_")[1]
            dpt = DepthAnythingV2(**model_configs[encoder])
            if not INFER_MODE:
                dpt.load_state_dict(torch.load(fused_model_ckpt, map_location='cpu'))
            fused_model = dpt.pretrained
            for param in fused_model.parameters():
                param.requires_grad = not fused_model_frozen
        elif fused_model_name.startswith('pointnext_'):
            from diffusion_policy.model.vision.pointnext_simple_encoder import PointNeXtSimpleEncoder
            self.fused_model_feature_dim = fused_model_feature_dim
            fused_model = PointNeXtSimpleEncoder(fused_model_name, in_dim=6, out_dim=fused_model_feature_dim, dropout=0.1)
            for param in fused_model.parameters():
                param.requires_grad = not fused_model_frozen
        elif fused_model_name != '':
            assert feature_aggregation == 'map'
            fused_model = timm.create_model(
                model_name=fused_model_name,
                pretrained=True,
                global_pool=global_pool,
                num_classes=0,
                img_size=image_shape[0],
                drop_path_rate=0.0,
            )
            for param in fused_model.parameters():
                param.requires_grad = False


        ###################################### Text Encoder Model #########################################
        text_encoder_model = None
        text_feature_dim = -1
        text_input_feature_dim = -1
        if text_encoder_name.startswith('talk2dino'):
            # Talk2DINO text encoder, reference: https://github.com/lorebianchi98/Talk2DINO
            from diffusion_policy.model.talk2dino.talk2dino_wrapper import Talk2DINO_Wrapper
            
            # 获取当前文件的目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 获取current_dir的上一级目录
            parent_dir = os.path.dirname(current_dir)
            if 'base' in model_name:
                proj_name = 'vitb_mlp_infonce'
            elif 'large' in model_name:
                proj_name = 'vitl_mlp_infonce'
            else:
                raise ValueError(f"Unsupported model_name: {model_name} for Talk2DINO fused_model. Should be one that contains [base, large].")
            
            config_path = os.path.join(parent_dir, "talk2dino", "configs", f"{proj_name}.yaml")
            weights_path = os.path.join(parent_dir, "talk2dino", "weights", f"{proj_name}.pth")
            print("Out-Dim of Talk2DINO: ", text_encoder_feature_dim)
            text_encoder_model = Talk2DINO_Wrapper(config_path, weights_path, out_dim=text_encoder_feature_dim, frozen=text_encoder_frozen)
            text_feature_dim = text_encoder_model.out_dim
            text_input_feature_dim = 512
        
        key_model_map["instruction"] = text_encoder_model
        self.text_feature_dim = text_feature_dim
        self.text_input_feature_dim = text_input_feature_dim

        ###################################### Calculate Feature Dimension #########################################
        feature_dim = None
        num_heads = None
        if model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('convnext'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('vit'):
            feature_dim = model.num_features
            num_heads = model.blocks[0].attn.num_heads
            if fused_model_name.startswith('pointnext_'):
                feature_dim = feature_dim + fused_model_feature_dim
            elif fused_model_name != '':
                feature_dim = feature_dim + fused_model.num_features
        if text_encoder_model is not None: 
            feature_dim = feature_dim + text_feature_dim

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
            )
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
            if imagenet_norm:
                transforms = transforms + [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        eval_transforms = None
        if transforms is not None:
            eval_transforms = [torchvision.transforms.Resize(size=image_shape[0], antialias=True)]
            if imagenet_norm:
                eval_transforms = eval_transforms + [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        eval_transform = nn.Identity() if transforms is None else torch.nn.Sequential(*eval_transforms)

        if three_augment:
            # Following DeiT III: https://arxiv.org/abs/2204.07118
            primary_tfl = [
                torchvision.transforms.RandomCrop(image_shape[0], padding=4, padding_mode='reflect'),
            ]
            secondary_tfl = [
                RandomChoice([torchvision.transforms.Grayscale(num_output_channels=3),
                              torchvision.transforms.RandomSolarize(threshold=0.5, p=1.0),
                              torchvision.transforms.GaussianBlur(kernel_size=5)]),
                torchvision.transforms.ColorJitter(0.3, 0.3, 0.3)
            ]
            final_tfl = [
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            transform = torch.nn.Sequential(*primary_tfl, *secondary_tfl, *final_tfl)
            assert eval_transform is not None and eval_transform != nn.Identity()

        low_dim_feat_dim = 0 
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                if not attr.get('ignore_by_policy', False):
                    rgb_keys.append(key)

                    this_model = model if share_rgb_model else copy.deepcopy(model)
                    key_model_map[key] = this_model
                    key_fused_model_map[key] = fused_model

                    this_transform = transform
                    key_transform_map[key] = this_transform
                    key_eval_transform_map[key] = eval_transform
            elif type == 'low_dim':
                if not attr.get('ignore_by_policy', False):
                    embedding_dim = attr['embedding_dim']
                    if embedding_dim <= 0:
                        low_dim_keys.append(key)
                        low_dim_feat_dim = low_dim_feat_dim + int(attr['horizon'] * np.array(attr['shape']).prod())
                    else:
                        embed_keys.append(key)
                        embed_dims.append(embedding_dim)
                        num_value = attr['num_value']
                        embed_num_values.append(num_value)         
            elif type == 'camera':
                pass
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        feature_map_shape = [x // downsample_ratio for x in image_shape]
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:         ', rgb_keys)
        print('low_dim_keys keys:', low_dim_keys)
        print("embed_keys keys:", embed_keys)

        self.model_name = model_name
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_fused_model_map = key_fused_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.embed_keys = embed_keys
        self.embed_dims = embed_dims
        self.embed_num_values = embed_num_values
        self.key_shape_map = key_shape_map
        self.key_eval_transform_map = key_eval_transform_map
        self.feature_aggregation = feature_aggregation
        self.fused_model_name = fused_model_name

        self.use_low_dim_encoder = use_low_dim_encoder
        self.low_dim_feat_dim = low_dim_feat_dim
        if self.use_low_dim_encoder:
            assert low_dim_feat_dim > 4
            # This is a simple MLP neural network (with encoder-decoder / information bottleneck format) for low_dim features, 
            # which make it possible to conduct domain confusion and compression for low_dim features.
            self.low_dim_encoder = nn.Sequential(
                nn.Linear(low_dim_feat_dim, low_dim_feat_dim // 2),
                nn.ReLU(),
                nn.Linear(low_dim_feat_dim // 2, low_dim_feat_dim // 4),
                nn.ReLU(),
                nn.Linear(low_dim_feat_dim // 4, low_dim_feat_dim // 2),
                nn.ReLU(),
                nn.Linear(low_dim_feat_dim // 2, low_dim_feat_dim),
            )
        else:
            # By default, we do not use low_dim_encoder and rely on Diffusion Decoder to learn the low_dim representation.
            self.low_dim_encoder = nn.Identity()


        self.embed_keys = embed_keys
        self.embed_num_values = embed_num_values
        self.embed_dims = embed_dims

        self.nn_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num_value, embedding_dim=dim) for key, dim in zip(embed_num_values, embed_dims)]
        )

        if model_name.startswith('vit'):
            # assert self.feature_aggregation is None # vit uses the CLS token
            if self.feature_aggregation == 'cls_token' or self.feature_aggregation == 'cls_concat' or self.feature_aggregation == 'only_fuse':
                pass
            elif self.feature_aggregation == 'map':
                # Multihead Attention Pooling, following https://arxiv.org/abs/1810.00825
                self.attn_pool = AttentionPoolLatent(
                    in_features=feature_dim,
                    num_heads=num_heads,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
            else:
                raise NotImplementedError(f"Unsupported feature_aggregation: {self.feature_aggregation}")

        if self.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'transformer':
            if position_encording == 'learnable':
                self.position_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim))
            elif position_encording == 'sinusoidal':
                num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                self.position_embedding = torch.zeros(num_features, feature_dim)
                position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(2 * num_features) / feature_dim))
                self.position_embedding[:, 0::2] = torch.sin(position * div_term)
                self.position_embedding[:, 1::2] = torch.cos(position * div_term)
            self.aggregation_transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                num_layers=4)
        elif self.feature_aggregation == 'attention_pool_2d':
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    
    def update_embedding_with_idx(self, index, new_num_embeddings):
        old_embedding = self.nn_embeddings[index]
        embedding_dim = old_embedding.embedding_dim
        new_embedding = nn.Embedding(num_embeddings=new_num_embeddings, embedding_dim=embedding_dim)
        self.nn_embeddings[index] = new_embedding
        self.embed_num_values[index] = new_num_embeddings
        print(f"Updated [{index}] embedding: num_embeddings={new_num_embeddings}, embedding_dim={embedding_dim}")
        return embedding_dim


    def update_embedding(self, key, new_num_embeddings):
        if self.embed_keys is None or len(self.embed_keys) == 0:
            return
        index = self.embed_keys.index(key)
        if index < 0:
            raise ValueError(f"Key {key} not found in embed_keys.")
        embedding_dim = self.update_embedding_with_idx(index, new_num_embeddings)
        print(f"Updated {key} embedding: num_embeddings={new_num_embeddings}, embedding_dim={embedding_dim}")


    def aggregate_feature(self, feature, fused_feature=None):
        if self.model_name.startswith('vit'):
            if self.feature_aggregation == 'cls_token':
                return feature[:, 0, :]
            elif self.feature_aggregation == 'cls_concat':
                return torch.cat([feature[:, 0, :], fused_feature], dim=-1)
            elif self.feature_aggregation == 'only_fuse':
                assert fused_feature is not None, "fused_feature must be provided for 'only_fuse' aggregation"
                return fused_feature
            elif self.feature_aggregation == 'map':
                feature = feature[:, 1:, :]
                if fused_feature is not None:
                    num_tokens = feature.shape[1]
                    fused_feature = fused_feature[:, -num_tokens:, :]
                    feature = torch.cat([feature, fused_feature], dim=2)
                feature = self.attn_pool(feature)
                return feature

        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1])
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1])
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1)
        elif self.feature_aggregation == 'transformer':
            zero_feature = torch.zeros(feature.shape[0], 1, feature.shape[-1], device=feature.device)
            if self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos_embedding = torch.concat([zero_feature, feature], dim=1) + self.position_embedding
            feature_output = self.aggregation_transformer(feature_with_pos_embedding)
            return feature_output[:, 0]
        else:
            assert self.feature_aggregation is None
            return feature
        
    def forward(self, obs_dict, return_feat="null", example=False):

        assert return_feat in ["null", "rgb", "low_dim", "rgb_low_dim"]
        features = list()
        features_vision = list()
        features_low_dim = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # process rgb input
        for key in self.rgb_keys:
            if 'pointcloud' in key:
                # we handle 3D point cloud as fused feature for rgb DINO features. 
                continue
            img = obs_dict[key]
            if img.ndim == 4:
                img = img[:, None]    # (B,C,H,W) -> (B,T,C,H,W)
            # Image.fromarray((255*img[0,0].permute(1,2,0).detach().cpu().numpy()).astype(np.uint8)).save("test.png")
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B*T, *img.shape[2:])
            if self.training:
                img = self.key_transform_map[key](img)
            else:
                img = self.key_eval_transform_map[key](img)

            raw_feature = self.key_model_map[key](img)
            fused_feature = None
            if self.fused_model_name.startswith("dpt"):
                fused_feature = self.key_fused_model_map[key](img, is_training=False, example=example)
            elif self.fused_model_name.startswith('pointnext_'):
                pcd_key = key.replace('rgb', 'pointcloud')
                assert pcd_key in obs_dict, f"Key 'pointcloud' not found in obs_dict for {self.fused_model_name}"
                pointcloud = obs_dict[pcd_key]   # (B, N, D)
                assert pointcloud.ndim == 4, f"Expected pointcloud shape (B, T, N, D), but got {pointcloud.shape}"
                assert pointcloud.shape[0] == B, f"Batch size of pointcloud mismatch: {pointcloud.shape[0]} vs {B}"
                assert pointcloud.shape[1] == T, f"Time dimension of pointcloud mismatch: {pointcloud.shape[1]} vs {T}"
                assert pointcloud.shape[3] == 6, f"Expected pointcloud shape (B, T, N, 6), but got {pointcloud.shape}, 6 ~ [x, y, z, r, g, b]"
                pointcloud = pointcloud.reshape(B*T, pointcloud.shape[2], pointcloud.shape[3])  # (B, T, N, D) -> (B*T, N, D)
                if example is True:
                    fused_feature = torch.zeros(B*T, self.fused_model_feature_dim, device=pointcloud.device)  # (B*T, D)
                else:
                    fused_feature = self.key_fused_model_map[key](pointcloud, pointcloud[:, :, :3])   # xyzrgb-as-feature, xyz-for-aggregation
                fused_feature = fused_feature.reshape(B, T, -1).mean(dim=1)  # (B*T, D) -> (B, T, D) -> (B, D)
            elif self.fused_model_name != '':
                fused_feature = self.key_fused_model_map[key](img)

            feature = self.aggregate_feature(raw_feature, fused_feature)
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            features_vision.append(feature.reshape(B, -1))

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features_low_dim.append(data.reshape(B, -1))
        if len(features_low_dim) > 0:
            features_low_dim = torch.cat(features_low_dim, dim=-1)
            features_low_dim = [self.low_dim_encoder(features_low_dim)]

        # instruction-based task steering
        key = "instruction"
        if key in self.key_model_map.keys() and self.key_model_map[key] is not None:
            text_feature = obs_dict["instruction"]  # it should has been processed as feature in dataset / inference infra. 
            fused_feature = self.key_model_map[key](text_feature)
            features.append(fused_feature)

        # embedding-based task steering
        for i, key in enumerate(self.embed_keys):
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert T ==1, f"For embedding-keys {key}, horizon must be set to 1, but currently {T}"
            features.append(self.nn_embeddings[i](data[:, 0].flatten().long()))

        result = torch.cat(features_vision + features_low_dim + features, dim=-1)

        if return_feat == "null":
            return result
        elif return_feat == 'rgb':
            result_vision = torch.cat(features_vision, dim=-1)
            return result, result_vision
        elif return_feat == 'low_dim':
            result_prop = torch.cat(features_low_dim, dim=-1)
            return result, result_prop
        elif return_feat == 'rgb_low_dim':
            result_vision_prop = torch.cat(features_vision + features_low_dim, dim=-1)
            return result, result_vision_prop
        else:
            raise ValueError(f"Not supported return_feat: {return_feat}. Should from [null, rgb, low_dim, rgb_low_dim]")
    

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        if 'instruction' in self.key_model_map and self.key_model_map['instruction'] is not None:
            example_obs_dict['instruction'] = torch.zeros(
                (1, self.text_input_feature_dim),
                dtype=self.dtype,
                device=self.device
            )
        example_output = self.forward(example_obs_dict, return_feat="null", example=True)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        
        return example_output.shape
    
    @torch.no_grad()
    def output_shape_feat_type(self, return_feat="null"):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        if 'instruction' in self.key_model_map and self.key_model_map['instruction'] is not None:
            example_obs_dict['instruction'] = torch.zeros(
                (1, self.text_input_feature_dim),
                dtype=self.dtype,
                device=self.device
            )
        _, example_vision_output = self.forward(example_obs_dict, return_feat=return_feat, example=True)
        assert len(example_vision_output.shape) == 2
        assert example_vision_output.shape[0] == 1
        return example_vision_output.shape

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

if __name__=='__main__':
    timm_obs_encoder = TimmObsEncoder(
        shape_meta=None,
        model_name='resnet18.a1_in1k',
        pretrained=False,
        global_pool='',
        transforms=None
    )
