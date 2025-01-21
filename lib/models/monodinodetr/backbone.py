# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from .dla import dla34, dla60
from .depth_anything_v2.dpt import DepthAnythingV2

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, images):
        xs = self.body(images)
        out = {}
        for name, x in xs.items():
            m = torch.zeros(x.shape[0], x.shape[2], x.shape[3]).to(torch.bool).to(x.device)
            out[name] = NestedTensor(x, m)
        return out

class BackboneBaseDLA(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer3' not in name and 'layer4' not in name and 'layer5' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"level3": "0", "level4": "1", "level5": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [128, 256, 512]             # DLA34
            # self.num_channels = [256, 512, 1024]          # DLA60

        else:
            return_layers = {'level5': "0"}
            self.strides = [32]
            self.num_channels = [512]                       # DLA34
            # self.num_channels = [1024]                    # DLA60

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, images):
        xs = self.body(images)
        out = {}
        for name, x in xs.items():
            m = torch.zeros(x.shape[0], x.shape[2], x.shape[3]).to(torch.bool).to(x.device)
            out[name] = NestedTensor(x, m)
        return out
class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

class BackboneDLA(BackboneBaseDLA):
    """DLA backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        if name == 'dla34':
            backbone = dla34(pretrained=is_main_process(), return_levels=True)
            print("DLA34 loaded")
        elif name == 'dla60':
            backbone = dla60(pretrained=is_main_process(), return_levels=True)
            print("DLA60 loaded")
        else:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=norm_layer)
            assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

class BackboneViT(nn.Module):
    
    """DINOv2 backbone"""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        super().__init__()
        # norm_layer = FrozenBatchNorm2d
        depthanything_model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        # Initialize DepthAnythingV2 with DINOv2 backbone
        if name in depthanything_model_configs:
            depth_anything = DepthAnythingV2(**depthanything_model_configs[name])
            model_weights_path = f'/home/usrg/MonoDINO-DETR/checkpoints/depth_anything_v2_{name}.pth'  # Update with your path

            # Load pre-trained weights
            depth_anything.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
            # depth_anything.to('cuda').eval()

            # Use the pretrained DINOv2 model as the backbone
            self.backbone = depth_anything.pretrained
            for name_1, param in self.backbone.named_parameters():
                # if not 'norm.' in name_1 and 'blocks.8' not in name_1 and 'blocks.11' not in name_1 and not 'cls_token' in name_1 and not 'pos_embed' in name_1 and not 'mask_token' in name_1:
                param.requires_grad_(False)
            self.intermediate_layer_idx = depth_anything.intermediate_layer_idx[name]

            self.dpt_head = depth_anything.depth_head
            for name_2, param in self.dpt_head.named_parameters():
                param.requires_grad_(False)
            print(f"DepthAnythingV2 DINOv2 model ({name})  with DPT head loaded.")
        else:
            raise ValueError(f"Unsupported model type: {name}")
        
        self.return_interm_layers = return_interm_layers
        self.train_backbone = train_backbone
        
         # Set the channels and strides for the transformer layers
        self.num_channels = depthanything_model_configs[name]['out_channels']
        self.embed_dim = self.backbone.embed_dim
        # Since ViT doesn't have strides like CNNs, we can set dummy stride values or compute them based on the patch size
        self.strides = [14, 14]  # Assuming patch size of 14x14

        self.patch_size = (14, 14)

    def forward(self, images, masks=None):
        """
        Forward pass through the DINOv2 backbone.
        """
        
        # Padding
        B, C, H, W = images.shape
        if W % self.patch_size[1] != 0:
            images = F.pad(images, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            images = F.pad(images, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        
        Hh, Ww = images.shape[-2], images.shape[-1]
        patch_h, patch_w = Hh // self.patch_size[0], Ww // self.patch_size[1]

        # interpolate images to target size
        target_size = 518
        images = F.interpolate(images, size=(target_size, target_size), mode='bilinear', align_corners=False)
        patch_h, patch_w = target_size // 14, target_size // 14

        features = self.backbone.get_intermediate_layers(
            images, masks, self.intermediate_layer_idx, return_class_token=True
        )
        del images

        # Pass features through DPT head
        depth = self.dpt_head(features, patch_h, patch_w)
        depth = F.relu(depth)

        # Prepare the output dictionary
        out = {}
        out['depth'] = depth
        
        feature_maps = {}

        # Original
        for i, (feat, idx) in enumerate(zip(features[-3:], self.intermediate_layer_idx[-3:])):
            # feat = features[2]
            x = feat[0]  # Patch tokens: shape [B, num_patches, embed_dim]
            
            # Reshape patch tokens to 2D feature maps
            x = x.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w).contiguous()  # Shape: [B, embed_dim, H', W']

            # Create a binary mask (all zeros since we have no padding)
            mask = torch.zeros(B, patch_h*(2**(2-i)), patch_w*(2**(2-i)), dtype=torch.bool, device=x.device)  # [B, H', W']
            
            # Store the feature map in the output dictionary
            feature_maps[str(i)] = NestedTensor(x, mask)

        out['feature_maps'] = feature_maps
        del features
        return out

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, images):
        xs = self[0](images)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

class Joiner2(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
        self.embed_dim = backbone.embed_dim

    def forward(self, images):
        # Get outputs from the backbone
        outputs = self[0](images)

        # Extract features and depth from the outputs
        features = outputs.get('feature_maps', {})
        depth = outputs.get('depth', None)

        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(features.items()):
            out.append(x)
            # Apply positional encoding to each feature map
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        # Return features, positional encodings, and depth output
        return out, pos, depth

def build_backbone(cfg):
    
    position_embedding = build_position_encoding(cfg)
    return_interm_layers = cfg['masks'] or cfg['num_feature_levels'] > 1
    if cfg['backbone'].startswith('dla'):
        backbone = BackboneDLA(cfg['backbone'], cfg['train_backbone'], return_interm_layers, cfg['dilation'])
        print("DLA model is selected")

        model = Joiner(backbone, position_embedding)
    elif cfg['backbone'].startswith('resnet'):
        backbone = Backbone(cfg['backbone'], cfg['train_backbone'], return_interm_layers, cfg['dilation'])
        print("ResNet is selected")

        model = Joiner(backbone, position_embedding)
    elif cfg['backbone'].startswith('vit'):
        backbone = BackboneViT(cfg['backbone'], cfg['train_backbone'], return_interm_layers, cfg['dilation'])
        print("ViT is selected")
        model = Joiner2(backbone, position_embedding)
    return model
