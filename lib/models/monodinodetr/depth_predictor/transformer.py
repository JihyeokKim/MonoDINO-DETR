import copy

import torch.nn as nn
import torch.nn.functional as F
import torch


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_key_padding_mask, pos):
        output = src

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #self.self_attn = LinearAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_key_padding_mask, pos):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# class TransformerEfficientEncoderLayer(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
#                  activation="relu", resolution=14, window_resolution=7, kernels=[5, 5, 5, 5]):
#         super().__init__()
#         self.d_model = d_model
#         self.resolution = resolution
#         self.window_resolution = window_resolution

#         # EfficientViTBlock replaces the standard attention and feed-forward layers
#         self.efficientvit_block = EfficientViTBlock(
#             type='s',  # 's' indicates the use of self-attention
#             ed=d_model,  # embedding dimension
#             kd=d_model // nhead,  # key dimension per head
#             nh=nhead,  # number of heads
#             ar=dim_feedforward // (d_model // nhead),  # attention ratio
#             resolution=resolution,
#             window_resolution=window_resolution,
#             kernels=kernels
#         )

#         # Layer normalization for 4D tensors
#         self.norm = nn.BatchNorm2d(d_model)  # Alternatively, use LayerNorm over channels

#     def with_pos_embed(self, tensor, pos):
#         return tensor if pos is None else tensor + pos  # Ensure pos is broadcastable to tensor shape

#     def forward(self, src, src_key_padding_mask=None, pos=None):
#         # src: (B, C, H, W)
#         src = self.with_pos_embed(src, pos)  # Add positional embeddings if provided
#         src2 = self.efficientvit_block(src)
#         src = src + src2  # Residual connection
#         src = self.norm(src)  # Normalize over channels
#         return src
    

# class Conv2d_BN(torch.nn.Sequential):
#     def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
#                  groups=1, bn_weight_init=1, resolution=-10000):
#         super().__init__()
#         self.add_module('c', torch.nn.Conv2d(
#             a, b, ks, stride, pad, dilation, groups, bias=False))
#         self.add_module('bn', torch.nn.BatchNorm2d(b))
#         torch.nn.init.constant_(self.bn.weight, bn_weight_init)
#         torch.nn.init.constant_(self.bn.bias, 0)

#     @torch.no_grad()
#     def fuse(self):
#         c, bn = self._modules.values()
#         w = bn.weight / (bn.running_var + bn.eps)**0.5
#         w = c.weight * w[:, None, None, None]
#         b = bn.bias - bn.running_mean * bn.weight / \
#             (bn.running_var + bn.eps)**0.5
#         m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
#             0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
#         m.weight.data.copy_(w)
#         m.bias.data.copy_(b)
#         return m
    
# class CascadedGroupAttention(torch.nn.Module):
#     r""" Cascaded Group Attention.

#     Args:
#         dim (int): Number of input channels.
#         key_dim (int): The dimension for query and key.
#         num_heads (int): Number of attention heads.
#         attn_ratio (int): Multiplier for the query dim for value dimension.
#         resolution (int): Input resolution, correspond to the window size.
#         kernels (List[int]): The kernel size of the dw conv on query.
#     """
#     def __init__(self, dim, key_dim, num_heads=8,
#                  attn_ratio=4,
#                  resolution=14,
#                  kernels=[5, 5, 5, 5],):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = key_dim ** -0.5
#         self.key_dim = key_dim
#         self.d = int(attn_ratio * key_dim)
#         self.attn_ratio = attn_ratio

#         qkvs = []
#         dws = []
#         for i in range(num_heads):
#             qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
#             dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim, resolution=resolution))
#         self.qkvs = torch.nn.ModuleList(qkvs)
#         self.dws = torch.nn.ModuleList(dws)
#         self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
#             self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

#         points = list(itertools.product(range(resolution), range(resolution)))
#         N = len(points)
#         attention_offsets = {}
#         idxs = []
#         for p1 in points:
#             for p2 in points:
#                 offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
#                 if offset not in attention_offsets:
#                     attention_offsets[offset] = len(attention_offsets)
#                 idxs.append(attention_offsets[offset])
#         self.attention_biases = torch.nn.Parameter(
#             torch.zeros(num_heads, len(attention_offsets)))
#         self.register_buffer('attention_bias_idxs',
#                              torch.LongTensor(idxs).view(N, N))

#     @torch.no_grad()
#     def train(self, mode=True):
#         super().train(mode)
#         if mode and hasattr(self, 'ab'):
#             del self.ab
#         else:
#             self.ab = self.attention_biases[:, self.attention_bias_idxs]

#     def forward(self, x):  # x (B,C,H,W)
#         B, C, H, W = x.shape
#         trainingab = self.attention_biases[:, self.attention_bias_idxs]
#         feats_in = x.chunk(len(self.qkvs), dim=1)
#         feats_out = []
#         feat = feats_in[0]
#         for i, qkv in enumerate(self.qkvs):
#             if i > 0: # add the previous output to the input
#                 feat = feat + feats_in[i]
#             feat = qkv(feat)
#             q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1) # B, C/h, H, W
#             q = self.dws[i](q)
#             q, k, v = q.flatten(2), k.flatten(2), v.flatten(2) # B, C/h, N
#             attn = (
#                 (q.transpose(-2, -1) @ k) * self.scale
#                 +
#                 (trainingab[i] if self.training else self.ab[i])
#             )
#             attn = attn.softmax(dim=-1) # BNN
#             feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W) # BCHW
#             feats_out.append(feat)
#         x = self.proj(torch.cat(feats_out, 1))
#         return x