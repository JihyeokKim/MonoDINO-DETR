import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerEncoder, TransformerEncoderLayer


class DepthAnythingPredictor(nn.Module):

    def __init__(self, model_cfg):
        """
        Initialize depth predictor and depth encoder
        Args:
            model_cfg [EasyDict]: Depth classification network config
        """
        super().__init__()
        depth_num_bins = int(model_cfg["num_depth_bins"])
        depth_min = float(model_cfg["depth_min"])
        depth_max = float(model_cfg["depth_max"])
        self.depth_max = depth_max

        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)
        
        # Create modules
        d_model = model_cfg["hidden_dim"]

        # New module to process weighted_depth
        self.depth_input_proj = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(32, d_model),
            nn.ReLU(),
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model))
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        self.upsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))

        self.depth_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU())

        self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1))

        depth_encoder_layer = TransformerEncoderLayer(
            d_model, nhead=8, dim_feedforward=256, dropout=0.1)

        self.depth_encoder = TransformerEncoder(depth_encoder_layer, 1)

        self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, 256)

    def forward(self, depth_map, mask, pos):
       
        """
        Args:
            weighted_depth: Depth map from depthanythingv2, shape [B, H_full, W_full]
            mask: Binary mask, shape [B, H_small, W_small]
            pos: Positional encodings, shape [B, C, H_small, W_small]
        """
        B, _, H_full, W_full = depth_map.shape          # Shape: [B, 1, H_full, W_full]

        ##############################
        H_small, W_small = mask.shape[-2], mask.shape[-1]

        # # Downsample weighted_depth to match H_small and W_small
        # weighted_depth_small = F.interpolate(
        #     depth_map, size=(H_small, W_small), mode='bilinear', align_corners=False
        # ).squeeze(1)  # Shape: [B, H_small, W_small]

        # # Proceed with the rest of the processing using weighted_depth_small
        # weighted_depth_small = weighted_depth_small.unsqueeze(1)  # Shape: [B, 1, H_small, W_small]
        ##############################

        # Process weighted_depth to match expected input dimensions
        # src = self.depth_input_proj(weighted_depth_small)  # Shape: [B, C, H, W]
        src = self.depth_input_proj(depth_map)  # Shape: [B, C, H, W]



        # src = self.depth_head(src)
        #ipdb.set_trace()
        depth_logits = self.depth_classifier(src)


        depth_probs = F.softmax(depth_logits, dim=1)
        weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)

        weighted_depth_interp = F.interpolate(
            weighted_depth.unsqueeze(1), size=(H_small, W_small), mode='bilinear', align_corners=False
        ).squeeze(1)  # Shape: [B, H_small, W_small]
        #ipdb.set_trace()

        src_interp = F.interpolate(
            src, size=(H_small, W_small), mode='bilinear', align_corners=False
        )  # Shape: [B, C, H_small, W_small]
        
        # depth embeddings with depth positional encodings
        B, C, H, W = src_interp.shape
        src_interp = src_interp.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos = pos.flatten(2).permute(2, 0, 1)

        depth_embed = self.depth_encoder(src_interp, mask, pos)
        depth_embed = depth_embed.permute(1, 2, 0).reshape(B, C, H, W)
        #ipdb.set_trace()
        depth_pos_embed_ip = self.interpolate_depth_embed(weighted_depth_interp)
        depth_embed = depth_embed + depth_pos_embed_ip

        return depth_logits, depth_embed, weighted_depth_interp, depth_pos_embed_ip

    def interpolate_depth_embed(self, depth):
        depth = depth.clamp(min=0, max=self.depth_max)
        pos = self.interpolate_1d(depth, self.depth_pos_embed)
        pos = pos.permute(0, 3, 1, 2)
        return pos

    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings - 1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta
