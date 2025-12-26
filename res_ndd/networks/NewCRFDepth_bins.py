import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP
from utils import DN_to_depth
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# -------------------------------------------------------------------------
# 0. 辅助函数
# -------------------------------------------------------------------------
def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

def sample_with_offset(map_to_sample, offset):
    """根据偏移量采样"""
    B, _, H, W = map_to_sample.shape
    device = map_to_sample.device
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    base_grid = torch.stack([xx, yy], dim=0).float().unsqueeze(0).repeat(B, 1, 1, 1) # [B, 2, H, W]
    
    # 假设 offset 是像素偏移
    sample_coords = base_grid + offset 
    
    # 归一化到 [-1, 1]
    norm_coords = torch.stack([
        2.0 * sample_coords[:, 0] / (W - 1) - 1.0,
        2.0 * sample_coords[:, 1] / (H - 1) - 1.0
    ], dim=3)
    
    sampled_map = F.grid_sample(map_to_sample, norm_coords, mode='bilinear', padding_mode='border', align_corners=True)
    return sampled_map

# -------------------------------------------------------------------------
# 1. 创新模块实现 (替换掉原有的 DispHead, OffsetHead, Fusion)
# -------------------------------------------------------------------------

# 【创新点1：几何引导的 Prompt Bin Layer】 (替代 DispHead)
class GeometryGuidedPromptBinLayer(nn.Module):
    def __init__(self, in_channels, n_bins=64, hidden_dim=128):
        super().__init__()
        self.n_bins = n_bins
        # 提示参数 Rpar
        self.prompt_embedding = nn.Parameter(torch.randn(hidden_dim, 1, 1)) 
        
        self.conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # New! 法线融合层：将几何分支的法线注入 Prompt
        self.normal_fusion = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 1),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.Sigmoid()
        )

        # 预测 Bin 宽度的头
        self.bin_width_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, n_bins, 1),
            nn.Softmax(dim=1) 
        )
        
        # 预测 Bin 概率的头
        self.bin_prob_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, n_bins, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x, normal_feat, min_depth, max_depth):
        # x: [B, C, H, W] (通常是1/4尺度)
        # normal_feat: [B, 3, H, W] (需与 x 同尺度)
        
        feat = self.conv_embed(x) 
        # normal_feat=upsample(normal_feat,scale_factor=4)
        # 交互：融合法线信息
        norm_feat = self.normal_fusion(normal_feat)
        attn_map = self.attention(torch.cat([feat, norm_feat], dim=1))
        # ：让模型自适应决定 “每个像素该用多少图像特征、多少几何法线特征”（几何引导的核心）。
        # Prompt 激活
        prompt = self.prompt_embedding
        feat_activated = feat * prompt * attn_map + feat 
        
        # 预测 Bins
        bin_widths_norm = self.bin_width_head(feat_activated) 
        bin_probs = self.bin_prob_head(feat_activated)       
        
        # 计算 Bin 中心
        bin_widths = (max_depth - min_depth) * bin_widths_norm 
        bin_widths = F.pad(bin_widths, (0, 0, 0, 0, 1, 0), mode='constant', value=min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        bin_centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        
        # 软回归
        depth = torch.sum(bin_centers * bin_probs, dim=1, keepdim=True)
        
        return depth, bin_probs


# 【创新点2：分布引导的 Offset Head】 (替代 OffsetHead)
class DistributionGuidedOffsetHead(nn.Module):
    def __init__(self, input_dim=128, n_bins=64):
        super().__init__()
        # New! 输入包含 Bin 的概率分布
        self.conv1 = nn.Conv2d(input_dim + n_bins, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 2, 3, padding=1) 
        
        nn.init.normal_(self.conv2.weight, mean=0, std=0.001)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x, bin_probs, scale):
        # x: [B, 128, H, W]
        # bin_probs: [B, 64, H, W]
        cat_feat = torch.cat([x, bin_probs], dim=1)
        
        x = self.relu(self.bn1(self.conv1(cat_feat)))
        offset = self.conv2(x)
        
        if scale > 1:
             offset = F.interpolate(offset, scale_factor=scale, mode='bilinear', align_corners=True)
        return offset


# 【创新点3：平面度感知融合】 (替代 PlanarFusionModule)
class PlanarityAwareFusion(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        # 可以保留一部分卷积层用于特征对齐
        self.gate_net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), # 输入: U_geo, U_bin, Planarity, 1
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, d_geo, d_bins, u_geo, u_bins, normal_map):
        # 1. 计算平面度 (利用法线梯度)
        # normal_map: [B, 3, H, W]
        if normal_map.shape[2] != d_geo.shape[2]:
            normal_map = F.interpolate(normal_map, size=d_geo.shape[2:], mode='bilinear')
            
        ny, nx = torch.gradient(normal_map, dim=(2, 3))
        # 梯度越小，平面度越高 (接近1)
        planarity = 1.0 - torch.tanh(torch.abs(nx) + torch.abs(ny)).mean(dim=1, keepdim=True) 
        
        # 2. 门控融合
        # 拼接不确定性和平面度
        gate_input = torch.cat([u_geo, u_bins, planarity, torch.ones_like(planarity)], dim=1)
        alpha = self.gate_net(gate_input) # alpha -> 1 选 Geo
        
        d_final = alpha * d_geo + (1 - alpha) * d_bins
        
        return d_final, alpha

# 保留基础 Head 用于几何分支的基础预测
class NormalHead(nn.Module):
    def __init__(self, input_dim=100):
        super(NormalHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 3, 3, padding=1)
    def forward(self, x, scale=1): # scale default 1
        x = self.conv1(x)
        if scale > 1: x = upsample(x, scale_factor=scale)
        return x

class DistanceHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DistanceHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, scale=1):
        x = self.sigmoid(self.conv1(x))
        if scale > 1: x = upsample(x, scale_factor=scale)
        return x

class UncerHead(nn.Module):
    def __init__(self, input_dim=100):
        super(UncerHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, scale=1):
        x = self.sigmoid(self.conv1(x))
        if scale > 1: x = upsample(x, scale_factor=scale)
        return x
    

class NewCRFDepth(nn.Module):
    """
    Depth network based on neural window FC-CRFs architecture.
    """
    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, mode='single', **kwargs):
        super().__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False
        self.mode = mode

        norm_cfg = dict(type='BN', requires_grad=True)

        window_size = int(version[-2:])

        if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )

        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        self.backbone = SwinTransformer(**backbone_cfg)
        v_dim = decoder_cfg['num_classes']*4
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]
        
        
        self.up_mode = 'bilinear'

        # 2. 共享解码器 (Shared Decoder) - 负责提取通用特征
        # 这里为了简化，我们假设它处理 Swin 的输出并融合到 1/4 分辨率
        embed_dim = 512
        decoder_channels = [128, 256, 512, 1024] # 假设值
        self.decoder_dims = 128 # 最终共享特征的维度


        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        self.crf0 = NewCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)

        self.decoder = PSP(**decoder_cfg)


        # Geometry Branch CRF Layers (e4 path)
        self.crf7 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=7, v_dim=v_dims[3], num_heads=32)
        self.crf6 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=7, v_dim=v_dims[2], num_heads=16)
        self.crf5 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=7, v_dim=v_dims[1], num_heads=8)
        self.crf4 = NewCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=7, v_dim=v_dims[0], num_heads=4)
        self.decoder2 = PSP(**decoder_cfg)

        # --- Heads 替换 (关键修改!) ---

        # 1. Geometry Branch Heads
        self.normal_head1 = NormalHead(input_dim=crf_dims[0])
        self.distance_head1 = DistanceHead(input_dim=crf_dims[0])
        self.uncer_head_geo = UncerHead(input_dim=crf_dims[0])
        
        # New! 分布引导 Offset
        self.offset_head = DistributionGuidedOffsetHead(input_dim=crf_dims[0], n_bins=64)

        # 2. Prompt Branch Heads (原 Regression Branch)
        # New! 几何引导 Prompt Bins
        self.prompt_bin_layer = GeometryGuidedPromptBinLayer(in_channels=crf_dims[0], n_bins=64)
        self.uncer_head_bins = UncerHead(input_dim=crf_dims[0])

        # 3. Fusion
        # New! 平面度感知融合
        self.fusion_module = PlanarityAwareFusion()

        # Tools
        self.dn_to_depth_layer = DN_to_depth() # 动态调整 height=1, width=1
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        if self.mode == 'single':
            self.decoder.init_weights()
        elif self.mode == 'dual':
            self.decoder2.init_weights()
            self.decoder3.init_weights()
        elif self.mode == 'triple':
            self.decoder.init_weights()
            self.decoder2.init_weights()
            # self.decoder3.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def upsample_mask(self, disp, mask):
        """ Upsample disp [H/4, W/4, 1] -> [H, W, 1] using convex combination """
        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(disp, kernel_size=3, padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, 4*H, 4*W)

    def forward(self, imgs, inv_K, epoch):        
        feats = self.backbone(imgs)
        B, C, H, W = imgs.shape

        # --- Decoder Stream 1 (Prompt/Reg Branch) ---
        ppm_out = self.decoder(feats)
        e3 = nn.PixelShuffle(2)(self.crf3(feats[3], ppm_out))
        e2 = nn.PixelShuffle(2)(self.crf2(feats[2], e3))
        e1 = nn.PixelShuffle(2)(self.crf1(feats[1], e2))
        e0 = self.crf0(feats[0], e1) # [B, 128, H/4, W/4]

        # --- Decoder Stream 2 (Geometry Branch) ---
        ppm_out2 = self.decoder2(feats)
        e7 = nn.PixelShuffle(2)(self.crf7(feats[3], ppm_out2))
        e6 = nn.PixelShuffle(2)(self.crf6(feats[2], e7))
        e5 = nn.PixelShuffle(2)(self.crf5(feats[1], e6))
        e4 = self.crf4(feats[0], e5) # [B, 128, H/4, W/4]


        # ==========================================================
        # 核心创新：交互与耦合
        # ==========================================================
        # 1. 几何分支基础预测 (1/4 Scale)
        # 即使这里 scale=1，也是相对于 e4 (H/4) 的 scale=1
        local_normal_quarter = self.normal_head1(e4, scale=1) 
        local_normal_quarter = F.normalize(local_normal_quarter, dim=1, p=2)
        local_dist_quarter = self.distance_head1(e4, scale=1) * self.max_depth
        u_geo = self.uncer_head_geo(e4, scale=1)


        # 2. 几何引导的 Prompt Bin 预测 (Prompt Branch)
        # New! 将 1/4 尺度的法线传给 Prompt Layer
        d_bins_quarter, bin_probs = self.prompt_bin_layer(
            e0, 
            local_normal_quarter.detach(), # 建议 detach 防止几何分支被 Prompt 梯度带偏
            self.min_depth, 
            self.max_depth
        )
        u_bins = self.uncer_head_bins(e0, scale=1)

        # 3. 分布引导的 Offset 预测 (Geometry Branch)
        # New! 将 1/4 尺度的 Bin Probs 传给 Offset Head
        offset = self.offset_head(e4, bin_probs.detach(), scale=1) # [B, 2, H/4, W/4]

        # 4. 几何分支：偏移校正
        seed_normal_quarter = sample_with_offset(local_normal_quarter, offset)
        seed_dist_quarter = sample_with_offset(local_dist_quarter, offset)
        seed_normal_quarter = F.normalize(seed_normal_quarter, dim=1, p=2)

        # 计算几何深度 (在 1/4 尺度计算，节省显存)
        # 需要缩放内参
        inv_K_scaled = inv_K.clone()
        inv_K_scaled[:, 0, 0] *= 4; inv_K_scaled[:, 1, 1] *= 4
        inv_K_scaled[:, 0, 2] *= 4; inv_K_scaled[:, 1, 2] *= 4
        
        self.dn_to_depth_layer.width = seed_normal_quarter.shape[3]
        self.dn_to_depth_layer.height = seed_normal_quarter.shape[2]
        self.dn_to_depth_layer.batch_size = B
        
        d_geo_quarter = self.dn_to_depth_layer(seed_normal_quarter, seed_dist_quarter, inv_K_scaled)
        d_geo_quarter = d_geo_quarter.clamp(self.min_depth, self.max_depth)

        # 5. 平面度感知融合
        d_final_quarter, alpha_map = self.fusion_module(
            d_geo_quarter, d_bins_quarter, u_geo, u_bins, seed_normal_quarter
        )

        # ==========================================================
        # 上采样与输出
        # ==========================================================
        d_final = upsample(d_final_quarter, 4)
        d_geo = upsample(d_geo_quarter, 4)
        d_bins = upsample(d_bins_quarter, 4)
        pred_normal = upsample(seed_normal_quarter, 4)
        pred_dist = upsample(seed_dist_quarter, 4)
        offset_up = upsample(offset, 4)

        # 返回值对应你的训练 Loop
        # 注意：d_reg 现在变成了 d_bins
        return d_final, d_geo, d_bins, pred_normal, pred_dist, alpha_map, u_geo, u_bins, offset_up
       
      
                    
# class DispHead(nn.Module):
#     def __init__(self, input_dim=100):
#         super(DispHead, self).__init__()
#         self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, scale):
#         x = self.sigmoid(self.conv1(x))
#         if scale > 1:
#             x = upsample(x, scale_factor=scale)
#         return x

# class NormalHead(nn.Module):
#     def __init__(self, input_dim=100):
#         super(NormalHead, self).__init__()
#         self.conv1 = nn.Conv2d(input_dim, 3, 3, padding=1)
       
#     def forward(self, x, scale):
#         x = self.conv1(x)
#         if scale > 1:
#             x = upsample(x, scale_factor=scale)
#         return x

# class DistanceHead(nn.Module):
#     def __init__(self, input_dim=100):
#         super(DistanceHead, self).__init__()
#         self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, scale):
#         x = self.sigmoid(self.conv1(x))
#         if scale > 1:
#             x = upsample(x, scale_factor=scale)
#         return x

# class UncerHead1(nn.Module):
#     def __init__(self, input_dim=100):
#         super(UncerHead1, self).__init__()
#         self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, scale):
#         x = self.sigmoid(self.conv1(x))
#         if scale > 1:
#             x = upsample(x, scale_factor=scale)
#         return x

# class UncerHead2(nn.Module):
#     def __init__(self, input_dim=100):
#         super(UncerHead2, self).__init__()
#         self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, scale):
#         x = self.sigmoid(self.conv1(x))
#         if scale > 1:
#             x = upsample(x, scale_factor=scale)
#         return x

# class DHead(nn.Module):
#     def __init__(self, input_dim=128, hidden_dim=128):
#         super(DHead, self).__init__()
#         self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
#         self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x_du, act_fn=F.tanh):
#         out = self.conv2(self.relu(self.conv1(x_du)))
#         return act_fn(out)


# def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
#     """Upsample input tensor by a factor of 2
#     """
#     return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
