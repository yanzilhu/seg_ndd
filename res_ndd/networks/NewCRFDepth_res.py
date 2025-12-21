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
# -------------------------------------------------------------------------
# 0. 辅助模块
# -------------------------------------------------------------------------

class PlanarFusionModule(nn.Module):
    def __init__(self, in_channels):
        super(PlanarFusionModule, self).__init__()
        # 输入: [Feat_ND(C) + Feat_Reg(C) + Depth_Geo(1) + Depth_Reg(1) + Uncer_ND(1) + Uncer_Reg(1)]
        fusion_dim = in_channels * 2 + 4 
        
        self.conv1 = nn.Conv2d(fusion_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(64, 2, kernel_size=3, padding=1) # 输出两个权重 map
        self.softmax = nn.Softmax(dim=1) # 保证权重和为1

        # 为两个特征分支分别定义 BN 和 ReLU
        # self.process_nd = nn.Sequential(
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.process_reg = nn.Sequential(
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, depth_geo, depth_reg, uncer_nd, uncer_reg, feat_nd, feat_reg):
        """
        depth_geo, depth_reg: [B, 1, H, W]
        uncer_nd, uncer_reg: [B, 1, H, W]
        feat_nd, feat_reg: [B, 128, H/4, W/4] 
        """
        # 1. 特征对齐 (如果 feature 还是 H/4, W/4，需要上采样)
        feat_nd = upsample(feat_nd, scale_factor=4)
        feat_reg=upsample(feat_reg, scale_factor=4)
        
        # feat_nd = self.process_nd(feat_nd)
        # feat_reg = self.process_reg(feat_reg)

        # 2. 拼接所有信息

        cat_feat = torch.cat([feat_nd, feat_reg, depth_geo, depth_reg, uncer_nd, uncer_reg], dim=1)
        
        # 3. 计算融合权重
        x = self.relu(self.bn1(self.conv1(cat_feat)))
        weights = self.softmax(self.conv2(x)) # [B, 2, H, W]
        
        w_geo = weights[:, 0:1, :, :]
        w_reg = weights[:, 1:2, :, :]
        
        # 4. 加权融合
        depth_final = w_geo * depth_geo + w_reg * depth_reg
        
        return depth_final, w_geo # 返回权重用于可视化分析


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
        
        # --------
        # depth
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        self.crf0 = NewCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)

        self.decoder = PSP(**decoder_cfg)
        self.disp_head1 = DispHead(input_dim=crf_dims[0])
        self.uncer_head1 = UncerHead1(input_dim=crf_dims[0])

        self.up_mode = 'bilinear'
        if self.up_mode == 'mask':
            self.mask_head = nn.Sequential(
                nn.Conv2d(crf_dims[0], 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16*9, 1, padding=0))
        
        # normal and distance
        self.crf7 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf6 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.crf5 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        self.crf4 = NewCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)
        self.decoder_geo = PSP(**decoder_cfg)
        self.decoder2 = PSP(**decoder_cfg)

        self.normal_head1 = NormalHead(input_dim=crf_dims[0])
        self.distance_head1 = DistanceHead(input_dim=crf_dims[0])
        self.uncer_head2 = UncerHead2(input_dim=crf_dims[0])

        self.up_mode = 'bilinear'
        if self.up_mode == 'mask':
            self.mask_head2 = nn.Sequential(
                nn.Conv2d(crf_dims[0], 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16*9, 1, padding=0))
            
        self.min_depth = min_depth
        self.max_depth = max_depth

        # --- 融合模块 ---
        self.fusion_module=PlanarFusionModule(in_channels=128)

        # self.dn2depth = DN_to_depth()
        self.dn_to_depth_layer = DN_to_depth()
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

        if self.with_neck:
            feats = self.neck(feats)
        
        # depth
        ppm_out = self.decoder(feats)

        e3 = self.crf3(feats[3], ppm_out)
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(feats[2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(feats[1], e2)
        e1 = nn.PixelShuffle(2)(e1)
        e0 = self.crf0(feats[0], e1)


        # normal and distance
        ppm_out2 = self.decoder2(feats)

        e7 = self.crf7(feats[3], ppm_out2)
        e7 = nn.PixelShuffle(2)(e7)
        e6 = self.crf6(feats[2], e7)
        e6 = nn.PixelShuffle(2)(e6)
        e5 = self.crf5(feats[1], e6)
        e5 = nn.PixelShuffle(2)(e5)
        e4 = self.crf4(feats[0], e5)
    
        
        # 3. Head 预测
        # 预测法线 [B, 3, H, W] (注意：这里假设 head 内部处理了上采样到原图尺寸，或者在这里做)
        # 通常 head 输出 scale=1 (原图) 或者 scale=4 (特征图)
        # 这里假设 head 输出与输入同尺寸，后续统一上采样到原图
        
        # 法线预测 (-1, 1)
        pred_normal = self.normal_head1(e4, scale=4) 
        pred_normal = F.normalize(pred_normal, dim=1, p=2) # 归一化非常重要
        
        # 距离预测 (0, 1) -> 映射到物理尺度
        pred_distance = self.distance_head1(e4, scale=4)
        pred_distance = pred_distance * self.max_depth # 映射到最大深度范围
        
        # 不确定性预测 (0, 1)
        pred_uncertainty_nd = self.uncer_head2(e4, scale=4)

        # 4. 几何深度合成 (ND -> Depth)
        # b, c, h, w = pred_normal.shape
        # device = pred_normal.device

        # 实例化转换层 (注意：如果尺寸变化，需要动态生成grid)
        # 实际工程中建议把 DN_to_depth 的 grid 生成移到 forward 外部或缓存起来以提速
        
        
        # 计算 D_geo = d / (n^T * K^-1 * p)
        depth_geo = self.dn_to_depth_layer(pred_normal, pred_distance, inv_K)
        depth_geo = depth_geo.clamp(self.min_depth, self.max_depth)

        # return depth_geo, pred_normal, pred_distance, pred_uncertainty_nd, e4
        # 非平面----------------------------------------

        # 3. Head 预测
        # 预测归一化的视差 (0, 1) 或者 深度
        # 如果是 DispHead，通常预测的是 sigmoid 输出
        pred_disp_reg = self.disp_head1(e0, scale=4)
        # 视差转深度: depth = 1 / (disp + epsilon) 或者按照 min/max 映射
        # 这里使用常见的线性映射反转:
        # D = min_depth + (max_depth - min_depth) * disp (如果是深度头)
        # 或者 D = 1 / (min_disp + (max_disp - min_disp) * disp) (如果是视差头)
        # 假设 DispHead 输出的是 0-1 的相对深度:
        depth_reg = pred_disp_reg * self.max_depth
        depth_reg = depth_reg.clamp(self.min_depth, self.max_depth)
        
        # 不确定性预测
        pred_uncertainty_reg = self.uncer_head1(e0, scale=4)


        d_final, w_geo = self.fusion_module(depth_geo/self.max_depth, depth_reg/self.max_depth, pred_uncertainty_nd, pred_uncertainty_reg, e4, e0)

        d_final=d_final*self.max_depth
        
        return d_final,depth_geo,depth_reg,pred_normal,pred_distance,w_geo
        # 4. Pack output
            # outputs = {
            #     'depth_final': d_final,
            #     'depth_geo': d_geo,
            #     'depth_reg': d_reg,
            #     'pred_normal': n_pred,
            #     'pred_distance': dist_pred,
            #     'w_geo': w_geo
            # }
      
                    
class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x

class NormalHead(nn.Module):
    def __init__(self, input_dim=100):
        super(NormalHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 3, 3, padding=1)
       
    def forward(self, x, scale):
        x = self.conv1(x)
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x

class DistanceHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DistanceHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x

class UncerHead1(nn.Module):
    def __init__(self, input_dim=100):
        super(UncerHead1, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x

class UncerHead2(nn.Module):
    def __init__(self, input_dim=100):
        super(UncerHead2, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x

class DHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(DHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_du, act_fn=F.tanh):
        out = self.conv2(self.relu(self.conv1(x_du)))
        return act_fn(out)

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128+192):
        super(SepConvGRU, self).__init__()

        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1))) 
        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class ProjectionInputDepth(nn.Module):
    def __init__(self, cost_dim, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
        self.convd1 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)

        self.convd3 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convd4 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        
        self.convd = nn.Conv2d(64*2+hidden_dim, out_chs - 2, 3, padding=1)
        
    def forward(self, cost, depth1, depth2):

        cor = F.relu(self.convc1(cost))
        cor = F.relu(self.convc2(cor))

        d1 = F.relu(self.convd1(depth1))
        d1 = F.relu(self.convd2(d1))

        d2 = F.relu(self.convd3(depth2))
        d2 = F.relu(self.convd4(d2))

        cor_d = torch.cat([cor, d1, d2], dim=1)
        
        out_d = F.relu(self.convd(cor_d))
                
        return torch.cat([out_d, depth1, depth2], dim=1)

class BasicUpdateBlockDepth(nn.Module):
    def __init__(self, hidden_dim=128, cost_dim=3, context_dim=192):
        super(BasicUpdateBlockDepth, self).__init__()
                
        self.encoder = ProjectionInputDepth(cost_dim=cost_dim, hidden_dim=hidden_dim, out_chs=hidden_dim)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim)
        self.d_head = DHead(hidden_dim, hidden_dim=hidden_dim)
        self.project = nn.Conv2d(256, hidden_dim, 1, padding=0)

    def forward(self, depth1, uncer1, depth2, uncer2, context, gru_hidden, seq_len=3):

        depth1_list = []
        depth1_list.append(depth1)
        depth2_list = []
        depth2_list.append(depth2)

        gru_hidden = torch.tanh(self.project(gru_hidden))
        diff = (depth1.detach() - depth2.detach()).abs()

        for i in range(seq_len):

            input_features = self.encoder(torch.cat([diff, uncer1.detach(), uncer2.detach()], 1), depth1.detach(),  depth2.detach())
            input_c = torch.cat([input_features, context], dim=1)

            gru_hidden = self.gru(gru_hidden, input_c)
            delta_d = self.d_head(gru_hidden)

            delta_d1 = delta_d[:, :1]
            delta_d2 = delta_d[:, 1:]
            
            depth1 = (depth1.detach() + delta_d1).clamp(1e-3, 1)
            depth2 = (depth2.detach() + delta_d2).clamp(1e-3, 1)
         
            depth1_list.append(depth1)
            depth2_list.append(depth2)
            
        return depth1_list, depth2_list

class DispUnpack(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128):
        super(DispUnpack, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 16, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, x, output_size):
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x)) # [b, 16, h/4, w/4]
        x = self.pixel_shuffle(x)

        return x

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
