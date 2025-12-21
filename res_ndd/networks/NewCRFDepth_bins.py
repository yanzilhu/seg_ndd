import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP
from utils import DN_to_depth

# # --- 辅助函数 ---
# def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
#     return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

# --- 创新模块 1: 几何引导的 Prompt Bin Layer ---
class GeometryGuidedPromptBinLayer(nn.Module):
    def __init__(self, in_channels, n_bins=64, hidden_dim=128):
        super().__init__()
        self.n_bins = n_bins
        # 提示参数 Rpar (Learnable Embedding) [Hidden, 1, 1]
        self.prompt_embedding = nn.Parameter(torch.randn(hidden_dim, 1, 1)) 
        
        # 特征变换层
        self.conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # --- 创新点：法线注入层 ---
        # 将几何分支传来的法线特征融合进 Prompt 交互中
        self.normal_fusion = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 1), # 假设输入法线图是3通道
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
        
        # 预测 Bin 概率的头 (分类)
        self.bin_prob_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, n_bins, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x, normal_feat, min_depth, max_depth):
        """
        x: 解码器特征 [B, C, H, W]
        normal_feat: 几何分支预测的法线图 [B, 3, H, W]
        """
        # 1. 基础特征处理
        feat = self.conv_embed(x) 
        
        # 2. --- 几何引导交互 (Geometry Guidance) ---
        # 让 Prompt 不仅看纹理，还看法线
        norm_feat = self.normal_fusion(normal_feat)
        
        # Prompt 注入 (Broadcasting)
        prompt = self.prompt_embedding 
        
        # 注意力机制：根据法线调整 Prompt 对特征的激活程度
        # 逻辑：如果法线显示是平坦区域，Prompt 应该更关注某种分布
        attn_map = self.attention(torch.cat([feat, norm_feat], dim=1))
        feat_activated = feat * prompt * attn_map + feat # 残差连接
        
        # 3. 预测 Bin 宽度 & 概率
        bin_widths_norm = self.bin_width_head(feat_activated) # [B, N, H, W] (归一化宽度)
        bin_probs = self.bin_prob_head(feat_activated)        # [B, N, H, W]
        
        # 4. 计算 Bin 中心 (Centers)
        # 将归一化宽度转换为实际深度范围
        bin_widths = (max_depth - min_depth) * bin_widths_norm 
        bin_widths = F.pad(bin_widths, (0, 0, 0, 0, 1, 0), mode='constant', value=min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        bin_centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:]) # [B, N, H, W]
        
        # 5. 软回归计算深度 (Soft Geometry Depth)
        depth_bins = torch.sum(bin_centers * bin_probs, dim=1, keepdim=True)
        
        return depth_bins, bin_probs

# --- 创新模块 2: 分布约束的 Offset Head ---
class DistributionGuidedOffsetHead(nn.Module):
    def __init__(self, input_dim=128, n_bins=64):
        super().__init__()
        # 输入不仅是图像特征，还有 Bin 的概率分布向量
        self.conv1 = nn.Conv2d(input_dim + n_bins, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 2, 3, padding=1) # Delta X, Delta Y
        
        nn.init.normal_(self.conv2.weight, mean=0, std=0.001)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x, bin_probs, scale):
        """
        x: 图像特征 [B, C, H, W]
        bin_probs: Prompt 分支输出的概率分布 [B, N_bins, H, W]
        """
        # 拼接特征与分布
        cat_feat = torch.cat([x, bin_probs], dim=1)
        
        x = self.relu(self.bn1(self.conv1(cat_feat)))
        offset = self.conv2(x)
        
        if scale > 1:
             offset = F.interpolate(offset, scale_factor=scale, mode='bilinear', align_corners=True)
        return offset

# --- 创新模块 3: 平面度感知的融合模块 ---
class PlanarityAwareFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 这是一个纯物理/概率计算模块，无参数，或者极少参数
        # 也可以加入一个简单的门控网络来学习
        self.gate_net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), # 输入: Uncertainty_Geo, Uncertainty_Bin, Planarity, 1
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, d_geo, d_bins, u_geo, u_bins, normal_map):
        """
        d_geo: 硬几何深度 (NDDepth)
        d_bins: 软几何深度 (Prompt Bins)
        u_geo, u_bins: 不确定性
        normal_map: 预测的法线图
        """
        # 1. 计算平面度 (Planarity)
        # 利用法线图的局部梯度来判断：法线变化小 -> 平面 -> 信任 d_geo
        # 利用 Sobel 算子或简单差分
        ny, nx = torch.gradient(normal_map, dim=(2, 3))
        planarity = 1.0 - torch.tanh(torch.abs(nx) + torch.abs(ny)).mean(dim=1, keepdim=True) # 1=完全平, 0=边缘
        
        # 2. 基础方差加权
        eps = 1e-6
        w_geo_var = 1.0 / (u_geo + eps)
        w_bins_var = 1.0 / (u_bins + eps)
        
        # 3. 结合平面度进行门控调整
        # 如果是平面(planarity->1)，强行提升 Geo 的权重
        # 如果是复杂区域(planarity->0)，强行提升 Bins 的权重
        
        # 简单融合策略: 
        # W_geo = w_geo_var * (1 + planarity)
        # W_bins = w_bins_var * (1 + (1-planarity))
        
        # 或者使用学习的 Gate
        gate_input = torch.cat([u_geo, u_bins, planarity, d_geo*0+1], dim=1) # placeholder
        alpha = self.gate_net(gate_input) # alpha 倾向于 1 则选 Geo
        
        d_final = alpha * d_geo + (1 - alpha) * d_bins
        
        return d_final, planarity, alpha

def sample_with_offset(map_to_sample, offset):
    """
    根据偏移量对特征图进行采样 (Grid Sample)
    map_to_sample: [B, C, H, W] (例如法线图或距离图)
    offset: [B, 2, H, W] (预测出的偏移量，单位是像素或归一化坐标)
    """
    B, _, H, W = map_to_sample.shape
    device = map_to_sample.device

    # 1. 生成基础网格 (0,0) 到 (W-1, H-1)
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    base_grid = torch.stack([xx, yy], dim=0).float().unsqueeze(0).repeat(B, 1, 1, 1) # [B, 2, H, W]

    # 2. 加上预测的偏移量
    # 假设 offset 输出的是像素级别的位移 (比如移动 10 个像素)
    # 如果 offset 输出范围较大，建议加个 tanh * max_offset 限制
    sample_coords = base_grid + offset 

    # 3. 归一化到 [-1, 1] 用于 grid_sample
    # X 轴归一化: 2 * x / (W-1) - 1
    # Y 轴归一化: 2 * y / (H-1) - 1
    norm_coords = torch.stack([
        2.0 * sample_coords[:, 0] / (W - 1) - 1.0,
        2.0 * sample_coords[:, 1] / (H - 1) - 1.0
    ], dim=3) # [B, H, W, 2]

    # 4. 采样
    # padding_mode='border' 意味着如果指到了图像外面，就用边缘的值
    sampled_map = F.grid_sample(map_to_sample, norm_coords, mode='bilinear', padding_mode='border', align_corners=True)
    
    return sampled_map

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

        

        # 这里你可以复用之前的 NewCRF 或 PSP 模块作为 Shared Decoder
        # 假设我们得到了一个特征图 shared_feat [B, 128, H/4, W/4]
        # 为了代码演示，我定义一个简单的融合层代替复杂的 CRF Decoder
        self.shared_decoder = nn.Conv2d(sum(decoder_channels), 128, 1) # 示意代码
            
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
