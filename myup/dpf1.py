训练代码:
while epoch < args.num_epochs:

        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)
        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()

            loss = 0

            loss_depth1 = 0

            loss_depth2 = 0
            before_op_time = time.time()

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))

            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))
            normal_gt = torch.autograd.Variable(sample_batched['normal'].cuda(args.gpu, non_blocking=True))

            inv_K = torch.autograd.Variable(sample_batched['inv_K'].cuda(args.gpu, non_blocking=True))

            inv_K_p = torch.autograd.Variable(sample_batched['inv_K_p'].cuda(args.gpu, non_blocking=True))

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                depth1, u1, depth2, u2, n1_norm, distance,final_depth,R,P_full,z_plane,_=model(image, inv_K, epoch)

        
                    mask = depth_gt > 0.1

            
                if epoch<5:

                    loss_depth1 = silog_criterion.forward(depth1, depth_gt, mask)

                    loss_depth2 = silog_criterion.forward(depth2, depth_gt, mask)

                    uncer1_gt = torch.exp(-5 * torch.abs(depth_gt - depth1.detach()) / (depth_gt + depth1.detach() + 1e-7))

                    uncer2_gt = torch.exp(-5 * torch.abs(depth_gt - depth2.detach()) / (depth_gt + depth2.detach() + 1e-7))

                else :

                    loss_depth1 = silog_criterion.forward(depth1[0], depth_gt, mask)

                    loss_depth2 = silog_criterion.forward(depth2[0], depth_gt, mask)

                    uncer1_gt = torch.exp(-5 * torch.abs(depth_gt - depth1[0].detach()) / (depth_gt + depth1[0].detach() + 1e-7))

                    uncer2_gt = torch.exp(-5 * torch.abs(depth_gt - depth2[0].detach()) / (depth_gt + depth2[0].detach() + 1e-7))


                loss_uncer1 = torch.abs(u1[mask.to(torch.bool)]-uncer1_gt[mask.to(torch.bool)]).mean()

                loss_uncer2 = torch.abs(u2[mask.to(torch.bool)]-uncer2_gt[mask.to(torch.bool)]).mean()
                loss_normal = 5 * ((1 - (normal_gt * n1_norm).sum(1, keepdim=True)) * mask.float()).sum() / (mask.float() + 1e-7).sum()

            
                normal_gt = torch.stack([normal_gt[:, 0], normal_gt[:, 2], normal_gt[:, 1]], 1)
                normal_gt_norm = F.normalize(normal_gt, dim=1, p=2)
                distance_gt = dn_to_distance(depth_gt, normal_gt_norm, inv_K_p)
                loss_distance = 0.25 * torch.abs(distance_gt[mask] - distance[mask]).mean()

                mask_float = mask.float() # 显式转换
                geom = geometry_losses(
                n_unit=n1_norm,           # (B,3,H,W)

                distance=distance,            # (B,1,H,W)

                R=R, P=P_full,                         # (B,3,3), a(B,3,H,W)

                z_pred=final_depth,               # (B,1,H,W)

                z_plane=z_plane,                  # (B,1,H,W)

                mask=mask_float,

                w_align=1.0, w_plane_smooth=1.0, w_recon=1.0

            )   

                loss_geom = geom['L_geom_total']

                if epoch<5:

                    loss = (  loss_depth1 + loss_depth2) +  loss_uncer1 + loss_uncer2 + loss_normal+loss_distance

                else:

                    loss = (  loss_depth1 + loss_depth2) + \

                    loss_uncer1 + loss_uncer2 + \

                    1.0*loss_geom+loss_normal+loss_distance



            # loss.backward()

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2

其中class silog_loss(nn.Module):

    def __init__(self, variance_focus):

        super(silog_loss, self).__init__()

        self.variance_focus = variance_focus



    def forward(self, depth_est, depth_gt, mask):

        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])

        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0



def geometry_losses(

    n_unit,                # (B,3,H,W)

    distance,              # (B,1,H,W)

    R, P,                  # (B,3,3), (B,K,H,W)

    z_pred, z_plane,       # (B,1,H,W), (B,1,H,W)

    mask=None,

    w_align=1.0, w_plane_smooth=1.0, w_recon=1.0

):

    L_align = manhattan_alignment_loss(n_unit, R, P, mask)

    L_plane  = plane_distance_smooth_var_loss(distance, P, mask)

    L_recon  = plane_reconstruction_loss(z_pred, z_plane, mask)

    L_total  = w_align * L_align + w_plane_smooth * L_plane + w_recon * L_recon

    return {

        'L_align': L_align,

        'L_plane': L_plane,

        'L_recon': L_recon,

        'L_geom_total': L_total

    }

def manhattan_alignment_loss(n_unit, R, P, mask=None, eps=1e-6):

    """

    n_unit: (B,3,H,W) 已归一化法线（如 n1_norm）

    R:      (B,3,3)   每图的Manhattan基（列为轴向）

    P:      (B,3,H,W) 平面门控概率（例如[墙,地,顶]）

    mask:   (B,1,H,W) 有效像素掩码（可为None）

    目标：让法线在各自门控下贴近对应主轴（软分配）

    L = sum_{i=1..3} mean( P_i * (1 - |n · axis_i|) )

    """

    B, _, H, W = n_unit.shape

    axes = [R[:, :, 0], R[:, :, 1], R[:, :, 2]]  # 每个轴 (B,3)



    loss = 0.0

    denom = 0.0

    for i in range(3):

        ai = axes[i].view(B, 3, 1, 1).expand(-1, -1, H, W)  # (B,3,H,W)

        cosi = torch.sum(n_unit * ai, dim=1, keepdim=True).abs()  # (B,1,H,W)

        li = (1.0 - cosi).clamp(min=0.0) * P[:, i:i+1, :, :]      # (B,1,H,W)

        if mask is not None:

            li = li * mask

            denom += mask.sum() + 1e-6

        else:

            denom += li.numel()

        loss += li.sum()

    return loss / (denom + eps)





def plane_distance_smooth_var_loss(distance, P, mask=None, eps=1e-6):

    """

    distance: (B,1,H,W) 到原点距离（你模型的 distance_head 输出 * max_depth）

    P:        (B,K,H,W) 平面门控（如K=3）

    由两部分构成：

      - TV平滑：sum_k mean( P_k * (|dx d| + |dy d|) )

      - 区域方差：sum_k Var_{P_k}(distance)

    """

    B, _, H, W = distance.shape

    K = P.shape[1]



    dx = torch.abs(distance[:, :, :, 1:] - distance[:, :, :, :-1])

    dy = torch.abs(distance[:, :, 1:, :] - distance[:, :, :-1, :])



    tv = 0.0

    tv_denom = 0.0

    for k in range(K):

        Pk = P[:, k:k+1, :, :]

        if mask is not None:

            Pk = Pk * mask



        tvx = (Pk[:, :, :, 1:] * dx).sum()

        tvy = (Pk[:, :, 1:, :] * dy).sum()

        tv += tvx + tvy

        tv_denom += (Pk[:, :, :, 1:]).sum() + (Pk[:, :, 1:, :]).sum() + eps



    # 区域方差：对每个k，μ_k = sum(P_k * d)/sum(P_k)

    var_loss = 0.0

    for k in range(K):

        Pk = P[:, k:k+1, :, :]

        if mask is not None:

            Pk = Pk * mask

        wsum = Pk.sum() + eps

        mu = (Pk * distance).sum() / wsum

        var = ((Pk * (distance - mu)) ** 2).sum() / wsum

        var_loss += var



    tv_term = tv / (tv_denom + eps)

    return tv_term + var_loss





def plane_reconstruction_loss(z_pred, z_plane, mask=None, delta=0.1, eps=1e-6):

    """

    z_pred:  (B,1,H,W) 网络预测深度（可取两路的平均/最终细化输出）

    z_plane: (B,1,H,W) DPF层拼回的平面深度

    Huber(SmoothL1) 损失

    """

    diff = z_pred - z_plane

    if mask is not None:

        diff = diff * mask

        denom = mask.sum() + eps

    else:

        denom = diff.numel()

    abs_diff = diff.abs()

    huber = torch.where(abs_diff < delta, 0.5 * (abs_diff ** 2) / delta, abs_diff - 0.5 * delta)

    return huber.sum() / denom


模型架构文件为：

from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP
from utils import DN_to_depth

import torch
import torch.nn as nn
import torch.nn.functional as F
# 曼哈顿
class ManhattanBasisHead(nn.Module):
    """
    输入：e0（B, C, H, W）等高层特征
    输出：R（B, 3, 3），表示每张图的Manhattan基（正交化，det>0）
    """
    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(hidden_dim, 9, kernel_size=1, padding=0)  # 直接预测9个数

    @staticmethod
    def _gram_schmidt_orthonormalize(B):  # B: (B, 3, 3)
        # 列向量正交化
        u1 = B[:, :, 0]
        e1 = F.normalize(u1, dim=1)

        u2 = B[:, :, 1] - (torch.sum(B[:, :, 1] * e1, dim=1, keepdim=True) * e1)
        e2 = F.normalize(u2 + 1e-8, dim=1)

        u3 = B[:, :, 2] \
             - (torch.sum(B[:, :, 2] * e1, dim=1, keepdim=True) * e1) \
             - (torch.sum(B[:, :, 2] * e2, dim=1, keepdim=True) * e2)
        e3 = F.normalize(u3 + 1e-8, dim=1)

        R = torch.stack([e1, e2, e3], dim=2)  # (B,3,3)
        # 保证det>0：若det<0则翻转第三列
        det = torch.det(R)
        flip = (det < 0).float().view(-1, 1)   # (B, 1)
        e3_flipped = e3 * (1.0 - 2.0 * flip)
        R = torch.stack([e1, e2, e3_flipped], dim=2)
        return R

    def forward(self, x):
        # x: (B,C,H,W)
        h = self.act(self.conv(x))
        b, c, hH, hW = h.shape
        # GAP到 (B, hidden, 1, 1) -> (B, 9)
        g = F.adaptive_avg_pool2d(h, 1)
        nine = self.proj(g).view(b, 9)
        Bmat = nine.view(b, 3, 3)
        R = self._gram_schmidt_orthonormalize(Bmat)
        return R  # (B,3,3)
# 平面门控
class PlaneGateHead(nn.Module):
    """
    输入：e0（B, C, H, W）
    输出：P（B, K, H, W），例如K=3分别表示{墙, 地, 顶}的软门控概率（softmax）
    """
    def __init__(self, in_channels, num_planes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(in_channels, num_planes, 1)

    def forward(self, x):
        h = self.act(self.conv1(x))
        logits = self.proj(h)             # (B,K,H,W)
        P = torch.softmax(logits, dim=1)  # 通道维softmax
        return P

# DPF可微笑（可微局部平面拟合 + 平面深度重建）
# 给定深度图 z、相机内参逆 inv_K 与K个区域权重 W（软掩码），先把 z 回投为点云，
# 再对每个区域做加权SVD/特征分解拟合平面 (n_k, d_k)，最后把各平面拼回像素层的平面深度 z_plane（用区域权重做软选择，保持可微）。
class DPFLayer(nn.Module):
    def __init__(self, tau=20.0, eps=1e-6, min_region_area=20):
        super().__init__()
        self.tau = tau
        self.eps = eps
        self.min_region_area = min_region_area

    @staticmethod
    def _make_rays(inv_K, H, W, device, dtype=torch.float32):
        # pixel homogeneous coords (3,H,W)
        ys, xs = torch.meshgrid(
            torch.arange(0, H, device=device),
            torch.arange(0, W, device=device),
            indexing='ij'
        )
        ones = torch.ones_like(xs, dtype=dtype, device=device)
        pix = torch.stack([xs.to(dtype=dtype), ys.to(dtype=dtype), ones], dim=0)  # (3,H,W)
        pix = pix.unsqueeze(0).expand(inv_K.shape[0], -1, -1, -1)  # (B,3,H,W)

        # inv_K: (B,3,3) ; rays = inv_K @ pix
        # use einsum with correct subscripts
        rays = torch.einsum('bij,bjhw->bihw', inv_K.to(dtype=dtype), pix)  # (B,3,H,W)
        rays = F.normalize(rays, dim=1)
        return rays

    def forward(self, depth_map, inv_K, region_w):
        """
        depth_map: (B,1,H,W) - may be float16 from autocast; we'll compute in float32
        inv_K:     (B,3,3)
        region_w:  (B,K,H,W) soft-weights
        """
        with torch.cuda.amp.autocast(enabled=False):
            B, _, H, W = depth_map.shape
            K = region_w.shape[1]
            device = depth_map.device

            # compute rays and X in float32 for numerical stability
            inv_K = inv_K[:, :3, :3]
            invK_f32 = inv_K.to(device=device).to(torch.float32)
            depth_f32 = depth_map.to(torch.float32)
            rays = self._make_rays(invK_f32, H, W, device, dtype=torch.float32)  # (B,3,H,W)
            X = rays * depth_f32  # (B,3,H,W)

            # normalize region weights per region
            w = region_w.clamp(min=0.0).to(torch.float32) + self.eps  # (B,K,H,W)
            w_sum = torch.sum(w, dim=(2, 3), keepdim=True) + self.eps     # (B,K,1,1)
            w_norm = w / w_sum

            # optionally zero-out regions that are too small (avoid degenerate cov)
            area = torch.sum((region_w > 0.01).to(torch.float32), dim=(2,3))  # (B,K)
            small_region_mask = (area < float(self.min_region_area)).unsqueeze(-1).unsqueeze(-1)  # (B,K,1,1)
            w_norm = torch.where(small_region_mask, torch.zeros_like(w_norm), w_norm)

            # weighted centroid mu_k
            X_flat = X.view(B, 3, -1)                         # (B,3,N)
            w_flat = w_norm.view(B, K, -1)                    # (B,K,N)
            mu = torch.einsum('bkn,bcn->bkc', w_flat, X_flat)  # (B,K,3)

            # weighted covariance
            X_exp = X_flat.unsqueeze(1).expand(B, K, 3, H*W)   # (B,K,3,N)
            mu_exp = mu.unsqueeze(-1)                         # (B,K,3,1)
            Xm = X_exp - mu_exp                               # (B,K,3,N)
            wN = w_flat.unsqueeze(2)                          # (B,K,1,N)
            Xm_w = Xm * wN                                    # (B,K,3,N)
            cov = torch.einsum('bkcn,bkdn->bkcd', Xm_w, Xm)   # (B,K,3,3)
            cov = 0.5 * (cov + cov.transpose(-1, -2))         # symmetrize

            cov = cov + torch.eye(3).to(cov.device) * 1e-6
            # eigen-decomp (float32). torch.linalg.eigh is differentiable.
            eigvals, eigvecs = torch.linalg.eigh(cov)         # eigvals:(B,K,3), eigvecs:(B,K,3,3)
            n = eigvecs[..., 0]                               # smallest-eigvec => normal (B,K,3)
            n = F.normalize(n, dim=-1)

            # canonicalize direction so that n_z >= 0 (camera forward z)
            sign = torch.where(n[..., 2:3] < 0, -1.0, 1.0)
            n = n * sign

            # d = - n^T mu
            d = -torch.sum(n * mu, dim=-1, keepdim=True)     # (B,K,1)

            # denom = n · rays  -> (B,K,H,W)
            denom = torch.einsum('bkc,bchw->bkhw', n, rays)  # (B,K,H,W)
            denom = denom.clamp(min=1e-3)  # avoid division-by-zero / huge values

            z_k_map = -d.unsqueeze(-1) / denom  # (B,K,H,W) broadcast -> ensure shapes OK
            # numerical guard: remove non-finite
            z_k_map = torch.where(torch.isfinite(z_k_map), z_k_map, torch.zeros_like(z_k_map))

            # soft selection via region_w with temperature
            gate = torch.softmax(self.tau * w, dim=1)        # (B,K,H,W)
            z_plane = torch.sum(gate * z_k_map, dim=1, keepdim=True)  # (B,1,H,W)

        # return as float32 (the caller can clamp/cast to original dtype)
        return n, d, z_plane, z_k_map

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

        self.up_mode = 'bilinear'
        if self.up_mode == 'mask':
            self.mask_head2 = nn.Sequential(
                nn.Conv2d(crf_dims[0], 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16*9, 1, padding=0))
            
        self.update = BasicUpdateBlockDepth()
        
        self.manhattan_head = ManhattanBasisHead(in_channels=5)
        self.plane_gate     = PlaneGateHead(in_channels=5, num_planes=3)  # [墙, 地, 顶]
        self.dpf_layer      = DPFLayer(tau=20.0)

                
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
        ppm_out = self.decoder(feats)  #B,512,15,20

        e3 = self.crf3(feats[3], ppm_out)
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(feats[2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(feats[1], e2)
        e1 = nn.PixelShuffle(2)(e1)
        e0 = self.crf0(feats[0], e1)
       
        if self.up_mode == 'mask':
            mask = self.mask_head(e0)
            d1 = self.disp_head1(e0, 1)
            u1 = self.uncer_head1(e0, 1)
            d1 = self.upsample_mask(d1, mask)
            u1 = self.upsample_mask(u1, mask)
        else:
            d1 = self.disp_head1(e0, 1)
            u1 = self.uncer_head1(e0, 1)

        # normal and distance
        ppm_out2 = self.decoder2(feats)

        e7 = self.crf7(feats[3], ppm_out2)
        e7 = nn.PixelShuffle(2)(e7)
        e6 = self.crf6(feats[2], e7)
        e6 = nn.PixelShuffle(2)(e6)
        e5 = self.crf5(feats[1], e6)
        e5 = nn.PixelShuffle(2)(e5)
        e4 = self.crf4(feats[0], e5)

        if self.up_mode == 'mask':
            mask2 = self.mask_head2(e4)
            n1 = self.normal_head1(e4, 1)
            dist1 = self.distance_head1(e4, 1)
            u2 = self.uncer_head2(e4, 1)
            n1 = self.upsample_mask(n1, mask2)
            dist1 = self.upsample_mask(dist1, mask2)
            u2 = self.upsample_mask(u2, mask2)
        else:
            n1 = self.normal_head1(e4, 1)
            dist1 = self.distance_head1(e4, 1)
            u2 = self.uncer_head2(e4, 1)

        b, c, h, w =  n1.shape 
        device = n1.device  
        dn_to_depth = DN_to_depth(b, h, w).to(device)

        distance = dist1 * self.max_depth 
        n1_norm = F.normalize(n1, dim=1, p=2)
        depth2 = dn_to_depth(n1_norm, distance, inv_K).clamp(0, self.max_depth)

        if epoch < 5:
            depth1 = upsample(d1, scale_factor=4) * self.max_depth
            u1 = upsample(u1, scale_factor=4)
            depth2 = upsample(depth2, scale_factor=4)
            u2 = upsample(u2, scale_factor=4)
            n1_norm = upsample(n1_norm, scale_factor=4)
            distance = upsample(distance, scale_factor=4)

            eps = 1e-6
            # u1/u2 期望和 uncer_gt 同分布 (较大 -> 更可信)
            w1 = u1.clamp(min=eps)
            w2 = u2.clamp(min=eps)
            final_depth = (w1 * depth1 + w2 * depth2) / (w1 + w2 + eps)

            geo_in = torch.cat([n1_norm, distance, final_depth], dim=1)
            # ---- 计算 R / P / DPF 拼回平面深度 ----
            R = self.manhattan_head(geo_in)                       # (B,3,3)
            P = self.plane_gate(geo_in)                           # (B,3,h,w) (与 geo_in 同分辨率)
            # 若 geo_in 分辨率 < depth 分辨率，可在此上采样，这里二者一致无需上采样
            P_full = P

            # n_k, d_k, z_plane, z_k_map = self.dpf_layer(final_depth, inv_K, P_full)
            _, _, z_plane, z_k_map = self.dpf_layer(final_depth, inv_K, P_full) 

            return depth1, u1, depth2, u2, n1_norm, distance,final_depth,R,P_full,z_plane,z_k_map
        
        else:
            depth1 = d1
            depth2 = depth2 / self.max_depth
            context = feats[0]
            gru_hidden = torch.cat((e0, e4), 1)
            depth1_list, depth2_list  = self.update(depth1, u1, depth2, u2, context, gru_hidden)

            for i in range(len(depth1_list)):
                depth1_list[i] = upsample(depth1_list[i], scale_factor=4) * self.max_depth
            u1 = upsample(u1, scale_factor=4)
            for i in range(len(depth2_list)):
                depth2_list[i] = upsample(depth2_list[i], scale_factor=4) * self.max_depth 
            u2 = upsample(u2, scale_factor=4)
            n1_norm = upsample(n1_norm, scale_factor=4)
            distance = upsample(distance, scale_factor=4)

            eps = 1e-6
            # u1/u2 期望和 uncer_gt 同分布 (较大 -> 更可信)
            w1 = u1.clamp(min=eps)
            w2 = u2.clamp(min=eps)
            final_depth = (w1 * depth1_list[-1] + w2 * depth2_list[-1]) / (w1 + w2 + eps)

            geo_in = torch.cat([n1_norm, distance, final_depth], dim=1)
            # ---- 计算 R / P / DPF 拼回平面深度 ----
            R = self.manhattan_head(geo_in)                       # (B,3,3)
            P = self.plane_gate(geo_in)                           # (B,3,h,w) (与 geo_in 同分辨率)
            # 若 geo_in 分辨率 < depth 分辨率，可在此上采样，这里二者一致无需上采样
            P_full = P

            # n_k, d_k, z_plane, z_k_map = self.dpf_layer(final_depth, inv_K, P_full) 
            _, _, z_plane, z_k_map = self.dpf_layer(final_depth, inv_K, P_full) 


            return depth1_list, u1, depth2_list, u2, n1_norm, distance, final_depth,R,P_full,z_plane,z_k_map                    
        
    

      
                    
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
