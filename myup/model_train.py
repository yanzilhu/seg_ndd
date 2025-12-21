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

    def forward(self, depth_geo, depth_reg, uncer_nd, uncer_reg, feat_nd, feat_reg):
        """
        depth_geo, depth_reg: [B, 1, H, W]
        uncer_nd, uncer_reg: [B, 1, H, W]
        feat_nd, feat_reg: [B, 128, H/4, W/4] 
        """
        # 1. 特征对齐 (如果 feature 还是 H/4, W/4，需要上采样)
        feat_nd = upsample(feat_nd, scale_factor=4)
        feat_reg=upsample(feat_reg, scale_factor=4)
        # if feat_nd.shape[2] != depth_geo.shape[2]:
        #     feat_nd = F.interpolate(feat_nd, size=depth_geo.shape[2:], mode='bilinear', align_corners=True)
        #     feat_reg = F.interpolate(feat_reg, size=depth_reg.shape[2:], mode='bilinear', align_corners=True)

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


        d_final, w_geo = self.fusion_module(depth_geo, depth_reg, pred_uncertainty_nd, pred_uncertainty_reg, e4, e0)


        return d_final,depth_geo,depth_reg,pred_normal,pred_distance,w_geo
    

    损失函数：
    class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
    
    def forward(self, pred, target, mask):
        # 计算 x 和 y 方向梯度
        pred_dy, pred_dx = self.gradient(pred)
        target_dy, target_dx = self.gradient(target)
        
        # 梯度差的 L1 Loss
        loss = torch.mean(torch.abs(pred_dy[mask] - target_dy[mask])) + \
               torch.mean(torch.abs(pred_dx[mask] - target_dx[mask]))
        return loss
    d_final,depth_geo,depth_reg,pred_normal,pred_distance,w_geo=model(image, inv_K, epoch)
                if args.dataset == 'nyu':
                    mask = depth_gt > 0.1
                else:
                    mask = depth_gt > 1.0
                normal_gt = torch.stack([normal_gt[:, 0], normal_gt[:, 2], normal_gt[:, 1]], 1)
                normal_gt_norm = F.normalize(normal_gt, dim=1, p=2)
                distance_gt = dn_to_distance(depth_gt, normal_gt_norm, inv_K_p)


                loss_depth_final=silog_criterion.forward(d_final, depth_gt, mask)
                loss_depth_geo =silog_criterion.forward(depth_geo, depth_gt, mask)
                loss_depth_reg  =silog_criterion.forward(depth_reg, depth_gt, mask)

                # --- B. 显式几何监督 (Normal & Distance) ---
                # 需要从 Depth GT 生成 Normal GT 和 Distance GT (Ground Truth Generation)
                # 这通常在 Dataloader 里做，或者在这里动态计算
                # 假设 targets 中已经有了由 GT Depth 生成的 pseudo-GT
                loss_normal = 5 * ((1 - (normal_gt_norm * pred_normal).sum(1, keepdim=True)) * mask.float()).sum() / (mask.float() + 1e-7).sum()
                loss_distance = 0.25 * torch.abs(distance_gt[mask] - pred_distance[mask]).mean()

                loss_grad =gradient_Loss.forward(d_final, depth_gt, mask)

                loss= loss_depth_final+0.5 * loss_depth_geo +0.5 * loss_depth_reg \
                    + loss_normal+loss_distance+10* loss_grad