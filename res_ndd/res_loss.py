import torch
import torch.nn as nn
import torch.nn.functional as F

class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.silog_loss = SILogLoss()
        self.grad_loss = GradientLoss()
        self.normal_loss = SurfaceNormalLoss()
    
    def forward(self, outputs, batch_data):
        """
        outputs: model 返回的字典
        batch_data: dataloader 返回的字典 (包含 depth_gt, normal_gt 等)
        """
        depth_gt = batch_data['depth'].cuda()
        # 如果有 GT 法线更好，如果没有，可以从 Depth_GT 计算
        # 假设 dataloader 提供了 'normal' (NYUv2通常需要预处理生成)
        normal_gt = batch_data.get('normal', None) 
        if normal_gt is not None: normal_gt = normal_gt.cuda()
        
        mask = depth_gt > 0.001 # 有效区域 Mask
        
        # 1. 基础深度损失 (SILog)
        # 对最终结果 D_final 进行强监督
        loss_final = self.silog_loss(outputs['d_final'], depth_gt, mask)
        
        # 对两个分支进行辅助监督 (Auxiliary Loss)
        loss_geo_aux = self.silog_loss(outputs['d_geo'], depth_gt, mask)
        loss_res_aux = self.silog_loss(outputs['d_res'], depth_gt, mask)
        
        # 2. 显式几何损失 (法线监督)
        loss_normal = 0
        if normal_gt is not None:
            # 监督 N-D 分支预测的法线
            loss_normal = self.normal_loss(outputs['normal'], normal_gt, mask)
        
        # 3. 结构导向损失 (梯度损失)
        # 使得边缘更锐利
        loss_grad = self.grad_loss(outputs['d_final'], depth_gt, mask)
        
        # 4. (可选) 曼哈顿法线约束
        # 强迫主要法线方向正交，这里简化为正则化项
        # loss_manhattan = ... 
        
        # 总损失加权
        # 权重经验值: final=1.0, aux=0.5, normal=0.5, grad=0.5
        total_loss = 1.0 * loss_final + \
                     0.5 * loss_geo_aux + \
                     0.5 * loss_res_aux + \
                     0.5 * loss_normal + \
                     0.5 * loss_grad
                     
        return total_loss

# ------------------------------------------------------------------
# 具体 Loss 实现细节
# ------------------------------------------------------------------

class SILogLoss(nn.Module):
    """ Scale-Invariant Logarithmic Loss """
    def __init__(self, lamb=0.85):
        super().__init__()
        self.lamb = lamb

    def forward(self, pred, target, mask=None):
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        
        g = torch.log(pred + 1e-6) - torch.log(target + 1e-6)
        # SILog = sqrt(E[g^2] - lambda * (E[g])^2) * 10
        Dg = torch.var(g) + (1 - self.lamb) * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)

class SurfaceNormalLoss(nn.Module):
    """ Cosine Similarity Loss for Normals """
    def __init__(self):
        super().__init__()

    def forward(self, pred_n, target_n, mask=None):
        # pred_n, target_n: [B, 3, H, W]
        # Cosine similarity: <n1, n2>
        # Loss = 1 - cos_sim
        cos_sim = F.cosine_similarity(pred_n, target_n, dim=1)
        loss = 1.0 - cos_sim
        
        if mask is not None:
            # mask 通常是 [B, 1, H, W] -> [B, H, W]
            mask = mask.squeeze(1)
            loss = loss[mask]
            
        return loss.mean()

class GradientLoss(nn.Module):
    """ L1 Loss on gradients to sharpen edges """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None):
        # 计算 x, y 方向梯度
        def gradient(x):
            h_x = x.size()[-2]
            w_x = x.size()[-1]
            r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
            l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
            t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
            b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
            return torch.abs(r - l), torch.abs(t - b)

        pred_dx, pred_dy = gradient(pred)
        target_dx, target_dy = gradient(target)
        
        loss = torch.abs(pred_dx - target_dx) + torch.abs(pred_dy - target_dy)
        
        if mask is not None:
            loss = loss[mask]
            
        return loss.mean()