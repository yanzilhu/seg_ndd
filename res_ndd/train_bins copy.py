import time
import sys
from datetime import datetime
from networks.NewCRFDepth_bins import NewCRFDepth
from utils import post_process_depth, flip_lr, silog_loss, DN_to_distance, DN_to_depth, colormap, colormap_magma, colormap_viridis, compute_errors, eval_metrics, \
    block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args, compute_seg, get_smooth_ND, normalize_image
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import argparse
from telnetlib import IP
import os
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy
print(numpy.__version__)


# from networks.NewCRFDepth import NewCRFDepth

# from skimage.segmentation import all_felzenszwalb as felz_seg

parser = argparse.ArgumentParser(
    description='NDDepth PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,
                    help='train or test', default='train')
parser.add_argument('--model_name',                type=str,
                    help='model name', default='nddepth')
parser.add_argument('--encoder',                   type=str,
                    help='type of encoder, base07, large07, tiny07', default='large07')
parser.add_argument('--pretrain',                  type=str,
                    help='path of pretrained encoder', default=None)

# Dataset
parser.add_argument('--dataset',                   type=str,
                    help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path',                 type=str,
                    help='path to the data', required=True)
parser.add_argument('--gt_path',                   type=str,
                    help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file',            type=str,
                    help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,
                    help='input height', default=480)
parser.add_argument('--input_width',               type=int,
                    help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float,
                    help='maximum depth in estimation', default=10)

# Log and save
parser.add_argument('--log_directory',             type=str,
                    help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,
                    help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,
                    help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                 type=int,
                    help='Checkpoint saving frequency in global steps', default=5000)

# Training
parser.add_argument('--weight_decay',              type=float,
                    help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--retrain',
                    help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                  type=float,
                    help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,
                    help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,
                    help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float,
                    help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float,
                    help='end learning rate', default=-1)
parser.add_argument('--variance_focus',            type=float,
                    help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Preprocessing
parser.add_argument('--do_random_rotate',
                    help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float,
                    help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',
                    help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',
                    help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,
                    help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                type=int,
                    help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                      type=int,
                    help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,
                    help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',              type=str,
                    help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,
                    help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)
# Online eval
parser.add_argument('--do_online_eval',
                    help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--data_path_eval',            type=str,
                    help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,
                    help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,
                    help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float,
                    help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float,
                    help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',
                    help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',
                    help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq',                 type=int,
                    help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                    'if empty outputs to checkpoint folder', default='')
parser.add_argument('--accumulation_steps',        type=int,
                    help='gradient accumulation steps', default=1)

# 修改后：
# 1. 生成平面 Mask (简单版：利用 GT 法线和视线的夹角，或者利用 GT 梯度的平滑度)
# 这里假设你没有额外的平面标签，我们用简单的梯度阈值来过滤掉边缘


def gradient(x):
    # 计算 x 方向梯度: I(x+1) - I(x)
    # 计算 y 方向梯度: I(y+1) - I(y)
    # h_x = x.size()[2]
    # w_x = x.size()[3]

    # 为了保证维度对齐，我们在最后一行/列补零，或者切片时对齐
    # 这里采用切片对齐方式：
    # grad_x: [:, :, :, :-1]
    grad_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    grad_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])

    return grad_x, grad_y
#  ai studio 有根据法线和距离的准则


def get_planar_mask(depth_gt):
    grad_x, grad_y = gradient(depth_gt)
    grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='constant', value=0)
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='constant', value=0)
    # 你得有个算梯度的函数
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
    # 梯度小的地方认为是平面
    planar_mask = grad_mag < 0.5  # 阈值需要根据数据调整
    return planar_mask


def save_w_geo_heatmap(w_geo, save_dir, step, img_idx_offset=0, cmap='jet'):
    """
    专门保存w_geo的热力图（直观展示权重分布，红=高权重，蓝=低权重）
    Args:
        w_geo: 权重张量，shape=[B,1,H,W]（GPU/CPU均可）
        save_dir: 热力图保存目录（自动创建）
        epoch: 训练轮数（用于文件名）
        batch_idx: 批次索引（用于文件名）
        img_idx_offset: 图片索引偏移（避免多批次索引重复）
        cmap: 热力图配色方案（jet/rainbow/viridis等，推荐jet）
    """
    # 1. 创建保存目录
    # os.makedirs(save_dir, exist_ok=True)

    # 2. 处理张量：GPU→CPU → 去通道维度 → 转numpy → 确保值在0~1（Softmax输出已满足）
    w_geo_np = w_geo.detach().cpu().squeeze(1).numpy()  # [B, H, W]

    # 3. 批量保存每张图的热力图
    batch_size = w_geo_np.shape[0]
    filename = f"w_geo_heatmap_epoch{step}.png"
    save_path = os.path.join(save_dir, filename)

    # 4. 绘制并保存热力图（无需归一化，w_geo本身是0~1）
    plt.imsave(
        save_path,
        w_geo_np[0],          # 当前图片的权重矩阵 [H, W]
        cmap=cmap,            # 配色方案（jet：蓝→青→黄→红，红=高权重）
        vmin=0.0,             # 最小值（对应蓝）
        vmax=1.0,             # 最大值（对应红）
        dpi=150               # 分辨率（越高越清晰）
    )
    # print(f"已保存{batch_size}张w_geo热力图到：{save_dir}")

# 辅助类：梯度损失


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

    def gradient(self, x):
        # 简单的 Sobel 或者 差分
        h_x = x.size()[-2]
        w_x = x.size()[-1]
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + 1e-6, 0.5)
        ygrad = torch.pow(torch.pow((t - b) * 0.5, 2) + 1e-6, 0.5)
        return ygrad, xgrad


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader


def online_eval(model, dataloader_eval, gpu, ngpus, group, epoch, post_process=False):

    # 创建保存可视化结果的文件夹 (建议)
    # vis_dir = f"./vis_results/epoch_{epoch}"
    # if not os.path.exists(vis_dir):
    #     os.makedirs(vis_dir)

    eval_measures = torch.zeros(10).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                continue

            image = torch.autograd.Variable(
                eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            inv_K = torch.autograd.Variable(
                eval_sample_batched['inv_K'].cuda(gpu, non_blocking=True))
            d_final, _, _, _, _, _, _, _, _ = model(image, inv_K, epoch)
            # d_final, d_geo, d_bins, pred_normal, pred_dist, alpha_map, u_geo, u_bins, offset_up
            # d_final,depth_geo,depth_reg,pred_normal,pred_distance,w_geo=model(image, inv_K, epoch)
            pred_depth = d_final
            # if post_process:
            #     image_flipped = flip_lr(image)
            #     depth1_flipped, u1_flipped, depth2_flipped, u2_flipped, n1_norm_flipped, distance_flipped,final_depth_flipped,_,_,_= model(image_flipped, inv_K, epoch)
            #     pred_depth = post_process_depth(final_depth, depth1_flipped)
            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352,
                                 left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(
            gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                          int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                              int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1

    # if args.multiprocessing_distributed:
    #     # group = dist.new_group([i for i in range(ngpus)])
    #     dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    # if not args.multiprocessing_distributed or
    if gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(
            int(cnt)), ', post_process: ', post_process)
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.4f}'.format(eval_measures_cpu[8]))
        return eval_measures_cpu

    return None


scaler = torch.amp.GradScaler("cuda")


# -----------------------------------------------------------------------------
# 辅助函数：从深度图计算法线 (用于 DTN Loss - 双向几何一致性)
# -----------------------------------------------------------------------------

def depth_to_normal(depth, inv_K):
    """
    通过最小二乘法或梯度叉乘从深度图恢复法线
    这里使用高效的梯度叉乘法 (Cross Product of Gradients)
    depth: [B, 1, H, W]
    inv_K: [B, 3, 3]
    """
    B, _, H, W = depth.shape
    device = depth.device

    # 1. 生成像素坐标
    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing='ij')
    grid = torch.stack([xx, yy, torch.ones_like(xx)], dim=0).float(
    ).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 3, H, W]

    # 2. 反投影到 3D (Back-projection)
    # P = depth * (K^-1 * uv)
    inv_K_expanded = inv_K.view(B, 3, 3, 1, 1)
    # [B, 3, 3] x [B, 3, H*W] -> [B, 3, H*W]
    cam_coords = torch.matmul(inv_K, grid.view(B, 3, -1)).view(B, 3, H, W)
    points_3d = depth * cam_coords

    # 3. 计算梯度 (Central Difference)
    # Pad 为了保持尺寸
    p_pad = F.pad(points_3d, (1, 1, 1, 1), mode='replicate')

    # x方向向量: P(x+1) - P(x-1)
    vec_x = p_pad[:, :, 1:-1, 2:] - p_pad[:, :, 1:-1, :-2]
    # y方向向量: P(y+1) - P(y-1)
    vec_y = p_pad[:, :, 2:, 1:-1] - p_pad[:, :, :-2, 1:-1]

    # 4. 叉乘得到法线
    normal = torch.cross(vec_x, vec_y, dim=1)
    normal = F.normalize(normal, dim=1, p=2)  # [B, 3, H, W]

    return normal

# -----------------------------------------------------------------------------
# 辅助函数：计算平面掩码 (用于 Distance Loss)
# -----------------------------------------------------------------------------


def get_planar_mask(depth_gt):
    # 计算深度梯度，梯度小的地方认为是平面
    # 使用简单的 Sobel 算子
    # 也可以利用 GT Normal 的梯度，这里用 Depth 梯度简化
    dy, dx = torch.gradient(depth_gt, dim=(2, 3))
    grad_mag = torch.sqrt(dy**2 + dx**2)
    # 阈值需根据数据尺度调整，NYU一般0.5-1.0有效
    planar_mask = grad_mag < 0.5
    return planar_mask

# -----------------------------------------------------------------------------
# 基础 Loss 类
# -----------------------------------------------------------------------------


class SILogLoss(nn.Module):
    def __init__(self, variance_focus=0.85):
        super(SILogLoss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, pred, gt, mask):
        # 避免 log(0)
        d = torch.log(pred[mask] + 1e-6) - torch.log(gt[mask] + 1e-6)
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.scales = [1, 2, 4]

    def forward(self, pred, gt, mask):
        total_loss = 0
        for scale in self.scales:
            step = scale
            pred_s = pred[:, :, ::step, ::step]
            gt_s = gt[:, :, ::step, ::step]
            mask_s = mask[:, :, ::step, ::step]

            if mask_s.sum() == 0:
                continue

            diff = torch.abs(pred_s - gt_s)

            # 计算 x, y 方向梯度
            grad_x = torch.abs(diff[:, :, :, 1:] - diff[:, :, :, :-1])
            grad_y = torch.abs(diff[:, :, 1:, :] - diff[:, :, :-1, :])

            # 只计算有效区域
            m_x = mask_s[:, :, :, 1:] & mask_s[:, :, :, :-1]
            m_y = mask_s[:, :, 1:, :] & mask_s[:, :, :-1, :]

            if m_x.sum() > 0:
                total_loss += grad_x[m_x].mean()
            if m_y.sum() > 0:
                total_loss += grad_y[m_y].mean()

        return total_loss

# -----------------------------------------------------------------------------
# 主 Loss 类：GeoPromptLoss
# -----------------------------------------------------------------------------


class GeoPromptLoss(nn.Module):
    def __init__(self, min_depth=0.1, max_depth=10.0):
        super(GeoPromptLoss, self).__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

        # 1. 深度监督
        self.silog_loss = SILogLoss()

        # 2. 梯度监督 (锐化边缘)
        self.grad_loss = GradientLoss()

        # 3. 几何监督 (L1)
        self.l1_loss = nn.L1Loss(reduction='mean')

    def uncertainty_loss(self, pred, gt, uncertainty, mask):
        """
        不确定性驱动的回归损失
        Loss = |y - y_hat| / sigma + log(sigma)
        """

        mask = F.interpolate(mask.float(),
                             size=uncertainty.shape[-2:],
                             mode='nearest')
        mask = mask.bool()  # 保持bool类型

        pred = F.interpolate(pred,
                             size=uncertainty.shape[-2:],
                             mode='bilinear',
                             align_corners=True)
        gt = F.interpolate(gt,
                           size=uncertainty.shape[-2:],
                           mode='bilinear',
                           align_corners=True)

        abs_diff = torch.abs(pred - gt)
        # 加上 mask
        abs_diff = abs_diff[mask]
        uncertainty = uncertainty[mask]

        # 加上 eps 防止除以0
        # loss = abs_diff / (uncertainty + 1e-6) + torch.log(uncertainty + 1e-6)
        # 修改
        loss = abs_diff * torch.exp(-uncertainty) + \
            uncertainty      # s = log_sigma

        return loss.mean()

    def forward(self, outputs, targets, epoch):
        """
        outputs: 字典或元组，包含模型输出
        targets: 字典，包含 'depth', 'normal', 'inv_K' 等
        """
        # 解包输入
        d_final = outputs['d_final']
        d_geo = outputs['d_geo']
        d_bins = outputs['d_bins']
        pred_normal = outputs['pred_normal']
        pred_dist = outputs['pred_dist']
        u_geo = outputs['u_geo']
        u_bins = outputs['u_bins']
        offset_up = outputs['offset_up']

        depth_gt = targets['depth']
        normal_gt = targets['normal']
        inv_K = targets['inv_K']
        mask = targets['mask']

        # --- 1. 深度分支监督 (Deep Supervision) ---
        # A. 最终融合深度 (Main Loss)
        loss_final = self.silog_loss(d_final, depth_gt, mask)

        # B. 几何深度 (带不确定性)
        # 几何分支容易在边缘出错，所以用 Uncertainty Loss 允许它在边缘“不自信”
        loss_geo = self.uncertainty_loss(d_geo, depth_gt, u_geo, mask)

        # C. Bin 深度 (带不确定性)
        loss_bins = self.uncertainty_loss(d_bins, depth_gt, u_bins, mask)

        # --- 2. 几何分支监督 (Normal & Distance) ---
        # A. 法线监督 (Cosine Similarity)
        # GT Normal 通常需要归一化
        # mask 广播到 3 通道
        mask_3 = mask.repeat(1, 3, 1, 1)
        valid_normal = (normal_gt.abs().sum(
            dim=1, keepdim=True) > 0)  # 过滤掉无效 GT
        normal_mask = mask & valid_normal

        if normal_mask.sum() > 0:
            # Cosine Loss: 1 - cos(theta)
            cos_sim = F.cosine_similarity(pred_normal, normal_gt, dim=1)
            loss_normal = torch.mean(1.0 - cos_sim[normal_mask.squeeze(1)])
        else:
            loss_normal = 0.0

        # B. 距离监督 (L1, 仅在平面区域)
        # 动态计算 GT Distance: d = N^T * P
        # 需要先算出 Point Cloud
        with torch.no_grad():
            # 使用你现有的 DN_to_distance 模块或者手动计算
            # 这里手动快速计算 GT Distance
            B, _, H, W = depth_gt.shape
            device = depth_gt.device
            yy, xx = torch.meshgrid(torch.arange(
                H, device=device), torch.arange(W, device=device), indexing='ij')
            grid = torch.stack([xx, yy, torch.ones_like(xx)],
                               dim=0).float().unsqueeze(0).repeat(B, 1, 1, 1)
            cam_coords = torch.matmul(
                inv_K[..., :3, :3], grid.view(B, 3, -1)).view(B, 3, H, W)

            points_gt = depth_gt * cam_coords  # [B, 3, H, W]
            dist_gt = torch.sum(normal_gt * points_gt, dim=1,
                                keepdim=True).abs()  # [B, 1, H, W]

        planar_mask = get_planar_mask(depth_gt)
        valid_dist_mask = mask & planar_mask

        if valid_dist_mask.sum() > 0:
            loss_dist = self.l1_loss(
                pred_dist[valid_dist_mask], dist_gt[valid_dist_mask])
        else:
            loss_dist = 0.0

        # --- 3. 几何一致性监督 (DTN Loss - 创新点！) ---
        # 强制 Prompt Bin 分支预测出的深度图，在转换为法线后，也必须准确
        # 这就是“王智彬论文”中的核心约束
        pred_normal_from_bins = depth_to_normal(d_bins, inv_K[..., :3, :3])
        # 1：可视化

        if normal_mask.sum() > 0:
            cos_sim_bins = F.cosine_similarity(
                pred_normal_from_bins, normal_gt, dim=1)
            loss_dtn = torch.mean(1.0 - cos_sim_bins[normal_mask.squeeze(1)])
        else:
            loss_dtn = 0.0

        # --- 4. 辅助监督 (Reg & Grad) ---
        # 偏移正则化 (防止 Offset 乱飘)
        loss_offset = torch.mean(torch.abs(offset_up))

        # 梯度损失 (边缘锐利)
        loss_grad = self.grad_loss(d_final, depth_gt, mask)

        if epoch < 5:
            # 只用 loss_final + 轻量 normal_loss
            total_loss = 1.5 * loss_final + 0.05 * loss_normal
        else:
            # 再加 geo / bins / dist / dtn 等
            total_loss = 1.0 * loss_final + \
                0.5 * loss_geo + \
                0.5 * loss_bins + \
                0.2 * loss_dist + \
                0.5 * loss_dtn + \
                0.1 * loss_offset + \
                0.05 * loss_grad + \
                0.05 * loss_normal

        # --- 5. 总 Loss 加权 ---
        # 权重建议 (根据 NDDepth 和其他论文经验)

        loss_dict = {
            'total': total_loss.item(),
            'silog': loss_final.item(),
            'geo': loss_geo.item(),
            'bins': loss_bins.item(),
            'normal': loss_normal if isinstance(loss_normal, float) else loss_normal.item(),
            'dist': loss_dist if isinstance(loss_dist, float) else loss_dist.item(),
            'dtn': loss_dtn if isinstance(loss_dtn, float) else loss_dtn.item(),
            'offset': loss_offset.item()
        }

        return total_loss, loss_dict


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("== Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # model
    model = NewCRFDepth(version=args.encoder, inv_depth=False,
                        max_depth=args.max_depth, pretrained=args.pretrain, mode='triple')
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape)
                            for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()
    # if int(torch.__version__.split('.')[0]) >= 2:
    #     print("== Using torch.compile for acceleration")
    #     model = torch.compile(model)

    if args.distributed:
        print("== Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("== Model Initialized")

    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)

    # Training parameters
    optimizer = torch.optim.Adam([{'params': model.module.parameters()}],
                                 lr=args.learning_rate)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(
                    args.checkpoint_path, weights_only=False)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(
                    args.checkpoint_path, map_location=loc, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not args.retrain:
                try:
                    global_step = checkpoint['global_step']
                    best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu(
                    )
                    best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu(
                    )
                    best_eval_steps = checkpoint['best_eval_steps']
                except KeyError:
                    print("Could not load values for online evaluation")

            print("== Loaded checkpoint '{}' (global_step {})".format(
                args.checkpoint_path, checkpoint['global_step']))
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True
        del checkpoint

    cudnn.benchmark = True

    dataloader = NewDataLoader(args, 'train')
    dataloader_eval = NewDataLoader(args, 'online_eval')

    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_directory + '/' +
                               args.model_name + '/summaries', flush_secs=30)
        if args.do_online_eval:
            if args.eval_summary_directory != '':
                eval_summary_path = os.path.join(
                    args.eval_summary_directory, args.model_name)
            else:
                eval_summary_path = os.path.join(
                    args.log_directory, args.model_name, 'eval')
            eval_summary_writer = SummaryWriter(
                eval_summary_path, flush_secs=30)

    silog_criterion = silog_loss(variance_focus=args.variance_focus)
    # 修改
    # dn_to_depth = DN_to_depth(args.batch_size, args.input_height, args.input_width).cuda(args.gpu)
    dn_to_depth = DN_to_depth().cuda(args.gpu)
    dn_to_distance = DN_to_distance(
        args.batch_size, args.input_height, args.input_width).cuda(args.gpu)
    gradient_Loss = GradientLoss()
    start_time = time.time()
    duration = 0

    num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != - \
        1 else 0.1 * args.learning_rate

    var_sum = [var.sum().item()
               for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("== Initial variables' sum: {:.3f}, avg: {:.3f}".format(
        var_sum, var_sum/var_cnt))
    print("== Using gradient accumulation with {} steps".format(
        args.accumulation_steps))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus_per_node)])
    save_dir = "result/w_geo_norm"
    os.makedirs(save_dir, exist_ok=True)

    criterion = GeoPromptLoss(min_depth=0, max_depth=args.max_depth).cuda()
    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()

            loss = 0
            # loss_depth1 = 0
            # loss_depth2 = 0

            before_op_time = time.time()

            image = torch.autograd.Variable(
                sample_batched['image'].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(
                sample_batched['depth'].cuda(args.gpu, non_blocking=True))
            normal_gt = torch.autograd.Variable(
                sample_batched['normal'].cuda(args.gpu, non_blocking=True))
            inv_K = torch.autograd.Variable(
                sample_batched['inv_K'].cuda(args.gpu, non_blocking=True))
            inv_K_p = torch.autograd.Variable(
                sample_batched['inv_K_p'].cuda(args.gpu, non_blocking=True))

            # 注意：forward 返回值顺序变了，要对应上
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                d_final, d_geo, d_bins, pred_normal, pred_dist, alpha_map, u_geo, u_bins, offset_up = model(
                    image, inv_K, epoch)
                if args.dataset == 'nyu':
                    mask = depth_gt > 0.1
                else:
                    mask = depth_gt > 1.0

                # 3. 准备 Outputs 字典
                outputs = {
                    'd_final': d_final,
                    'd_geo': d_geo,
                    'd_bins': d_bins,
                    'pred_normal': pred_normal,
                    'pred_dist': pred_dist,
                    'u_geo': u_geo,
                    'u_bins': u_bins,
                    'offset_up': offset_up
                }
                targets = {
                    'depth': depth_gt,
                    'normal': normal_gt,
                    'inv_K': inv_K,
                    'mask': mask
                }
                # 5. 计算 Loss
                # if epoch < 5:

                loss, loss_dict_val = criterion(outputs, targets, epoch)

                total_loss = loss_dict_val['total']
                silog_loss_val = loss_dict_val['silog']
                geo_loss = loss_dict_val['geo']
                bins_loss = loss_dict_val['bins']
                normal_loss = loss_dict_val['normal']
                dist_loss = loss_dict_val['dist']
                dtn_loss = loss_dict_val['dtn']
                offset_loss = loss_dict_val['offset']

            # loss.backward()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10, norm_type=2)

            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * \
                    (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                if global_step % 10 == 0:
                    print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(
                        epoch, step, steps_per_epoch, global_step, current_lr, loss))
                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                var_sum = [var.sum().item()
                           for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (
                    num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(
                ), var_sum.item()/var_cnt, time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    writer.add_scalar('loss', loss, global_step)
                    writer.add_scalar(
                        'total loss',  loss_dict_val['total'], global_step)
                    writer.add_scalar(
                        'silog', loss_dict_val['silog'], global_step)
                    writer.add_scalar('geo_loss', geo_loss, global_step)
                    writer.add_scalar('bins_loss', bins_loss, global_step)
                    writer.add_scalar('normal_loss', normal_loss, global_step)
                    writer.add_scalar('dist_loss', dist_loss, global_step)

                    writer.add_scalar('dtn_loss', dtn_loss, global_step)
                    writer.add_scalar('offset_loss', offset_loss, global_step)

                    writer.flush()

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                time.sleep(0.1)
                model.eval()
                with torch.no_grad():
                    # eval_measures = online_eval(model, dataloader_eval, gpu, ngpus_per_node, group, epoch, post_process=True)
                    eval_measures = online_eval(
                        model, dataloader_eval, gpu, ngpus_per_node, group, epoch, post_process=False)
                    # save_w_geo_heatmap(w_geo,save_dir,global_step)
                if eval_measures is not None:
                    exp_name = '%s' % (datetime.now().strftime('%m%d'))
                    log_txt = os.path.join(
                        args.log_directory + '/' + args.model_name, exp_name+'_logs.txt')
                    with open(log_txt, 'a') as txtfile:
                        txtfile.write(">>>>>>>>>>>>>>>>>>>>>>>>>Step:%d>>>>>>>>>>>>>>>>>>>>>>>>>\n" % (
                            int(global_step)))
                        txtfile.write("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}\n".format('silog',
                                                                                                               'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
                        txtfile.write("depth estimation\n")
                        line = ''
                        for i in range(9):
                            line += '{:7.4f}, '.format(eval_measures[i])
                        txtfile.write(line+'\n')

                    for i in range(9):
                        eval_summary_writer.add_scalar(
                            eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item(
                            )
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                            old_best = best_eval_measures_higher_better[i-6].item(
                            )
                            best_eval_measures_higher_better[i -
                                                             6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(
                                old_best_step, eval_metrics[i], old_best)
                            model_path = args.log_directory + '/' + args.model_name + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(
                                global_step, eval_metrics[i], measure)
                            print('New best for {}. Saving model: {}'.format(
                                eval_metrics[i], model_save_name))
                            checkpoint = {'global_step': global_step,
                                          'model': model.state_dict(),
                                          'optimizer': optimizer.state_dict(),
                                          'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                          'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                          'best_eval_steps': best_eval_steps
                                          }
                            torch.save(checkpoint, args.log_directory +
                                       '/' + args.model_name + model_save_name)
                    eval_summary_writer.flush()
                model.train()
                block_print()
                enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer.close()
        if args.do_online_eval:
            eval_summary_writer.close()


def main():
    if args.mode != 'train':
        print('train.py is only for training.')
        return -1

    exp_name = '%s' % (datetime.now().strftime('%m%d'))
    args.log_directory = os.path.join(args.log_directory, exp_name)
    command = 'mkdir ' + os.path.join(args.log_directory, args.model_name)
    os.system(command)

    args_out_path = os.path.join(args.log_directory, args.model_name)
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)

    save_files = True
    if save_files:
        aux_out_path = os.path.join(args.log_directory, args.model_name)
        networks_savepath = os.path.join(aux_out_path, 'networks')
        dataloaders_savepath = os.path.join(aux_out_path, 'dataloaders')

        # ===== 核心修改：动态获取当前文件夹路径 =====
        current_script = sys.argv[0]  # 当前执行的脚本路径（如 ./train.py 或 /a/b/train.py）
        current_dir = os.path.dirname(
            os.path.abspath(current_script))  # 当前脚本所在文件夹的绝对路径
        # 若 networks/dataloaders 在当前文件夹下，直接拼接
        networks_src = os.path.join(current_dir, 'networks', '*.py')
        dataloaders_src = os.path.join(current_dir, 'dataloaders', '*.py')

        # 复制当前执行脚本（原有逻辑）
        script_name = os.path.basename(current_script)
        command = f'cp {current_script} {os.path.join(aux_out_path, script_name)}'
        os.system(command)

        # ===== 替换硬编码的 nddepth，使用动态路径 =====
        command = f'mkdir -p {networks_savepath} && cp {networks_src} {networks_savepath}'
        os.system(command)
        command = f'mkdir -p {dataloaders_savepath} && cp {dataloaders_src} {dataloaders_savepath}'
        os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
