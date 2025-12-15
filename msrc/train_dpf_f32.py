import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy; print(numpy.__version__)
import math
import os, sys, time
from telnetlib import IP
import argparse
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter

from utils import post_process_depth, flip_lr, silog_loss, DN_to_distance, DN_to_depth, colormap, colormap_magma, colormap_viridis, compute_errors, eval_metrics, \
                       block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args, compute_seg, get_smooth_ND, normalize_image,compute_plane_loss, \
                       compute_strong_plane_loss
# from networks.NewCRFDepth import NewCRFDepth
from networks.NewCRFDepth_dpf import NewCRFDepth
from networks.dpf_fl32 import NewCRFDepth
from datetime import datetime

# from skimage.segmentation import all_felzenszwalb as felz_seg

parser = argparse.ArgumentParser(description='NDDepth PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='nddepth')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07, tiny07', default='large07')
parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default=None)

# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=5000)

# Training
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)
# Online eval
parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                 'if empty outputs to checkpoint folder', default='')
parser.add_argument('--accumulation_steps',        type=int,   help='gradient accumulation steps', default=1)

# ------------------------DPF
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
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from PIL import Image
def visualize_p_full(image, p_full, save_path, step):
    """
    image: (B, 3, H, W) 原始图像 Tensor
    p_full: (B, 3, H, W) 平面门控概率 Tensor
    save_path: 保存路径
    step: 当前步数
    """
    # 1. 数据预处理：取 Batch 中的第一个样本并转为 Numpy
    img = image[0].detach().cpu().numpy().transpose(1, 2, 0)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)*255
    img_np = img.astype(np.uint8)

    # p_full 形状为 (3, H, W)
    probs = p_full[0].detach().cpu().numpy()
    # 4. 保存原图
    os.makedirs(save_path, exist_ok=True)

    Image.fromarray(img_np).save(os.path.join(save_path, f'img_step_{step}.png'))
    # 5. 保存三个概率图（灰度图）
    for i, name in enumerate(['wall', 'floor', 'ceiling']):
        # 归一化概率到 0-255
        prob_img = (probs[i] * 255).astype(np.uint8)
        Image.fromarray(prob_img).save(
            os.path.join(save_path, f'prob_{name}_step_{step}.png')
        )
    
    # # 2. 创建绘图窗口
    # fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # # 展示原图
    # axes[0].imshow(img)
    # axes[0].set_title("Original Image")
    # axes[0].axis('off')
    
    # # 展示三个通道
    # titles = ['Wall Probability', 'Floor Probability', 'Ceiling Probability']
    # cmaps = ['Reds', 'Greens', 'Blues'] # 使用不同颜色区分
    
    # for i in range(3):
    #     im = axes[i+1].imshow(probs[i], cmap=cmaps[i], vmin=0, vmax=1)
    #     axes[i+1].set_title(titles[i])
    #     axes[i+1].axis('off')
    #     fig.colorbar(im, ax=axes[i+1], fraction=0.046, pad=0.04)

    # # 3. 保存
    # plt.tight_layout()
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # plt.savefig(os.path.join(save_path, f'p_full_step_{step}.png'))
    # plt.close()

    

# -----------------

# 再次提醒：train.py 中的 compute_gate_loss 必须是 Soft Label 版本
def compute_gate_loss(depth_geo, depth_pixel, depth_gt, gate_logits, mask_valid):
    err_geo = torch.abs(depth_geo - depth_gt)
    err_pixel = torch.abs(depth_pixel - depth_gt)
    
    # 软标签：生成 0~1 之间的 float target
    # 几何误差越小，target 越接近 1
    soft_target = err_pixel / (err_pixel + err_geo + 1e-6)
    
    # 使用 BCEWithLogitsLoss
    loss = F.binary_cross_entropy_with_logits(
        gate_logits[mask_valid], 
        soft_target[mask_valid].detach() 
    )
    return loss


def compute_smooth_loss(tgt_map):
    """
    计算梯度的 L1 范数 (平滑度损失)
    tgt_map: [B, C, H, W]
    return: gradient_map [B, 1, H, W] (为了和 mask 相乘，保持维度一致)
    """
    def gradient(x):
        # 计算 x 方向梯度: I(x+1) - I(x)
        # 计算 y 方向梯度: I(y+1) - I(y)
        h_x = x.size()[2]
        w_x = x.size()[3]
        
        # 为了保证维度对齐，我们在最后一行/列补零，或者切片时对齐
        # 这里采用切片对齐方式：
        # grad_x: [:, :, :, :-1]
        grad_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        grad_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        
        return grad_x, grad_y

    grad_x, grad_y = gradient(tgt_map)
    
    # 填充回原始大小，保持 H, W 一致以便和 Mask 相乘
    # grad_x 在 W 维度少 1，grad_y 在 H 维度少 1
    grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='constant', value=0)
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='constant', value=0)
    
    # 对 Channel 维度求均值 (或者求和)，得到每个像素点的梯度大小
    return grad_x.mean(1, keepdim=True) + grad_y.mean(1, keepdim=True)

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

            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            inv_K = torch.autograd.Variable(eval_sample_batched['inv_K'].cuda(gpu, non_blocking=True))
            depth1, u1, depth2, u2, n1_norm, distance,final_depth,R,P_full,z_plane,_=model(image, inv_K, epoch)
            # depth_final, gate_logits_full, depth1, u1, depth2, u2, n1, dist=model(image, inv_K, epoch)

            pred_depth=final_depth
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
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
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
        print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.4f}'.format(eval_measures_cpu[8]))
        return eval_measures_cpu

    return None

scaler = torch.amp.GradScaler("cuda")
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("== Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # model
    model = NewCRFDepth(version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=args.pretrain, mode='triple')
    model.train()

    
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
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
                checkpoint = torch.load(args.checkpoint_path,weights_only=False)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc,weights_only=False)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not args.retrain:
                try:
                    global_step = checkpoint['global_step']
                    best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                    best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                    best_eval_steps = checkpoint['best_eval_steps']
                except KeyError:
                    print("Could not load values for online evaluation")

            print("== Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True
        del checkpoint

    cudnn.benchmark = True

    dataloader = NewDataLoader(args, 'train')
    dataloader_eval = NewDataLoader(args, 'online_eval')

    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
        if args.do_online_eval:
            if args.eval_summary_directory != '':
                eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
            else:
                eval_summary_path = os.path.join(args.log_directory, args.model_name, 'eval')
            eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    silog_criterion = silog_loss(variance_focus=args.variance_focus)
    dn_to_depth = DN_to_depth(args.batch_size, args.input_height, args.input_width).cuda(args.gpu)
    dn_to_distance = DN_to_distance(args.batch_size, args.input_height, args.input_width).cuda(args.gpu)

    start_time = time.time()
    duration = 0

    num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("== Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))
    print("== Using gradient accumulation with {} steps".format(args.accumulation_steps))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch
   
    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus_per_node)])

    
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

          
            depth1, u1, depth2, u2, n1_norm, distance,final_depth,R,P_full,z_plane,_=model(image, inv_K, epoch)
            if args.dataset == 'nyu':
                    mask = depth_gt > 0.1
            else:
                    mask = depth_gt > 1.0
                
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

            
            normal_gt = torch.stack([normal_gt[:, 0], normal_gt[:, 2], normal_gt[:, 1]], 1)
            normal_gt_norm = F.normalize(normal_gt, dim=1, p=2)
            loss_normal = 5 * ((1 - (normal_gt_norm * n1_norm).sum(1, keepdim=True)) * mask.float()).sum() / (mask.float() + 1e-7).sum()
                
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
            if epoch < 5:
                w_geom = 0.0
            else:
                start_epoch = 5
                end_epoch = 20
                # 确保不超出范围
                progress = min(max((epoch - start_epoch) / (end_epoch - start_epoch), 0.0), 1.0)
                w_geom = 0.1 + (1.0 - 0.1) * progress  # 从0.1线性增加到1.0
                        # (min(epoch - 4, 3) / 3.0) * 1.0  # 0→1


            loss_final = silog_criterion.forward(final_depth, depth_gt, mask)

            loss = (  loss_depth1 + loss_depth2) + \
                loss_uncer1 + loss_uncer2 +loss_normal+loss_distance+ \
                w_geom*loss_geom+loss_final

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)

            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr
            
            scaler.step(optimizer)
            scaler.update()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                if global_step % 10 == 0:
                    print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))
                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    writer.add_scalar('loss', loss, global_step)

                    writer.add_scalar('silog_loss', (  loss_depth1 + loss_depth2) , global_step)
                    writer.add_scalar('uncer_loss', (loss_uncer1 + loss_uncer2), global_step)
                    writer.add_scalar('normal_loss', loss_normal, global_step)
                    writer.add_scalar('distance_loss', loss_distance, global_step)

                    writer.add_scalar('learning_rate', current_lr, global_step)
                    writer.add_scalar('var average', var_sum.item()/var_cnt, global_step)
                    writer.add_scalar('loss_geom', loss_geom, global_step)
                    
                
                    writer.flush()

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded :
                time.sleep(0.1)
                model.eval()
                with torch.no_grad():
                    # eval_measures = online_eval(model, dataloader_eval, gpu, ngpus_per_node, group, epoch, post_process=True)
                    eval_measures = online_eval(model, dataloader_eval, gpu, ngpus_per_node, group, epoch, post_process=False)
                if eval_measures is not None:
                    exp_name = '%s'%(datetime.now().strftime('%m%d'))
                    log_txt = os.path.join(args.log_directory + '/' + args.model_name, exp_name+'_logs.txt')
                    with open(log_txt, 'a') as txtfile:
                        txtfile.write(">>>>>>>>>>>>>>>>>>>>>>>>>Step:%d>>>>>>>>>>>>>>>>>>>>>>>>>\n"%(int(global_step)))
                        txtfile.write("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}\n".format('silog', 
                                        'abs_rel', 'log10', 'rms','sq_rel', 'log_rms', 'd1', 'd2','d3'))
                        # txtfile.write("depth estimation\n")
                        line = ''
                        for i in range(9):
                            line +='{:7.4f}, '.format(eval_measures[i])
                        txtfile.write(line+'\n')

                    for i in range(9):
                        eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                            old_best = best_eval_measures_higher_better[i-6].item()
                            best_eval_measures_higher_better[i-6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                            model_path = args.log_directory + '/' + args.model_name + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[i], measure)
                            print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                            checkpoint = {'global_step': global_step,
                                          'model': model.state_dict(),
                                          'optimizer': optimizer.state_dict(),
                                          'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                          'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                          'best_eval_steps': best_eval_steps
                                          }
                            torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                    eval_summary_writer.flush()
                model.train()
                block_print()
                enable_print()

                save_dir = os.path.join(args.log_directory, args.model_name, 'visual_debug')
                visualize_p_full(image, P_full, save_dir, global_step)

            if global_step and global_step % 20000 == 0:
                # 定义保存名称，区别于“best”模型，使用“step”标识
                model_save_name = '/model-{}-step.pth'.format(global_step)
                print('Step {}. Periodic model saving: {}'.format(global_step, model_save_name))
                
                periodic_checkpoint = {
                    'global_step': global_step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_eval_measures_higher_better': best_eval_measures_higher_better,
                    'best_eval_measures_lower_better': best_eval_measures_lower_better,
                    'best_eval_steps': best_eval_steps
                }
                
                # 保存路径：log_directory/model_name/model-20000-step.pth
                torch.save(periodic_checkpoint, args.log_directory + '/' + args.model_name + model_save_name)

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
    
    exp_name = '%s'%(datetime.now().strftime('%m%d'))  
    args.log_directory = os.path.join(args.log_directory,exp_name)  
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
        # command = 'cp nddepth/train.py ' + aux_out_path

        current_script = sys.argv[0]  # 当前执行的脚本
        script_name = os.path.basename(current_script)
        command = f'cp {current_script} {os.path.join(aux_out_path, script_name)}'
        os.system(command)
        command = 'mkdir -p ' + networks_savepath + ' && cp nddepth/networks/*.py ' + networks_savepath
        os.system(command)
        command = 'mkdir -p ' + dataloaders_savepath + ' && cp nddepth/dataloaders/*.py ' + dataloaders_savepath
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
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
