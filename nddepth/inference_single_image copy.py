import torch
import torch.backends.cudnn as cudnn

import numpy; print(numpy.__version__)
from numpy.distutils.core import setup
import os, sys
import argparse
import numpy as np
from tqdm import tqdm

from utils import post_process_depth, flip_lr, compute_errors
from pointcloud import files_to_pointcloud, arrays_to_pointcloud
from networks.NewCRFDepth import NewCRFDepth
from PIL import Image 
from torchvision import transforms
import matplotlib.pyplot as plt


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='NDDepth PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name',  type=str,   help='model name', default='nddepth')
parser.add_argument('--encoder', type=str,   help='type of encoder, base07, large07', default='large07')
parser.add_argument('--checkpoint_path',type=str,   help='path to a checkpoint to load', default='checkpoint/nyu.pth')
parser.add_argument('--dataset', type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--image_path', type=str, help='path to the image for inference', default="/home/huangtupo/MYZ/bysj/NDDepth/Estimation/mytest/nyu/rgb_00001.jpg")
                    # "/home/huangtupo/MYZ/bysj/NDDepth/Estimation/mytest/ondoor/3.jpg")
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()


def inference(model, epoch=5, post_process=False):
    
    image = np.asarray(Image.open(args.image_path), dtype=np.float32) / 255.0

    # # ------------- 新增：将图像 resize 为 640×480（宽×高）-------------
    # 使用 PIL 库的 resize 函数，注意参数是 (width, height)
    image = Image.fromarray((image * 255).astype(np.uint8))  # 转回 uint8 格式 PIL 处理
    image = image.resize((640, 480), Image.BILINEAR)  # 调整为 640×480，双线性插值
    image = np.asarray(image, dtype=np.float32) / 255.0  # 再次归一化到 [0, 1]
    # -----------------------------------------------------------------
    if args.dataset == 'kitti':
        height = image.shape[0]
        width = image.shape[1]
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

    image = torch.from_numpy(image.transpose((2, 0, 1)))
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

    if args.dataset == 'kitti':
        K = np.array([[716.88 / 4.0, 0, 596.5593 / 4.0, 0],
                  [0, 716.88 / 4.0, 149.854 / 4.0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
        inv_K = np.linalg.pinv(K)
        inv_K = torch.from_numpy(inv_K)
           
    elif args.dataset == 'nyu':
        K = np.array([[518.8579 / 4.0, 0, 325.5824 / 4.0, 0],
                  [0, 518.8579 / 4.0, 253.7362 / 4.0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32) 
        inv_K = np.linalg.pinv(K)
        inv_K = torch.from_numpy(inv_K)

    with torch.no_grad():
        image = torch.autograd.Variable(image.unsqueeze(0).cuda())
        inv_K = torch.autograd.Variable(inv_K.unsqueeze(0).cuda())

        depth1_list, uncer1_list, depth2_list, uncer2_list, normals, distances = model(image, inv_K, epoch)
        if epoch < 5:
            pred_depth = 0.5 * (depth1_list + depth2_list)
        else:
            pred_depth = 0.5 * (depth1_list[-1] + depth2_list[-1])
        if post_process:
            image_flipped = flip_lr(image)
            depth1_list_flipped, uncer1_list_flipped, depth2_list_flipped, uncer2_list_flipped, normals_flipped, distances_flipped = model(image_flipped, inv_K, epoch)
            if epoch < 5:
                pred_depth_flipped = 0.5 * (depth1_list_flipped + depth2_list_flipped)
            else:
                pred_depth_flipped = 0.5 * (depth1_list_flipped[-1] + depth2_list_flipped[-1])
            pred_depth = post_process_depth(pred_depth, pred_depth_flipped)
        normals = (0.5 * (normals + 1)).permute(0, 2, 3, 1)

        pred_depth = pred_depth.cpu().numpy().squeeze()
        pred_normal = normals.cpu().numpy().squeeze()
       
        if args.dataset == 'kitti':
            plt.imsave('depth.png', np.log10(pred_depth), cmap='magma')
            plt.imsave('normal.png', pred_normal)
            # save point cloud for kitti if needed (user can call files_to_pointcloud)
        else:
            # NYU: save depth as float image and also write PLY
            depth_path = 'depth_nyu.png'
            plt.imsave(depth_path, pred_depth, cmap='jet')
            plt.imsave('normal_nyu.png', pred_normal)
            # Attempt to save colored point cloud using NYU intrinsics
            try:
                # read original input RGB (resized earlier to 640x480)
                rgb_img = Image.open(args.image_path).resize((640, 480), Image.BILINEAR)
                rgb_np = np.asarray(rgb_img).astype(np.float32) / 255.0
                # pred_depth is already resized to 480x640 according to the code path
                out_ply = 'mytest/ply/predicted_nyu.ply'
                arrays_to_pointcloud(rgb_np, pred_depth, {'fx':518.8579,'fy':518.8579,'cx':325.5824,'cy':253.7362}, out_ply, depth_scale=1.0)
                print(f"Saved point cloud to {out_ply}")
            except Exception as e:
                print('Failed to save point cloud:', e)
          
def main_worker(args):

    model = NewCRFDepth(version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=None)
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    model = torch.nn.DataParallel(model)
    model.cuda()

    print("== Model Initialized")

    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print("== Loaded checkpoint '{}'".format(args.checkpoint_path))
            del checkpoint
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))

    cudnn.benchmark = True

    # ===== Inference ======
    model.eval()
    with torch.no_grad():
        inference(model, post_process=True)


def main():
    torch.cuda.empty_cache()
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1
    
    main_worker(args)


if __name__ == '__main__':
    main()
