import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

import numpy as np
from PIL import Image
import os
import random
import copy

from utils import DistributedSamplerNoEvenlyDivisible


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class NewDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None
    
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   drop_last=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
        
        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))
            
            
class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()
    
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
    
    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        # focal = float(sample_path.split()[2])
        focal = 518.8579

        if self.mode == 'train':
            if self.args.dataset == 'kitti':
                rgb_file = sample_path.split()[0]
                depth_file = os.path.join(sample_path.split()[0].split('/')[0], sample_path.split()[1])
                normal_file = depth_file[:-14] + 'normal_' + depth_file[-14:]
                if self.args.use_right is True and random.random() > 0.5:
                    rgb_file = rgb_file.replace('image_02', 'image_03')
                    depth_file = depth_file.replace('image_02', 'image_03')
                    normal_file = normal_file.replace('image_02', 'image_03')
            else:
                rgb_file = sample_path.split()[0]
                depth_file = sample_path.split()[1]
                normal_file = depth_file[:-15] + 'normal_' + depth_file[-9:]

            image_path = os.path.join(self.args.data_path, rgb_file)
            depth_path = os.path.join(self.args.gt_path, depth_file)
            normal_path = os.path.join(self.args.gt_path, normal_file)
    
            # mask 
            mask_path = image_path.replace('/sync/', '/semantic_masks/').replace('.jpg', '.png')    
        #   容错处理：如果替换后路径不对（比如 data_path 本身不含 sync），尝试直接拼接
            if not os.path.exists(mask_path):
                 # 假设 rgb_file 是类似 basement_0001a/rgb_00000.jpg
                 mask_root = '/media/huangtupo/LHYZ/BYSJ/data/ndd_modify_package/nyu_depth_v2/semantic_masks'
                 mask_path = os.path.join(mask_root, rgb_file.replace('.jpg', '.png'))

            # 读取图片、深度、法向

            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)
            normal = Image.open(normal_path)
            
            # ------duqu mask
            try:
                mask_planar = Image.open(mask_path).convert('L')
            except IOError:
                # 如果找不到mask，给一个全黑的，避免报错崩溃
                mask_planar = Image.new('L', image.size, 0)
            
            # -----
            if self.args.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                normal = normal.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                # [新增] 同步裁剪 Mask
                mask_planar = mask_planar.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # To avoid blank boundaries due to pixel registration
            if self.args.dataset == 'nyu':
                if self.args.input_height == 480:
                    depth_gt = np.array(depth_gt)
                    normal = np.array(normal)
                    valid_mask = np.zeros_like(depth_gt)
                    valid_mask[45:472, 43:608] = 1
                    depth_gt[valid_mask==0] = 0
                    normal[valid_mask==0] = 0
                    depth_gt = Image.fromarray(depth_gt)
                    normal = Image.fromarray(normal)
                    # [新增] Mask 处理
                    # 这里的 valid_mask 是原代码为了处理边界黑边做的，我们对 mask_planar 也要做同样处理吗？
                    # 通常语义分割 Mask 已经是对应原图的，不需要额外 mask，但为了保持尺寸一致，转 numpy
                    mask_planar = np.array(mask_planar)
                    # 注意：如果原代码在这里对 depth_gt 做了屏蔽，Mask 最好也保持一致，不过通常不需要
                    mask_planar = Image.fromarray(mask_planar)

                else:
                    depth_gt = depth_gt.crop((43, 45, 608, 472))
                    image = image.crop((43, 45, 608, 472))
                    normal = normal.crop((43, 45, 608, 472))
                    # [新增] 同步裁剪 Mask 统一裁剪所有数据到NYUv2的标准有效区域。crop((left, top, right, bottom))
                    mask_planar = mask_planar.crop((43, 45, 608, 472))
    
            if self.args.do_random_rotate is True: 
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
                normal = self.rotate_image(normal, random_angle, flag=Image.NEAREST)
                # [新增] 同步旋转 Mask (使用最近邻插值保持 0/255 离散值)
                mask_planar = self.rotate_image(mask_planar, random_angle, flag=Image.NEAREST)
            
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            normal = np.asarray(normal, dtype=np.float32) / 127.5 -1

            # [新增] Mask 转 Numpy 并归一化
            mask_planar = np.asarray(mask_planar, dtype=np.float32)
            # 确保 Mask 是 0 或 1 (输入是0或255)
            mask_planar = (mask_planar > 127.5).astype(np.float32)
            mask_planar = np.expand_dims(mask_planar, axis=2) # [H, W, 1]

            if self.args.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            else:
                depth_gt = depth_gt / 256.0

            if image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width:
                # image, depth_gt, normal, offset = self.random_crop(image, depth_gt, normal, self.args.input_height, self.args.input_width)

                # [修改] 传入 mask_planar 到 random_crop
                image, depth_gt, normal, mask_planar, offset = self.random_crop(image, depth_gt, normal, mask_planar, self.args.input_height, self.args.input_width)
            if self.args.dataset == 'nyu':
                offset = 0
            # image, depth_gt, normal = self.train_preprocess(image, depth_gt, normal)
            # [修改] 传入 mask_planar 到 train_preprocess
            image, depth_gt, normal, mask_planar = self.train_preprocess(image, depth_gt, normal, mask_planar)


            # https://github.com/ShuweiShao/URCDC-Depth
            # image, depth_gt, normal = self.Cut_Flip(image, depth_gt, normal)
            # [修改] 传入 mask_planar 到 Cut_Flip
            image, depth_gt, normal, mask_planar = self.Cut_Flip(image, depth_gt, normal, mask_planar)
            
            # sample = {'image': image, 'normal': normal, 'depth': depth_gt, 'offset': offset, 'focal': focal}
            # [新增] 将 Mask 加入返回字典，并调整维度 [H, W, 1] -> [1, H, W] (如果 ToTensor 不自动处理的话)
            # 这里的 ToTensor class 通常会把 (H,W,C) 转为 (C,H,W)，所以保持 (H,W,1) 即可
            sample = {'image': image, 'normal': normal, 'depth': depth_gt, 'mask_planar': mask_planar, 'offset': offset, 'focal': focal}
        
        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                gt_path = self.args.gt_path_eval
                depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
                if self.args.dataset == 'kitti':
                    depth_path = os.path.join(gt_path, sample_path.split()[0].split('/')[0], sample_path.split()[1])
                has_valid_depth = False
                try:
                    depth_gt = Image.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    if self.args.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                    else:
                        depth_gt = depth_gt / 256.0

            if self.args.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            
            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth}
            else:
                sample = {'image': image, 'focal': focal}
        
        if self.transform:
            sample = self.transform([sample, self.args.dataset])
        
        return sample
    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, normal,mask, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        normal = normal[y:y + height, x:x + width, :]
        # [新增]
        mask = mask[y:y + height, x:x + width, :]
        return img, depth, normal, mask, x
        # return img, depth, normal, x

    def train_preprocess(self, image, depth_gt, normal,mask):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
            normal = (normal[:, ::-1, :]).copy()
            # [新增]
            mask = (mask[:, ::-1, :]).copy()
    
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt, normal, mask
        # return image, depth_gt, normal
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def Cut_Flip(self, image, depth, normal,mask):

        p = random.random()
        if p < 0.5:
            return image, depth, normal,mask
        image_copy = copy.deepcopy(image)
        depth_copy = copy.deepcopy(depth)
        normal_copy = copy.deepcopy(normal)
        mask_copy = copy.deepcopy(mask) # [新增]


        h, w, c = image.shape
        N = 2     # split numbers
        h_list = []
        h_interval_list = []   # hight interval
        for i in range(N-1):
            h_list.append(random.randint(int(0.2*h), int(0.8*h)))
        h_list.append(h)
        h_list.append(0)  
        h_list.sort()
        h_list_inv = np.array([h]*(N+1))-np.array(h_list)
        for i in range(len(h_list)-1):
            h_interval_list.append(h_list[i+1]-h_list[i])
        for i in range(N):
            image[h_list[i]:h_list[i+1], :, :] = image_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i], :, :]
            depth[h_list[i]:h_list[i+1], :, :] = depth_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i], :, :]
            normal[h_list[i]:h_list[i+1], :, :] = normal_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i], :, :]
            # [新增]
            mask[h_list[i]:h_list[i+1], :, :] = mask_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i], :, :]

        return image, depth, normal, mask
        # return image, depth, normal

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample_dataset):
        sample = sample_dataset[0]
        dataset = sample_dataset[1]

        image, focal = sample['image'], sample['focal']
        H, W, _ = image.shape
        image = self.to_tensor(image)
        image = self.normalize(image)


        # -------------------------- 新增：读取并转换 mask_planar --------------------------
        mask_planar = None
        if 'mask_planar' in sample:  # 只在有这个键时处理（避免测试模式报错）
            mask_planar = sample['mask_planar']
            mask_planar = self.to_tensor(mask_planar)  # 和 depth/normal 用同样的转换逻辑
        # --------------------------------------------------------------------------------

        if dataset == 'kitti':
            K = np.array([[716.88 / 4.0, 0, 596.5593 / 4.0, 0],
                  [0, 716.88 / 4.0, 149.854 / 4.0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
            K_p = np.array([[716.88, 0, 596.5593, 0],
                  [0, 716.88, 149.854, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
            inv_K = np.linalg.pinv(K)
            inv_K_p = np.linalg.pinv(K_p)
            K = torch.from_numpy(K)
            inv_K = torch.from_numpy(inv_K)
            inv_K_p = torch.from_numpy(inv_K_p)
            
        elif dataset == 'nyu':
            K = np.array([[518.8579 / 4.0, 0, 325.5824 / 4.0, 0],
                  [0, 518.8579 / 4.0, 253.7362 / 4.0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
            K_p = np.array([[518.8579, 0, 325.5824, 0],
                  [0, 518.8579, 253.7362, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
            inv_K = np.linalg.pinv(K)
            inv_K_p = np.linalg.pinv(K_p)
            K = torch.from_numpy(K)
            inv_K = torch.from_numpy(inv_K)
            inv_K_p = torch.from_numpy(inv_K_p)
            
        if self.mode == 'test':
            return {'image': image, 'K': K, 'inv_K': inv_K, 'inv_K_p': inv_K_p, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            normal = sample['normal']
            normal = self.to_tensor(normal)
            if dataset == 'kitti':
                K = np.array([[716.88 / 4.0, 0, 596.5593 / 4.0, 0],
                    [0, 716.88 / 4.0, 149.854 / 4.0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=np.float32)
                K[0][2] -= sample['offset'] / 4.0
                K_p = np.array([[716.88, 0, 596.5593, 0],
                  [0, 716.88, 149.854, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
                K_p[0][2] -= sample['offset'] 
                inv_K = np.linalg.pinv(K)
                inv_K_p = np.linalg.pinv(K_p)

                K = torch.from_numpy(K)
                K_p = torch.from_numpy(K_p)
                inv_K = torch.from_numpy(inv_K)
                inv_K_p = torch.from_numpy(inv_K_p)
            elif dataset == 'nyu':
                K = np.array([[518.8579 / 4.0, 0, 325.5824 / 4.0, 0],
                    [0, 518.8579 / 4.0, 253.7362 / 4.0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=np.float32)
                K_p = np.array([[518.8579, 0, 325.5824, 0],
                  [0, 518.8579, 253.7362, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
                inv_K = np.linalg.pinv(K)
                inv_K_p = np.linalg.pinv(K_p)

                K = torch.from_numpy(K)
                K_p = torch.from_numpy(K_p)
                inv_K = torch.from_numpy(inv_K)
                inv_K_p = torch.from_numpy(inv_K_p)

            return {
                'image': image, 
                'depth': depth, 
                'normal': normal, 
                'mask_planar': mask_planar,  # 关键：补充这一行
                'K': K, 
                'inv_K': inv_K, 
                'inv_K_p': inv_K_p, 
                'focal': focal
            }
        # 修改
            # return {'image': image, 'depth': depth, 'normal': normal, 'K': K, 'inv_K': inv_K, 'inv_K_p': inv_K_p, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            result = {
                'image': image, 
                'depth': depth, 
                'K': K, 
                'inv_K': inv_K, 
                'inv_K_p': inv_K_p, 
                'focal': focal, 
                'has_valid_depth': has_valid_depth
            }
            if mask_planar is not None:
                result['mask_planar'] = mask_planar
            return result
            # return {'image': image, 'depth': depth, 'K': K, 'inv_K': inv_K, 'inv_K_p': inv_K_p, 'focal': focal, 'has_valid_depth': has_valid_depth}
    
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
