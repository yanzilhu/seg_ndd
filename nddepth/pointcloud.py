import os
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt


def arrays_to_pointcloud(rgb_array, depth_array, intrinsics, output_ply_path,
                         depth_scale=1.0, invert_depth=False, sample_step=1):
    """
    将 RGB (H,W,3) 和深度(H,W) 的 numpy arrays 转为 PLY 并保存。
    depth_array 可以是 uint16(mm) 或 float(m)
    返回保存路径
    """
    rgb = rgb_array.copy()
    depth = depth_array.copy().astype(np.float32)

    # 如果深度有多个通道，取第一个
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    h, w = depth.shape

    # depth scaling: 如果深度很大（比如以毫米保存），将其转为米
    if depth.max() > 1000.0:
        depth = depth * (depth_scale / 1000.0)
    else:
        depth = depth * depth_scale

    if invert_depth:
        with np.errstate(divide='ignore'):
            depth = 1.0 / depth

    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    v, u = np.indices((h, w))
    valid = (depth > 0) & np.isfinite(depth)

    if sample_step > 1:
        valid[::sample_step, ::sample_step] &= True

    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = depth[valid]

    X = (u_valid - cx) * z_valid / fx
    Y = (v_valid - cy) * z_valid / fy
    pts = np.stack([X, Y, z_valid], axis=-1)

    colors = rgb.reshape(-1, 3)[valid.ravel()]
    if colors.dtype != np.float32 and colors.max() > 1.0:
        colors = colors.astype(np.float32) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_dir = os.path.dirname(output_ply_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    o3d.io.write_point_cloud(output_ply_path, pcd)
    return output_ply_path


def files_to_pointcloud(rgb_path, depth_path, intrinsics, output_ply_path, depth_scale=1.0, invert_depth=False, sample_step=1):
    rgb = plt.imread(rgb_path)
    depth = plt.imread(depth_path)
    # ensure rgb is HxWx3
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]

    # resize depth to rgb
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    return arrays_to_pointcloud(rgb, depth, intrinsics, output_ply_path, depth_scale, invert_depth, sample_step)
