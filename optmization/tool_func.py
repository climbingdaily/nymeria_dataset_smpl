################################################################################
# File: /tool_func.py                                                          #
# Created Date: Tuesday July 26th 2022                                         #
# Author: climbingdaily                                                        #
# -----                                                                        #
# Modified By: the developer climbingdaily at yudidai@stu.xmu.edu.cn           #
# https://github.com/climbingdaily                                             #
# -----                                                                        #
# Copyright (c) 2022 yudidai                                                   #
# -----                                                                        #
# HISTORY:                                                                     #
################################################################################

import sys
import time
import os
import logging

import numpy as np
import torch
import json
import torchgeometry as tgm
import open3d as o3d
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.signal import savgol_filter, gaussian
from scipy.ndimage import convolve1d

sys.path.append('.')
sys.path.append('..')

from smpl import rotation_matrix_to_axis_angle
from utils import load_point_cloud, poses_to_vertices_torch

__all__ = ['find_stable_foot', 
           'crop_scene',
           'loadLogger',
           'load_scene_for_opt',
           'cal_global_trans',
           'get_lidar_to_head',
           'check_nan',
           'log_dict',
           'set_loss_dict',
           'set_foot_states',
           'compute_delta_rt',
           'compute_curvature',
           'fill_and_smooth_mano_poses',
           'mask_dict_to_tensor',
           'get_hand_pose',
           'get_hand_kpt',
           'fit_trajectory_with_confidence'
           ]

def read_json_file(file_name):
    """
    Reads a json file
    Args:
        file_name:
    Returns:
    """
    with open(file_name) as f:
        try:
            data = json.load(f)
        except:
            data = {}
    return data

def mask_dict_to_tensor(mask_dict, w, h, scale=0.5):
    mask = torch.zeros((len(mask_dict), 2, h, w)).type(torch.bool)
    for idx, (_, v) in enumerate(mask_dict.items()):
        if 1 in v:  # left hand
            mask[idx, 0] = torch.from_numpy(v[1])
        if 2 in v:  # right hand
            mask[idx, 1] = torch.from_numpy(v[2])
    return F.interpolate(mask.float(), scale_factor=scale, mode="nearest")

def get_hand_pose(mano_dict, lenght):
    hand_poses = torch.zeros((lenght, 2, 15, 3))
    for idx, (_, v) in enumerate(mano_dict.items()):
        for i, r in enumerate(v['right']):
            hp = rotation_matrix_to_axis_angle(torch.tensor(v['hand_pose'][i]))
            if r == 0:
                hp[:, 1:] *= -1
            hand_poses[idx][r] = hp

    hand_poses = hand_poses.reshape(-1, 30, 3)
    # hand_poses = fill_and_smooth_mano_poses(hand_poses)
    return hand_poses

def detect_outliers_by_velocity(x_seg, y_seg, ind_seg, window_size=5, sigma_threshold=2.0):
    """
    Detects outliers based on frame-aware velocity.
    
    Parameters:
    - x_seg, y_seg: x and y coordinates of the segment
    - ind_seg: Frame indices (not uniform)
    - window_size: Number of neighboring points for local statistics
    - sigma_threshold: Outlier threshold based on std dev

    Returns:
    - outlier_indices: Indices of detected outliers within this segment
    """
    num_points = len(x_seg)
    if num_points < 3:
        return []

    # Compute time-aware velocities
    delta_t = np.diff(ind_seg)  # Frame index differences
    delta_x = np.diff(x_seg)
    delta_y = np.diff(y_seg)
    velocity = np.sqrt(delta_x**2 + delta_y**2) / delta_t  # Velocity (Δdistance / Δtime)

    # Compute rolling mean and std deviation of velocity
    local_means = np.array([np.mean(velocity[max(0, i - window_size): i + 1]) for i in range(len(velocity))])
    local_stds = np.array([np.std(velocity[max(0, i - window_size): i + 1]) for i in range(len(velocity))])

    # Outlier detection: velocity > (mean + threshold * std)
    outlier_mask = velocity > (local_means + sigma_threshold * local_stds)

    # Shift because np.diff affects next point
    outlier_indices = np.where(outlier_mask)[0] + 1  

    return outlier_indices

def fit_trajectory_with_confidence(points, confidence_threshold=0.5, smoothing=10, max_gap=30, alpha=0.1, enable_filling=True, is_show=False):
    """
    Fit a 2D trajectory using B-spline smoothing with confidence-based filtering and adaptive smoothing.

    Parameters:
    - points: (N, 3) NumPy array containing (x, y, confidence)
    - confidence_threshold: Points with confidence below this value are ignored (default: 0.5)
    - base_smoothing: Base smoothing factor (default: 5)
    - max_gap: Maximum allowed gap between frames before splitting (default: 30 frames)
    - alpha: Scaling factor for smoothing adjustment (default: 0.1)

    Returns:
    - all_x_smooth, all_y_smooth: List of smoothed trajectory segments
    - processed_data: (N, 3) array of retained points
    """

    num_points = len(points)
    indices = np.arange(num_points)  # Frame indices as time

    # Step 1: Filter out low-confidence points
    valid_mask = (points[:, 2] > confidence_threshold) & (points[:, 1] > 0) & (points[:, 0] > 0) 
    valid_data = points[valid_mask]
    valid_indices = indices[valid_mask]

    if len(valid_data) < 2:
        raise ValueError("Not enough valid points for B-spline fitting!")

    x, y = valid_data[:, 0], valid_data[:, 1]

    processed_data = points.copy()

    # Step 2: Find large gaps and split into segments
    gaps = np.diff(valid_indices)  # Compute gaps between consecutive valid frames
    split_indices = np.where(gaps > max_gap)[0] + 1  # Find large gaps

    ind_segments = np.split(valid_indices, split_indices)
    x_segments = np.split(x, split_indices)
    y_segments = np.split(y, split_indices)

    fitted_indices = []

    if is_show:
        plt.figure(figsize=(10, 10))
        plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=30, cmap='coolwarm', edgecolors='k', marker='x', label="Original Points")
    # Step 3: Fit B-spline to each segment
    for _, (x_seg, y_seg, ind_seg) in enumerate(zip(x_segments, y_segments, ind_segments)):
        start, end = (ind_seg[0], ind_seg[-1])
        ids = np.arange(start, end+1)
        outliers = detect_outliers_by_velocity(x_seg, y_seg, ind_seg)
        nonoutlier = np.setdiff1d(np.arange(len(x_seg)), outliers)

        outliers = np.setdiff1d(ids, ind_seg[nonoutlier])  # Keep only valid points

        if len(ind_seg) <= 4:
            continue  # Skip tiny segments
        # Compute average distance between consecutive points
        avg_distance = np.mean(np.sqrt(np.diff(x_seg)**2 + np.diff(y_seg)**2))
        
        # Adjust smoothing dynamically
        adaptive_s = smoothing * (1 + avg_distance)  # Higher avg_distance → lower s, and vice versa
        
        tck, _ = splprep([x_seg[nonoutlier], y_seg[nonoutlier]], u=ind_seg[nonoutlier], s=adaptive_s)

        # Generate smooth trajectory for this segment
        if len(outliers) > 0:
            x_smooth, y_smooth = splev(outliers, tck)
            processed_data[outliers, 2] = 0

            if enable_filling:
                valid_fit = (x_smooth > 0) & (y_smooth > 0)
                processed_data[outliers[valid_fit], 0] = x_smooth[valid_fit]
                processed_data[outliers[valid_fit], 1] = y_smooth[valid_fit]
                processed_data[outliers[valid_fit], 2] = 0.51
                fitted_indices.append(outliers[valid_fit])

            if is_show:
                plt.scatter(processed_data[indices, 0], processed_data[indices, 1], 
                            c=processed_data[indices, 2], cmap='coolwarm', edgecolors='k', marker='o', 
                            label="Processed Points (Retained + Interpolated)")

                # Draw lines connecting original to processed points
                for i in outliers[valid_fit]:
                    plt.plot([points[i, 0], processed_data[i, 0]],  # X-coordinates
                            [points[i, 1], processed_data[i, 1]],  # Y-coordinates
                            'gray', linestyle='--', alpha=0.6)   # Dashed gray lines for visibility
                    
                plt.plot(processed_data[start:end, 0], processed_data[start:end, 1], 'r--', label=f"B-Spline Fit")
                
    if is_show:
        plt.gca().invert_yaxis()  

        plt.colorbar(label="Confidence")
        plt.legend()
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.title("Trajectory Fitting with Index-Based Interpolation")
        plt.show()
    return processed_data, fitted_indices

def get_hand_kpt(keypoints, window_size=5, threshold=0.5, enable_filling=False, is_show=False):
    keypoint_names = ['Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist']
    coco_elbow = [7, 8]
    coco_wrist = [9, 10]
    coco_hand_0 = [91, 112]
    # coco_left = [92, 93, 94, 96, 97, 98, 100, 101, 102, 104, 105, 106, 108, 109, 110]
    # coco_right = [113, 114, 115, 117, 118, 119, 121, 122, 123, 125, 126, 127, 129, 130, 131]
    coco_left = [92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106]
    coco_right = [113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
    keypoint_ids = coco_elbow + coco_wrist + coco_hand_0 + coco_left + coco_right
    # 选取手肘和手腕关键点
    selected_kpts = keypoints[:, keypoint_ids, :].astype(float)
    
    # 过滤置信度低的关键点，将 x, y 设为 -1，置信度设为 0
    # mask = selected_kpts[:, :, 2] <= threshold
    # selected_kpts[mask] = [-1, -1, 0]
    
    # 统计每个关键点被拟合的次数
    num_filled_per_kpt = np.zeros(4, dtype=int)

    smoothed_kpts = np.copy(selected_kpts)
    for i in range(smoothed_kpts.shape[1]):
        smoothed_kpts[:, i], _ = fit_trajectory_with_confidence(smoothed_kpts[:, i], 0.5, max_gap=10, enable_filling=enable_filling, is_show=is_show)
    
    if is_show:
        # 可视化关键点轨迹，分别绘制原始和优化后的数据
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        for i in range(smoothed_kpts.shape[1]):
            original_points = selected_kpts[:, i, :2]
            smoothed_points = smoothed_kpts[:, i, :2]
            
            original_valid = original_points[(original_points>0).all(axis=1)]
            smoothed_valid = smoothed_points[(smoothed_points>0).all(axis=1)]
            
            axes[0].plot(original_valid[:, 0], original_valid[:, 1], linestyle='dashed', label=f'{keypoint_names[i]} (Original)')
            axes[1].plot(smoothed_valid[:, 0], smoothed_valid[:, 1], linestyle='dashed', label=f'{keypoint_names[i]} (Smoothed)')
        
        for ax, title in zip(axes, ['Original Keypoints', 'Smoothed Keypoints']):
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.legend()
            ax.set_title(title)
            ax.axis('equal')  # 保持画布1:1比例
        
        plt.suptitle(f'Keypoint Trajectories (Filled Points per Keypoint: {num_filled_per_kpt})')
        plt.show()
    
    return smoothed_kpts, num_filled_per_kpt

def fill_and_smooth_mano_poses(poses, missing_threshold=0.5, window_size=5, poly_order=2, sigma=1.0, z_thresh=3.0):
    """
    Perform outlier detection, interpolation, and smoothing on mano pose data.
    
    :param poses: np.array of shape (n, 30, 3), mano pose data
    :param missing_threshold: If missing frames exceed this ratio, use linear interpolation
    :param window_size: Window size for Savitzky-Golay filter
    :param poly_order: Polynomial order for Savitzky-Golay filter
    :param sigma: Standard deviation for Gaussian filter
    :param z_thresh: Z-score threshold for detecting outliers
    :return: Smoothed pose data
    """
    poses = np.array(poses)  # Ensure numpy array
    n_frames, n_joints, dims = poses.shape
    
    # 1. Detect missing frames (all-zero frames) and replace them with NaN
    zero_mask = (poses == 0).all(axis=(1, 2))
    poses[zero_mask] = np.nan
    
    # 2. Detect outliers
    for j in range(n_joints):
        for d in range(dims):
            data = poses[:, j, d]
            
            # Compute Z-score to detect outliers
            mean, std = np.nanmean(data), np.nanstd(data)
            z_scores = np.abs((data - mean) / std)
            outliers = z_scores > z_thresh
            
            # Set outliers to NaN for interpolation
            poses[outliers, j, d] = np.nan
    
    # 3. Perform interpolation to fill missing values
    for j in range(n_joints):
        for d in range(dims):
            valid_idx = np.where(~np.isnan(poses[:, j, d]))[0]
            invalid_idx = np.where(np.isnan(poses[:, j, d]))[0]
            
            if len(valid_idx) == 0:
                continue  # Skip if all values are missing
            
            if len(valid_idx) < missing_threshold * n_frames:
                interp_func = CubicSpline(valid_idx, poses[valid_idx, j, d], extrapolate=True)
                poses[invalid_idx, j, d] = interp_func(invalid_idx)
            else:
                poses[invalid_idx, j, d] = np.interp(invalid_idx, valid_idx, poses[valid_idx, j, d])  # Use linear interpolation
    
    # 4. Apply Savitzky-Golay filter
    poses = savgol_filter(poses, window_length=window_size, polyorder=poly_order, axis=0, mode='nearest')
    
    # 5. Further smoothing using Gaussian filter
    gauss_kernel = gaussian(window_size, sigma)
    gauss_kernel /= gauss_kernel.sum()  # Normalize kernel
    for j in range(n_joints):
        for d in range(dims):
            poses[:, j, d] = convolve1d(poses[:, j, d], gauss_kernel, mode='nearest')
    
    return poses


def find_stable_foot(smpl_verts, translations, pre_smpl = None, frame_time=0.05, thresh_vel = 5):
    """
    It takes in the SMPL vertices and the translations, and returns the stable foot (left or right) and
    the foot movement
    
    Args:
      smpl_verts: the vertices of the SMPL model
      translations: the translation of the SMPL model
      pre_smpl: the previous smpl vertices, if any
    """
    contact_verts = load_vertices()
    # back['verts']
    if pre_smpl is not None:
        smpl_verts = torch.cat((pre_smpl, pre_smpl), dim = 0)
    left_heels = smpl_verts[:, contact_verts['left_heel']]
    left_toes = smpl_verts[:, contact_verts['left_toe']]
    right_heels = smpl_verts[:, contact_verts['right_heel']]
    right_toes = smpl_verts[:, contact_verts['right_toe']]

    root_moves = (translations[1:] - translations[:-1]).norm(dim=1)

    lh_move = (left_heels[1:] - left_heels[:-1]).mean(dim=1).norm(dim=1)
    lt_move = (left_toes[1:] - left_toes[:-1]).mean(dim=1).norm(dim=1)
    rh_move = (right_heels[1:] - right_heels[:-1]).mean(dim=1).norm(dim=1)
    rt_move = (right_toes[1:] - right_toes[:-1]).mean(dim=1).norm(dim=1)

    left_foot = torch.cat((left_heels, left_toes), dim=1) 
    right_foot = torch.cat((right_heels, right_toes), dim=1)
    
    left_foot_moves = (left_foot[1:] - left_foot[:-1]).mean(dim=1).norm(dim=1)
    right_foot_moves = (right_foot[1:] - right_foot[:-1]).mean(dim=1).norm(dim=1)

    lhp = left_heels.mean(dim=1)[:,2]
    ltp = left_toes.mean(dim=1)[:,2]
    rhp = right_heels.mean(dim=1)[:,2]
    rtp = right_toes.mean(dim=1)[:,2]
    lhp -= lhp[0].item()
    ltp -= ltp[0].item()
    rhp -= rhp[0].item()
    rtp -= rtp[0].item()
    
    states = []
    # If the foot velocity < 0.1 m/s, set as stable
    max_move_dist = 0.1 * frame_time

    for left_move, right_move, hip_move in zip(left_foot_moves, right_foot_moves, root_moves):
        if hip_move <= max_move_dist or (right_move <= max_move_dist and left_move <= max_move_dist):
            states.append(0)
        elif left_move <= max_move_dist:
            states.append(-1) # left foot stable
        elif right_move <= max_move_dist:
            states.append(1)   # right foot stable
        elif right_move < frame_time * thresh_vel / 2 and left_move < frame_time * thresh_vel / 2:
            if right_move < left_move :
                states.append(1)
            else:
                states.append(-1)
        else:
            states.append(-2) # bad case (foot sliding)
    states = np.asarray(states, dtype=np.int32)

    while True:
        count = 0
        for i in range(2, states.shape[0]-2):
            if states[i-2] == states[i-1] and states[i+1] == states[i+2] and states[i] != states[i+1] and states[i] != states[i-1]:
                states[i] = states[i-1]
                count += 1
        if count == 0:
            break

    while True:
        count = 0
        for i in range(2, states.shape[0]-2):
            if states[i] != states[i+1] and states[i] != states[i-1]:
                states[i] = states[i-1]
                count += 1
        if count == 0:
            break

    if pre_smpl is None:
        states = np.concatenate((np.asarray(states[:1]), states))
        lfoot_move = torch.cat((left_foot_moves[:1].clone(), left_foot_moves))
        rfoot_move = torch.cat((right_foot_moves[:1].clone(), right_foot_moves))
        lh_move = torch.cat((lh_move[:1].clone(), lh_move))
        lt_move = torch.cat((lt_move[:1].clone(), lt_move))
        rh_move = torch.cat((rh_move[:1].clone(), rh_move))
        rt_move = torch.cat((rt_move[:1].clone(), rt_move))
    lfoot_move = lfoot_move.detach().cpu().numpy()
    rfoot_move = rfoot_move.detach().cpu().numpy()
    lh_move = lh_move.detach().cpu().numpy()
    lt_move = lt_move.detach().cpu().numpy()
    rh_move = rh_move.detach().cpu().numpy()
    rt_move = rt_move.detach().cpu().numpy()
    
    return states, lfoot_move, rfoot_move


def crop_scene(scene_pcd, positions, radius=1):
    """
    It takes a point cloud and a list of positions, and returns a point cloud that contains only the
    points that are within a certain radius of the positions
    
    Args:
      scene_pcd: the point cloud of the scene
      positions: the trajectory of the robot, in the form of a list of 3D points.
      radius: the radius of the sphere around the trajectory that we want to crop. Defaults to 1
    
    Returns:
      a point cloud of the ground points.
    """
    trajectory = o3d.geometry.PointCloud()
    scene = o3d.geometry.PointCloud()
    scene.points = scene_pcd.vertices
    trajectory.points = o3d.utility.Vector3dVector(positions - np.array([0 , 0, 0.8]))
    dist = np.asarray(scene.compute_point_cloud_distance(trajectory))
    valid_list = np.arange(len(scene_pcd.vertices))[dist < radius]
    ground = scene_pcd.select_by_index(sorted(valid_list))
    return ground

def load_vertices():
    """
    It loads the vertices of the contact points of the foot
    
    Returns:
      The vertices of the contact points of the feet and the back.
    """
    root_path = "/".join(os.path.abspath(__file__).split('/')[:-1])
    all_vertices = read_json_file(os.path.join(root_path, 'vertices', 'all_new.json'))

    back = all_vertices['back_new']

    contact_verts = {}
    for part in ['right_toe', 'left_toe', 'left_heel', 'right_heel']:
        contact_verts[part] = np.array(
            all_vertices[part]['verts'], dtype=np.int32)

    contact_verts['back'] = np.array(
        all_vertices['back_new']['verts'], dtype=np.int32)
    return contact_verts


def loadLogger(work_dir, is_save=True, work_file=None, name=''):
    """
    It creates a logger that prints to the console and saves to a file
    
    Args:
      work_dir: the directory where the log file will be saved.
      is_save: whether to save the log file. Defaults to True
    
    Returns:
      A logger object and a logger time.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                               datefmt="%H:%M:%S")
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    sHandler.setLevel(logging.INFO)
    logger.addHandler(sHandler)

    logger_time = 0
    if is_save:
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        logger_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        work_file = os.path.join(
            work_dir, f'{logger_time}_{name}.log') if work_file is None else work_file
        fHandler = logging.FileHandler(work_file, mode='a')
        fHandler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                datefmt="%a %b %d %H:%M:%S %Y")
        fHandler.setFormatter(formatter)

        logger.addHandler(fHandler)

    else:
        # 禁用所有终端日志处理器的输出
        sHandler.setLevel(logging.CRITICAL)
        
    return logger, logger_time, work_file


def load_scene_for_opt(scene):
    """
    It loads a point cloud from a file, estimates normals if they don't exist, and returns the point
    cloud and a kdtree
    
    Args:
    scene: the path to the scene point cloud
    
    Returns:
    geometry, kdtree
    """
    geometry = load_point_cloud(scene)
    print(f'[Data loading]: Load scene in {scene}')
    # kdtree = o3d.geometry.KDTreeFlann(geometry)
    if geometry.get_geometry_type() == geometry.PointCloud and not geometry.has_normals():
            geometry.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=40))
            normals_file = scene.replace('.pcd', '_normal.pcd')
            o3d.io.write_point_cloud(normals_file, geometry)
            print('Save scene normals in: ', normals_file)
    elif geometry.get_geometry_type() == geometry.TriangleMesh and not geometry.has_vertex_normals():
            geometry.compute_vertex_normals()
            normals_file = scene.replace('.', '_normal.')
            o3d.io.write_point_cloud(normals_file, geometry)
            print('Save scene normals in: ', normals_file)
    print(geometry)
    return geometry

def cal_global_trans(pose, trans, is_cuda=False):
    """
    It calculates the distance between the feet of the first and last frame of the animation
    
    Args:
      pose: the pose of the character, in the form of a tensor of shape (num_frames, 24, 3)
      trans: the translation vector for the model.
      is_cuda: whether to use GPU or not. Defaults to False
    
    Returns:
      The distance between the feet of the first and last frame of the sequence.
    """
    def get_feetcenter(p, s):
        with torch.no_grad():
            verts, joints, _ = poses_to_vertices_torch(p, s, 1024, is_cuda=is_cuda)
        feet_center     = (joints[:,7].mean(dim=0) + joints[:,8].mean(dim=0))/2
        feet_center[2]  = verts[..., 2].min()
        return feet_center

    start_pos = get_feetcenter(pose[:5], trans[:5])
    end_pos = get_feetcenter(pose[-5:], trans[-5:])
    dist = torch.norm(end_pos - start_pos) * 1000
    return dist.item()


def get_lidar_to_head(joints, rots, lidar_traj, lowest_height=None, revert=True):
    lidar2head = (joints - lidar_traj)
    if lowest_height is not None:
        lidar2head[:, -1] -= lowest_height
    # rr = torch.tensor(MOCAP_INIT).float().to(rots.device)
    # global_head_rot = rots @ rr
    global_head_rot = rots
    if revert:
        global_head_rot = global_head_rot.transpose(2, 1)
    lidar2head = torch.einsum('bij, bj->bi', global_head_rot, lidar2head)
    return lidar2head

def check_nan(**args):
    has_nan = False
    for key in args:
        if torch.any(torch.isnan(args[key])):
            print(f'\n{key} nan')
            has_nan = True
    return has_nan

def log_dict(logger, loss_dict, transA=None, transB=None, rotsA=None, rotsB=None):
    """
    It logs all losses
    
    Args:
      logger: a logger object
      loss_dict: a dictionary of losses
    """
    # todo: W&B
    for prefix in loss_dict:
        if type(loss_dict[prefix]) is not dict:
            continue

        sum_num, sum_loss, sum_loss_no_opt = 0, 0, 0
        loss, cat_loss, cat_loss_no_opt = 0, 0, 0
        num = 0
        loss_no_opt = 0
        
        for i in loss_dict[prefix]:
            try:
                num = loss_dict[prefix][i]['num']
                loss = loss_dict[prefix][i]['loss'] * num
                loss_no_opt = loss_dict[prefix][i]['loss_no_opt'] * num

                if 'concat_loss' in loss_dict[prefix][i]:
                    cat_loss = loss_dict[prefix][i]['concat_loss']
                    cat_loss_no_opt = loss_dict[prefix][i]['concat_loss_no_opt']
                
                sum_loss += loss + cat_loss
                sum_loss_no_opt += loss_no_opt + cat_loss_no_opt
                sum_num += num + 1

            except Exception as e:
                pass
                # logger.error(f'{prefix} {i} {e.args[0]}')

        logger.info(
            f'{prefix} loss w/o_opt(total) w/o_opt(batch):\t{sum_loss/(sum_num+0.1):.2f} / {sum_loss_no_opt/(sum_num+0.1):2.2f} {(loss+cat_loss)/(num+1):.2f} / {(loss_no_opt+cat_loss_no_opt)/(num+1):2.2f}')
    
    

    try:
        delta_rot, degrees = compute_delta_rt(rotsA, rotsB)
    except Exception as e:
        delta_rot, degrees = 0, 0
        print(e.args[0])
        
    delta_trans = (transA - transB)[-5:].clone().detach().mean(axis=0)

    if torch.any(torch.isnan(delta_trans)):
        delta_trans = torch.zeros_like(delta_trans)

    logger.info(f'Delta trans / rot: [{delta_trans[0].item() * 100:.2f} {delta_trans[1].item() * 100:.2f}, {delta_trans[2].item() * 100:.2f}] cm / {degrees:.1f} deg.')

    return delta_trans, delta_rot
                    

def set_loss_dict(loss_list, weight, loss_dict, indexes, num_list=1, prefix='sliding', concat=False):
    """
    > This function is used to set the loss dictionary for the sliding window loss
    
    Args:
      loss_list: the list of loss value
      weight: the weight of the loss
      loss_dict: a dictionary that stores the loss values for each window
      index: the index of the window
      num_list: number of sliding windows. Defaults to 1
      prefix: the prefix of the loss, e.g. 'sliding'. Defaults to sliding
      concat: whether to concatenate the loss with the previous loss. Defaults to False
    """

    weight_loss, str = 0, ''

    scale = 180 / np.pi if prefix == 'rot' or prefix == 'pose' else 100

    index, iters = indexes

    if prefix not in loss_dict: 
        loss_dict[prefix] = {}
    if index not in loss_dict[prefix]: 
        loss_dict[prefix][index] = {}

    target = loss_dict[prefix][index]

    if loss_list is not None and len(loss_list) > 0:
        loss, num = 0, 0
        for l in loss_list:
            if l:
                if prefix in ['sliding', 'contact', 'm2p']:
                # if prefix in ['m2p']:
                    l = l * l.item() * 100 
                # if l > 0.03 and prefix in ['trans']:
                #     l = l * l.item() * 100
                loss += l
                num += 1
                
        if num > 0:
            loss /= num

            print_loss = torch.as_tensor(loss_list)[torch.as_tensor(loss_list) > 0].mean().item() * scale

            if concat:
                if 'concat_loss' not in target:
                    target['concat_loss_no_opt'] = print_loss

                target['concat_loss'] = print_loss
                str = f'c{prefix} {print_loss:.3f} '
            else:
                if 'loss' not in target:
                    target['num'] = num
                    target['loss_no_opt'] = print_loss
                target['loss'] = print_loss
                str = f'{prefix} {print_loss:.3f} '
            
            weight_loss = weight * loss

    return weight_loss, str


def set_foot_states(mocap_rots, mocap_trans, frame_time, betas, gender='male'):
    
    if type(mocap_rots) != torch.Tensor:
        mocap_rots = torch.from_numpy(mocap_rots).type(torch.FloatTensor)
    if type(mocap_trans) != torch.Tensor:
        mocap_trans = torch.from_numpy(mocap_trans).type(torch.FloatTensor)

    smpl_verts, _, _ = poses_to_vertices_torch(
        mocap_rots, batch_size=1024, trans=mocap_trans, betas=betas, gender=gender)
    
    foot_states, lfoot_move, rfoot_move = find_stable_foot(
        smpl_verts.detach().cpu(), mocap_trans, frame_time=frame_time)
    
    return foot_states, lfoot_move, rfoot_move


def compute_delta_rt(rotsA, rotsB, last_num=5):
    """
    It computes the rotation matrix that transforms the last 5 frames of the first sequence to the last
    5 frames of the second sequence
    
    Args:
      rotsA: the ground truth rotation matrices
      rotsB: the ground truth rotation
      last_num: the number of frames to use for computing the delta rotation. Defaults to 5
    
    Returns:
      The rotation matrix and the angle in degrees
    """

    from scipy.spatial.transform import Rotation as R
    from losses import compute_similarity_transform_torch

    if len(rotsA.shape) == 2:
        r1 = tgm.angle_axis_to_rotation_matrix(rotsA)[-last_num:, :3, :3]
    else:
        r1 = rotsA
    if len(rotsB.shape) == 2:
        r2 = tgm.angle_axis_to_rotation_matrix(rotsB)[-last_num:, :3, :3]
    else:
        r2 = rotsB

    r1t = torch.transpose(r1, 2, 1).detach().cpu()
    r2t = torch.transpose(r2, 2, 1).detach().cpu()

    delta_rot, _, _ = compute_similarity_transform_torch(r1t.reshape(-1, 3), r2t.reshape(-1, 3))

    delta_degree = np.linalg.norm(R.from_matrix(delta_rot).as_rotvec()) * 180 / np.pi

    return delta_rot, delta_degree

def compute_curvature(points):
    """
    计算轨迹点的曲率 (By @chatgpt)
    Args:
        points (torch.Tensor): n x 3 的轨迹点坐标，每行表示一个点的XYZ坐标
    Returns:
        curvature (torch.Tensor): n x 1 的曲率，每个元素表示对应点的曲率值
    """
    # 计算相邻点之间的向量
    v1 = points[:-2] - points[1:-1]
    v2 = points[2:] - points[1:-1]
    
    # 计算三角形的法向量，使用叉积
    normals = torch.cross(v1, v2, dim=1)
    
    # 计算三角形的面积
    areas = 0.5 * torch.norm(normals, dim=1)
    
    # 计算相邻点之间的距离
    lengths = torch.norm(points[:-1] - points[1:], dim=1)
    
    # 计算曲率
    curvature = 2 * areas / (lengths[:-1] ** 3)
    
    # 将首尾两个点的曲率设为0
    curvature = torch.cat((torch.tensor([0.]), curvature, torch.tensor([0.])))
    
    return curvature
