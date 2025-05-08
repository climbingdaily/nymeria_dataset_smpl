################################################################################
# File: /losses.py                                                             #
# Created Date: Wednesday July 27th 2022                                       #
# Author: climbingdaily                                                        #
# -----                                                                        #
# Modified By: the developer climbingdaily at yudidai@stu.xmu.edu.cn           #
# https://github.com/climbingdaily                                             #
# -----                                                                        #
# Copyright (c) 2022 yudidai                                                   #
# -----                                                                        #
# HISTORY:                                                                     #
################################################################################

"""
Defines losses used in the optimization
"""
import os
import sys

import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch.nn import functional as F

from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.metrics.render import mask_iou
from kaolin.ops.mesh import index_vertices_by_faces, face_normals
from kaolin.render.mesh import rasterize, camera

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (    
    RasterizationSettings,    
    MeshRasterizer,    
    MeshRenderer,
    SoftSilhouetteShader,
    SoftGouraudShader,
    HardFlatShader,
    MeshRendererWithFragments,
    PerspectiveCameras,)

sys.path.append(os.path.dirname(os.path.split(os.path.abspath( __file__))[0]))

import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as cham
# import ChamferDistancePytorch.chamfer2D.dist_chamfer_2D as cham2D
distChamfer = cham.chamfer_3DDist()
# cham2d = cham2D.chamfer_2DDist()

from utils import read_json_file
from smpl import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle, BODY_PARTS as BP

root_path = "/".join(os.path.abspath(__file__).split('/')[:-1])

CONTACT_VERTS = None

# list all losses func with __all__ set to
__all__ = ['trans_imu_smooth', 
           'joints_smooth', 
           'compute_similarity_transform_torch',
           'contact_constraint', 
           'foot_collision', 
           'sliding_constraint', 
           'get_optmizer', 
           'get_contacinfo', 
           'mesh2point_loss', 
           'points2smpl_loss', 
           'collision_loss', 
           'load_vertices',
           'cam_loss',
           'reprojection_loss',
           'create_distance_mask']

def create_distance_mask(h=1024, w=1024, r=1.174):
    cx, cy = w // 2, h // 2  # 图像的中心点
    r = r * w / 2
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)  # 计算到中心的距离
            if distance >= r:
                mask[y, x] = 255  # 距离大于 r 的区域设为 255

    return mask

def chamfer_distance_x2y(x, y):
    """
    Args:
        x: Tensor of shape (B, N, D).
        y: Tensor of shape (B, M, D).
    """
    # Compute pairwise distances between each point in x and y
    distances = torch.sum((x[:, :, None] - y[:, None]) ** 2, dim=-1)

    # Find the nearest neighbor in y for each point in x
    min_distances, _ = distances.min(dim=-1)

    # # Find the nearest neighbor in x for each point in y
    min_distances2, _ = distances.min(dim=1)

    return min_distances[0], min_distances2[0]

def loss_filter(loss, a=50):
    loss = 1 - 1 / (a * loss + 1) 
    return loss

def iou_loss(points_coord, mask_coord, min_pixel_dist = 1):
    """
    The function calculates the intersection over union (IOU) loss between two sets of points and masks.
    
    Args:
      points_coord: A tensor containing the coordinates of points in the predicted mask. size must be (1, n1, 2)
      mask_coord: The coordinates of the mask, which is typically a binary image indicating the region
    of interest.
      min_pixel_dist: The minimum distance (in pixels) between a point and a mask for them to be
    considered as intersecting. If the distance is less than this value, they are considered as
    non-intersecting. Defaults to 1
    
    Returns:
      two values: iou_loss1 and iou_loss2.
    """
    p2m, m2p, idx1, idx2 = cham2d(points_coord, mask_coord)

    # p2m, m2p = chamfer_distance_x2y(points_coord, mask_coord)

    non_inter_points = loss_filter(torch.relu(p2m - (min_pixel_dist ** 2 + 1e-4))).sum()
    non_inter_mask = loss_filter(torch.relu(m2p - (2 ** 2 + 1e-4))).sum()

    iou_loss1 = non_inter_points / len(p2m[0])
    iou_loss2 = non_inter_mask / len(m2p[0])
    loss = (non_inter_mask + non_inter_points) / (non_inter_points + len(m2p[0]))
    return iou_loss1, iou_loss2, loss

def camera_to_pixel(cam_in, X):
    f = cam_in[:2]
    c = cam_in[2:]
    XX = X[..., :2] / (X[..., 2:])
    return f * XX + c

def filter_points(points, min_angle_deg=20, min_distance=0.2, max_distance=80):
    """
    Filter a 3D point cloud based on the angle between the points and the XY plane of the camera coordinate system,
    as well as the minimum and maximum distance from the camera.

    Args:
    - points: a numpy array of shape (N, 3) containing the 3D points in camera coordinates to be filtered
    - min_angle_deg: the minimum angle between a point and the XY plane of the camera coordinate system, in degrees
    - min_distance: the minimum distance between a point and the camera origin, in meters
    - max_distance: the maximum distance between a point and the camera origin, in meters

    Returns:
    - A filtered 3D point cloud as a numpy array of shape (N', 3)
    """

    # Calculate the distance and angle between each point and the XY plane
    points = points[points[:, 2] > 0]
    distance = torch.norm(points, dim=1)
    xy_norm = torch.norm(points[:, :2], dim=1)
    angle = torch.arccos(xy_norm / distance) * 180 / 3.141592653

    # Filter points based on the angle, minimum distance, and maximum distance criteria
    mask = (angle > min_angle_deg) & (distance > min_distance) & (distance < max_distance)
    filtered_points = points[mask]

    return filtered_points

def opencv_extrinsic_to_lookat(R, t):
    """将 OpenCV 的外参 (R, t) 转换为 CG 视角的 eye, at, up"""
    eye = -R.T @ t  # 计算相机在世界坐标中的位置
    at = eye + R.T @ torch.tensor([0, 0, -1])  # 计算相机看向的目标点
    up = R.T @ torch.tensor([0, 1, 0])  # 计算相机的上方向
    return eye.flatten(), at.flatten(), up.flatten()

def intrinsics_to_fov(K, width, height):
    """从 OpenCV 内参矩阵 K 计算水平和垂直 FOV"""
    fx, fy = K[0, 0], K[1, 1]
    fov_x = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi  # 转换为度数
    fov_y = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi
    return fov_x, fov_y

def project_point_cloud_to_image(points, extrnsics, intrinsic, w=1024, h=1024):
    R = extrnsics[:3, :3].astype(np.float32) # (3, 3)
    T = extrnsics[:3, 3:].astype(np.float32) # (3,3)

    camera_coordinates = R.dot(points.T) + T    # (3, n)

    z = camera_coordinates[2, :]
    valid_points = z > 0 

    uv = intrinsic @ camera_coordinates[:, valid_points]
    u, v = uv[0, :] / uv[2, :], uv[1, :] / uv[2, :]

    valid_uv = (u > 0) & (u < w) & (v > 0) & (v < h)

    return u[valid_uv], v[valid_uv], z[valid_points][valid_uv]

def dice_loss(pred_maks, gt_maks, eps=1e-6, beta=1):
    """
    The function calculates the dice loss between the predicted masks and the ground truth masks.
    
    Args:
      pred_maks: A tensor of shape (B, H, W) representing the predicted masks.
      gt_maks: A tensor of shape (B, H, W) representing the ground truth masks.
      eps: A small constant to avoid division by zero. Defaults to 1e-6
    
    Returns:
      A scalar representing the dice loss.
    """
    intersection = torch.sum(pred_maks * gt_maks, dim=(1, 2))
    union = torch.sum(beta * pred_maks, dim=(1, 2)) + torch.sum(gt_maks, dim=(1, 2))
    dice_loss = 1 - ((1+beta) * intersection + eps) / (union + eps)
    return torch.mean(dice_loss)

def reprojection_loss(points, gt_points, cameras:camera, min_margin=5, max_margin=500, thresh_ratio=0.5):
    pred_points = cameras.transform_points_screen(points)  # (B, N, 3)
    ratio = gt_points[..., 2]

    # Compute Smooth L1 loss per point (B, N, 2)
    pointwise_loss = torch.nn.functional.smooth_l1_loss(pred_points[..., :2], gt_points[..., :2], reduction='none')  # (B, N, 2)
    pointwise_loss = pointwise_loss.mean(dim=-1)  # (B, N)
    if gt_points.shape[-1] == 3:
        pointwise_loss = pointwise_loss * ratio

    # Create mask to ignore invalid keypoints
    valid_pred_mask = pred_points[..., -1] > 0  # (B, N) 
    valid_gt_mask = (gt_points > 0).all(dim=-1)  # (B, N) -> True for valid keypoints
    valid_ratio_mask = ratio > thresh_ratio  # filter ratio < 0.6 的点
    valid_mask = (valid_gt_mask & valid_pred_mask) & valid_ratio_mask & (pointwise_loss > min_margin) & (pointwise_loss < max_margin)

    # Get valid batch indices (samples with at least one valid keypoint)
    loss_indices = valid_mask.any(dim=-1).nonzero(as_tuple=True)[0].tolist()  # List of valid batch indices

    # Mask invalid points (set their loss to 0)
    pointwise_loss = pointwise_loss * valid_mask  # (B, N)

    # Compute mean loss per valid sample (B,)
    per_sample_loss = pointwise_loss.sum(dim=-1) / (valid_mask.sum(dim=-1) + 1e-6)  # Avoid division by zero

    # l2 loss
    # loss = torch.norm(pred_points - gt_points, dim=-1)  # (N,)
    # loss = (loss ** 2).mean()
    # smooth l1 loss
    # loss = torch.abs(pred_points - gt_points).mean()

    return per_sample_loss[loss_indices], loss_indices

def false_pos_neg_loss(pred, target, alpha=0.5):
    false_pos = (pred - target).clamp(min=0).mean()  # 预测多余的部分
    false_neg = (target - pred).clamp(min=0).mean()  # 预测少的部分
    return alpha * false_pos + (1 - alpha) * false_neg
def unmatched_area_loss(pred, target):
    union = pred + target  # A ∪ B
    intersection = pred * target  # A ∩ B
    diff = union - 2 * intersection  # 计算 (A ∪ B) - (A ∩ B)
    return diff.sum()  / (target.sum() + 100)
def cam_loss(mask, vertices, faces, cameras:camera, bs=1, face_indices=None, empty_mask=None, scale=0.5):
    sum_loss = []
    loss_dict = []
    if face_indices is not None:
        mask_vert, maks_face = face_indices
        mesh = Meshes(verts=[v for v in vertices], faces=[faces[maks_face]] * len(vertices))
    else:
        mesh = Meshes(verts=[v for v in vertices], faces=[faces] * len(vertices))

    raster_settings = RasterizationSettings(image_size=int(cameras.image_size[0][0] * scale),   # Resolution    
                                            blur_radius=0.0,  # No anti-aliasing    
                                            faces_per_pixel=1, # Number of faces to store)
                                            cull_backfaces=True,
                                            perspective_correct=False,
                                            bin_size=0,
                                            z_clip_value=0.1)

    rasterizer = MeshRasterizer(cameras=cameras,    
                                raster_settings=raster_settings)
    sil_renderer = MeshRenderer(rasterizer=rasterizer, 
                                shader=SoftSilhouetteShader())
    soft_mask = sil_renderer(mesh)[..., 3] # (B,H, W)
    if empty_mask is not None:
        soft_mask[:, empty_mask>0] = 0
        mask[:,:, empty_mask>0] = 0
    # soft_mask = soft_mask ** 2
    for idx, sf in enumerate(soft_mask):    
        if (mask[idx].sum() > 1000):
            # sum_loss.append(mask_iou(sf.unsqueeze(0), mask[idx:idx+1].sum(dim=1)))
            # sum_loss.append(dice_loss(sf.unsqueeze(0), mask[idx:idx+1].sum(dim=1)))
            sum_loss.append(unmatched_area_loss(sf.unsqueeze(0), mask[idx:idx+1].sum(dim=1)))
            loss_dict.append(idx)
        # fragments = rasterizer(mesh[idx])
        # faces_idx = fragments.pix_to_face[..., 0] # (1, h, w)
        # valid_mask = faces_idx >= 0               # (1, h, w)

        # verts_idx = faces[faces_idx[valid_mask]]  # (n, 3)
        # verts_selected = verts[verts_idx] # (n, 3, 3)
        # bary_coords = fragments.bary_coords[valid_mask][..., 0, :]  # (n, 3)

        # P3d = (bary_coords[..., None] * verts_selected).sum(dim=-2)  # (n, 3)
        # P2d = cameras[idx].transform_points_screen(P3d)[:, :2]


    # Kaolin cameras
    # vertices_camera = cameras[1].extrinsics.transform(vertices)
    # vertices_image = cameras[1].intrinsics.transform(vertices_camera)
    # face_vertices_camera = index_vertices_by_faces(vertices_camera, faces)[:, face_indices]
    # face_vertices_image = index_vertices_by_faces(vertices_image, faces)[..., :2][:, face_indices]
    # in_face_features = torch.ones(tuple([len(vertices)] + list(faces[face_indices].shape) + [1]), dtype=cameras.dtype, device=cameras.device)
    # valid_faces = (face_vertices_camera[:, :, :, 2] < 0).all(dim=-1)  # Shape: (B, n)

    # for idx in range(len(mask)):
    #     _, face_idx = rasterize(
    #         cameras.height, cameras.width,
    #         face_features=in_face_features[idx:idx+1],
    #         face_vertices_z=face_vertices_camera[..., -1][idx:idx+1],  # can be face_vertices_image[..., -1] instead?
    #         face_vertices_image=face_vertices_image[idx:idx+1],
    #         valid_faces=valid_faces[idx:idx+1],)
    
    #     loss = mask_iou((face_idx>=0).float(), mask[0][:1] + mask[0][1:])
    #     sum_loss.append(loss)
    #     loss_dict.append(idx)
    
    return sum_loss, loss_dict


def show_image_with_uv(u, v, w=1024, h=1024):
    """
    Displays an image with a marker at the given (u, v) location.
    
    :param image: The image to display (numpy array)
    :param u: Normalized x-coordinate (0 to 1)
    :param v: Normalized y-coordinate (0 to 1)
    :param w: Image width
    :param h: Image height
    """
    import cv2
    # Convert normalized coordinates to pixel coordinates
    
    # Make a copy of the image to draw on
    img_copy = np.zeros((w, h, 3), dtype=np.uint8)
    
    for x, y in zip(u, v):
        x = int(x * w)
        y = int(y * h)
        # Draw a circle at (x, y)
        cv2.circle(img_copy, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
    
    # Show the image
    cv2.imshow("Image", img_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def opencv_to_opengl_viewmatrix(ex):
    M_flip = torch.tensor([[
        [1., 0, 0, 0],
        [0, -1, 0, 0],   
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ]], device=ex.device)
    return M_flip @ ex

def load_vertices():
    global CONTACT_VERTS
    """
    It loads the vertices of the contact regions
    
    Returns:
      A dictionary of vertices for each part of the foot.
    """
    all_vertices = read_json_file(os.path.join(root_path, 'vertices', 'all_new.json'))
    back = all_vertices['back_new']
    if CONTACT_VERTS is None:
        CONTACT_VERTS = {}
        for part in ['right_toe', 'left_toe', 'left_heel', 'right_heel']:
            CONTACT_VERTS[part] = np.array(all_vertices[part]['verts'], dtype=np.int32)

        CONTACT_VERTS['back'] = np.array(all_vertices['back_new']['verts'], dtype=np.int32)
    return CONTACT_VERTS
    
def trans_imu_smooth(trans_params, jump_list=[], mode='XY', noise=0.01):
    if mode == 'XY':
        select = [0, 1]
    elif mode == 'XYZ':
        select = [0, 1, 2]
    else:
        select = [0, 1, 2]
    # trans_diffs_a = torch.norm(trans_params[1:-1,select] - trans_params[0:-2, select], dim =1)
    # trans_diffs_b = torch.norm(trans_params[2:,select] - trans_params[1:-1, select], dim =1)
    # acc = torch.norm(trans_params[2:,select] - 2*trans_params[1:-1, select] + trans_params[0:-2, select], dim =1)
    acc = torch.abs(trans_params[2:,select] - 2*trans_params[1:-1, select] + trans_params[0:-2, select])
    acc = acc @ torch.from_numpy(np.array([1., 1., 0.333])).float().to(acc.device)
    # valid_list = np.array(sorted(list(set(np.arange(len(trans_params))) - set(jump_list))))[2:] - 2
    # imu_diffs = torch.norm(imu_trans[:-1,select] - imu_trans[1:, select], dim =1)
    # diffs = trans_diffs - imu_diffs
    # diffs = acc[valid_list]
    # diffs_new2 = F.relu(acc - noise)
    return acc, np.arange(2, len(trans_params)).tolist()

def joints_smooth(joints, mode='XY', weight=None, noise=0.01, framerate=20):
    """
    It takes a sequence of joints and returns a sequence of the same length, where each element is the
    average distance between the current joint and the next one
    
    Args:
      joints: the joints of the skeleton
      mode: 'XY' or 'XYZ'. Defaults to XY
      noise: the threshold for the difference between two consecutive joints.
    
    Returns:
      The mean of the difference between the joints.
    """
    if mode == 'XY':
        select = [0, 1]
    elif mode == 'XYZ':
        select = [0, 1, 2]
    else:
        select = [0, 1, 2]
    # loss = torch.mean(torch.mean(torch.abs(joints[1:,select] - joints[:-1,select]), dim=-1), dim=-1)
    rel_joints = joints - joints[:, :1, :]

    def robust_filter(res, frame_rate=20):
        ratio = 1/frame_rate
        squared_res = res ** 2
        dist = torch.div(squared_res, squared_res + ratio ** 2)
        return ratio ** 2 * dist
    
    acc = torch.abs(rel_joints[2:, 1:, select] + rel_joints[:-2,1:,select] - 2 * rel_joints[1:-1,1:,select])
    acc = acc.reshape(len(acc), -1, 3)
    # acc = robust_filter(acc)
    if weight is not None:
        if len(weight) < acc.shape[1]:
            weight = np.concatenate((weight, 0.2 * np.ones(acc.shape[1]-len(weight)))).astype(np.float32)
        loss = acc.mean(-1)  @ torch.from_numpy(weight).to(acc.device)
    else:
        loss = acc.mean(-1).sum(-1)
    # joints_diffs = torch.norm(rel_joints[:-1, :, select] - joints[1:, :, select], dim =-1)
    # loss = F.relu(joints_diffs - noise).mean(dim=-1)

    return loss

def joint_orient_error(pred_mat, gt_mat, num=1e-4):
    """
    Find the orientation error between the predicted and GT matrices
    Args:
        pred_mat: Batch x 3 x 3
        gt_mat: Batch x 3 x 3
    Returns:

    """
    if len(pred_mat.shape) == 2:
        r1 = axis_angle_to_rotation_matrix(pred_mat)
    else:
        r1 = pred_mat

    if len(gt_mat.shape) == 2:
        r2 = axis_angle_to_rotation_matrix(gt_mat)
    else:
        r2 = gt_mat

    r2t = torch.transpose(r2, 2, 1)
    r = torch.bmm(r1, r2t)

    pad_tensor = F.pad(r, [0, 1])
    residual = rotation_matrix_to_axis_angle(pad_tensor)
    # norm_res = torch.linalg.norm(residual * 1e4, dim=1) / 1e4
    norm_res = torch.norm(residual, p=2, dim=1)
    # norm_res = F.relu(norm_res - num)
    return norm_res

def sliding_constraint(smpl_verts, foot_states, jump_list, lfoot_move = None, rfoot_move = None):
    """
    For each frame, if the foot is in contact with the ground, we compute the difference between the
    current frame and the previous frame for the vertices that are in contact with the ground. 
    
    We then compute the norm of the difference vector and compare it to the minimum distance the foot
    can move. 
    
    If the norm is greater than the minimum distance, we add the difference to the loss. 
    
    We then return the mean of the loss and the number of frames that the foot was in contact with the
    ground. 
    
    The number of frames is used to normalize the loss. 
    
    The reason we do this is because we want to penalize the model for sliding more than the minimum
    distance, but we don't want to penalize the model for sliding less than the minimum distance. 
    
    If we just returned the mean of the loss, then the model would be penalized for sliding less than
    the minimum distance.
    
    Args:
      smpl_verts: the vertices of the SMPL model
      foot_states: a list of integers, each integer represents the state of the foot at that frame.
      jump_list: a list of indices of frames where the character jumps
      lfoot_move: the minimum distance the left foot can move in a frame
      rfoot_move: the minimum distance the right foot can move in one frame
    
    Returns:
      The loss and the number of sliding frames
    """
    contact_verts = load_vertices()
    # vertex_diffs = smpl_verts[:-1] - smpl_verts[1:]
    # vertex_diffs = vertex_diffs.reshape(-1, 3)
    # valid_vertex_diffs = vertex_diffs[frame_verts, :]
    # normed_vertex_diffs = torch.norm(valid_vertex_diffs,  p = 2, dim = 1)

    # _min_move = np.array([lfoot_move, rfoot_move]).min(axis=0)
    # min_move = []
    right_foot = np.concatenate((contact_verts['right_heel'], contact_verts['right_toe']))
    left_foot = np.concatenate((contact_verts['left_heel'], contact_verts['left_toe']))[:71]
    feet = np.concatenate((left_foot, right_foot))

    # valid_vertex_diffs = torch.empty(0,3).to(smpl_verts.device)
    index = []
    losses = []
    if lfoot_move is not None and rfoot_move is not None:
        for i, s in enumerate(foot_states):
            if i==0 or i in jump_list:
                continue
            if s == 0:
                vertex_diffs = smpl_verts[i, feet] - smpl_verts[i-1, feet]
            elif s==-1:
                vertex_diffs = smpl_verts[i, left_foot] - smpl_verts[i-1, left_foot] 
            elif s==1:
                vertex_diffs = smpl_verts[i, right_foot] - smpl_verts[i-1, right_foot]
            elif lfoot_move[i] > rfoot_move[i]:
                vertex_diffs = smpl_verts[i, right_foot] - smpl_verts[i-1, right_foot]
                # min_move += [rfoot_move[i]] * len(right_foot)
            else:
                vertex_diffs = smpl_verts[i, left_foot] - smpl_verts[i-1, left_foot] 
                # min_move += [lfoot_move[i]] * len(left_foot)

            vertex_diffs = vertex_diffs.reshape(-1, 3)
            # diff_i = (vertex_diffs * vertex_diffs).mean() * 100
            diff_i = torch.norm(vertex_diffs, dim=1).mean()
            if diff_i and not torch.any(torch.isnan(diff_i)):
                index.append(i)
                losses.append(diff_i)
    else:
        # cur_idx = -1
        pre_state = foot_states[0]
        pre_stable_foot = feet

        for i, cur_state in enumerate(foot_states):

            if i == 0:
                continue
            if cur_state == pre_state and cur_state != -2:
                if cur_state == 1:
                    vertex_diffs = smpl_verts[i, right_foot] - smpl_verts[i-1, right_foot] 
                    pre_stable_foot = right_foot
                elif cur_state == -1:
                    vertex_diffs = smpl_verts[i, left_foot] - smpl_verts[i-1, left_foot] 
                    pre_stable_foot = left_foot
                elif cur_state == 0:
                    vertex_diffs = smpl_verts[i, feet] - smpl_verts[i-1, feet] 
                vertex_diffs = vertex_diffs.reshape(-1, 3)
                diff_i = torch.norm(vertex_diffs, dim=1).mean()                
                # diff_i = (vertex_diffs * vertex_diffs).mean() * 100
                if diff_i and not torch.any(torch.isnan(diff_i)):
                    index.append(i)
                    losses.append(diff_i)

            pre_state = cur_state
            
    return losses, index


def get_contacinfo(foot_states, jump_list, ground, smpl_verts):
    """
    > Given a list of foot states, a list of jump frames, a list of ground planes, and a list of SMPL
    vertices, return a dictionary containing the vertices of the contact points, the indices of the
    contact points, the ground planes, and the indices of the frames with contact points
    
    Args:
      foot_states: a list of the foot states for each frame.
      jump_list: a list of frame indices that are not part of the walk cycle.
      ground: a list of ground planes, one for each frame.
      smpl_verts: the vertices of the SMPL model
    
    """

    contact_verts = load_vertices()

    contact_info = {}
    batch_verts = np.empty(0)
    # batch_planes = o3d.geometry.PointCloud()
    num_contact = []
    verts_index = []
    verts_num = 0

    for i in range(len(foot_states)):
        if i in jump_list:
            continue

        if foot_states[i] == 1:
            foot = 'right' 
        elif foot_states[i] == -1:
            foot = 'left'
        elif foot_states[i] == 0:
            left_heels = smpl_verts[i, contact_verts['left_heel']].contiguous()
            left_toes = smpl_verts[i, contact_verts['left_toe']].contiguous()
            right_heels = smpl_verts[i, contact_verts['right_heel']].contiguous()
            right_toes = smpl_verts[i, contact_verts['right_toe']].contiguous()

            left_foot = torch.cat((left_heels, left_toes), dim=0).contiguous()
            right_foot = torch.cat((right_heels, right_toes), dim=0).contiguous()
            z_lf = left_foot.mean(dim=0)[2]
            z_rf = right_foot.mean(dim=0)[2]
            foot = 'right' if z_rf < z_lf else 'left'
        else:
            # 'not walking'
            continue

        z_postion_h = smpl_verts[i, contact_verts[foot + '_heel']].mean(dim=0)[2]
        z_postion_t = smpl_verts[i, contact_verts[foot + '_toe']].mean(dim=0)[2]
        
        part = 'heel' if z_postion_h < z_postion_t else 'toe'

        # lowest_position = smpl_verts[i, contact_verts[f'{foot}_{part}']].mean(dim=0)[2]
        # batch_planes += ground_list[i]
        batch_verts = np.concatenate((batch_verts, contact_verts[f'{foot}_{part}'] + 6890 * i))
        verts_index.append([verts_num, verts_num + len(contact_verts[f'{foot}_{part}'])])
        verts_num += len(contact_verts[f'{foot}_{part}'])
        num_contact.append(i)

    # if batch_planes.has_points():
    #     device = smpl_verts.device
    #     batch_planes = batch_planes.voxel_down_sample(voxel_size=0.01)
    #     batch_planes_points = torch.from_numpy(np.asarray(batch_planes.points)[
    #                                         None, :]).type(torch.FloatTensor).to(device)
    # else:
    #     batch_planes_points = None

    if ground.has_vertices():
        faces = torch.from_numpy(np.asarray(ground.triangles)).long()
        vertices = torch.from_numpy(np.asarray(ground.vertices)).float()[None,]
        face_vertices = index_vertices_by_faces(vertices, faces).to(smpl_verts.device)
    else:
        face_vertices = None

    contact_info['verts']       = batch_verts     # (X, )  stores all index numbers of smpl vertices having foot contacting
    contact_info['verts_index'] = verts_index     # [N * [start, end]] an N-lenght list
    contact_info['planes']      = face_vertices   # (1, P, 3, 3)    scenes faces vertices
    contact_info['index']       = num_contact     # an N-lenght list, indicating the frame number that has foot contacting

    if face_vertices is not None:
        contact_info['face_normals'] = face_normals(face_vertices, unit=True)
    else:
        contact_info['face_normals'] = None

    return contact_info

def check_points_to_faces(points, faces, faces_normals):
    """
    check a set of 3D points are on the same side as their corresponding triangle faces.
    
    Args:
        points: a tensor of shape (B, P, 3) representing the 3D points to project.
        faces: a tensor of shape (B, P, 3, 3) representing the triangle faces.
        faces_normals: a tensor of shape (B, P, 3) representing the normal vectors of the faces.
        
    Returns:
        A boolean tensor of shape (B, P) indicating whether each point is on the same side as the face normal or not.
    """
    # Compute the dot product between each vertex-to-point vector and the face normal
     # (B, P, 3, 3) (B, P, 3) -> (B, P, 3)
    dot_product = torch.einsum('bpv, bpnv->bpn', faces_normals, points.unsqueeze(dim=2) - faces)
    
    # Return True if any dot products are greater than or equal to zero, False otherwise
    return dot_product.ge(0).sum(dim=-1)

def foot_collision(smpl_verts, 
                   contact_info: dict, 
                   weight=1.,
                   is_filter=True) ->tuple:

    if contact_info['planes'] is not None and contact_info['face_normals'] is not None:
        sum_loss   = torch.zeros(len(smpl_verts), device=smpl_verts.device)
        feet_verts = smpl_verts[:, list(set(BP['feet']+BP['legs']))].reshape(1, -1, 3).contiguous()

        cd_losses, index, dist_type = point_to_mesh_distance(feet_verts, contact_info['planes'])
        selected = dist_type == 0

        sameside = check_points_to_faces(feet_verts[selected].unsqueeze(0), 
                                         contact_info['planes'][:, index[selected]], 
                                         contact_info['face_normals'][:, index[selected]])
        
        cd_losses[0].index_put_(
                        indices=(selected[0].nonzero(as_tuple=False)[sameside.squeeze() >= 2],),
                        values=torch.tensor(0.0).to(cd_losses.device))
        cd_losses[0].index_put_(
                        indices=((~selected)[0].nonzero(as_tuple=False),),
                        values=torch.tensor(0.0).to(cd_losses.device))
        
        if is_filter:
            cd_losses = filter_loss(cd_losses, a=0.3, b=0.02)
        else:
            cd_losses = torch.sqrt(cd_losses + 1e-8)

        for i, loss in enumerate(cd_losses.reshape(smpl_verts.shape[0], -1)):
            if (loss>1e-8).sum() > 0:
                sum_loss[i] = loss[loss>1e-6].sum() * weight 
        valid_list = list((sum_loss>1e-6).nonzero(as_tuple=False))
        return sum_loss[sum_loss>1e-6], valid_list
    
    return [], []

def contact_constraint(smpl_verts, 
                       contact_info: dict, 
                       shoes_height: float = 0.0) ->tuple:
    """
    It takes in the SMPL vertices and the contact information, and returns the loss and the indices of
    the contact points
    
    Args:
       - smpl_verts: the vertices of the SMPL model
       - contact_info: a dictionary containing the scene and contacting information
    """

    loss_list  = []
    cd_losses  = 0
    valid_list = []

    if contact_info['planes'] is not None and len(contact_info['verts']) > 0:
        smpl_verts = smpl_verts.reshape(-1, 3)
        verts = smpl_verts[contact_info['verts'], :].reshape(1, -1, 3).contiguous()

        cd_losses, _, dist_type = point_to_mesh_distance(verts, contact_info['planes'])
        cd_losses[dist_type != 0] = 0

        cd_losses[cd_losses < shoes_height**2] = 0
        cd_losses = torch.sqrt(cd_losses[0] + 1e-8)

        for i, v in enumerate(contact_info['verts_index']):
            cd = cd_losses[v[0]:v[1]].mean()
            if cd and not torch.isnan(cd):
                loss_list.append(cd)
                valid_list.append(i)

    return loss_list, [contact_info['index'][i] for i in valid_list]

def get_optmizer(opt_vars, param_list, learn_rate):
    for param in param_list:
        param.grad = None
        param.requires_grad = True
    optimizer = torch.optim.Adam(param_list, learn_rate, betas=(0.9, 0.999))
    print(f'Optimizing {opt_vars}')
    return optimizer

def filter_loss(loss, a=0.3, b=0.02):
    """
    The function takes in a loss value and two optional parameters, and returns a filtered value based
    on a mathematical formula.
    
    Args:
      loss: The input parameter "loss" is a variable that represents the loss value of a model during
    training. It is used in the calculation of the output value "y" in the function "filter_loss".
      a: The parameter "a" is a constant value used in the calculation of the filtered loss. It is used
    to scale the output of the filter.
      b: The parameter "b" is a constant value used in the calculation of the filtered loss. It is set
    to a default value of 0.02, but can be adjusted as needed. It is used in the denominator of the
    equation to prevent division by zero and to control the rate at which the
    
    Returns:
      the value of `y`, which is calculated using the formula `y = a - a * b / (loss + b)`.
    """
    y = a - a * b / (loss + b) 
    return y

def mesh2point_loss(smpl_verts, 
                    human_points, 
                    vis_smpl_idx, 
                    start        = 0, 
                    thresh       = 0.3, 
                    trunk_dist   = 0.3, 
                    is_filter    = True,
                    min_loss_num = 300):
    """
    This function calculates the mesh-to-point loss between SMPL vertices and human point clouds, filtered
    by distance thresholds and body part categories.
    
    Args:
      `smpl_verts`: The vertices of the SMPL model (n, 3)
      `human_points`: a dictionary containing the human body point clouds for each frame.
    The keys of the dictionary are frame numbers and the values are numpy arrays of shape
    (num_points, 3).
      `vis_smpl_idx`: a list of indices indicating which SMPL vertices are visible in each frame. It is used to filter out vertices that are not visible in the LiDAR view.
      `start`: the index for processing the data. Defaults to 0
      `thresh`: a value used for filtering the mesh-to-point distances.
      `trunk_dist`: the maximum distance threshold between a SMPL vertex and a human point for them to be
    considered a match.
      `is_filter`: a boolean parameter that determines whether to apply a filter to the loss. 
    
    Returns:
      two lists: `losses` and `num_m2pl`. `losses` contains the mesh-to-point losses for each frame,
    while `num_m2pl` contains the frame numbers for which the mesh-to-point losses were calculated.
    """

    frames = [k for k in human_points.keys()]
    
    losses   = []
    num_m2pl = []

    def parts_cd_dist(verts, points, verts_rest, point_rest, weight=1.):
        """
        This function calculates the distance between vertices and points, filters them based on distance
        thresholds, and returns the remaining points.
        """
        if len(verts_rest) == 0 or len(point_rest) == 0:
            return torch.tensor([]).cuda(), point_rest

        m2p_loss, p2m_loss, index, _ = distChamfer(verts[verts_rest].contiguous().unsqueeze(0), 
                                                   points[point_rest].contiguous().unsqueeze(0))
        
        if len(m2p_loss[0]) > 0 and not torch.any(torch.isnan(m2p_loss)):
            w  = m2p_loss[0] <= trunk_dist ** 2     # SMPL verts that are  < trunk_dist to the points
            w2 = p2m_loss[0] <= 0.08**2             # points that are < 5cm to the smpl verts
            if is_filter:
                m2p_loss = filter_loss(m2p_loss[0][w], a=thresh, b=0.02)     # y=\frac{0.3x^{2}}{x^{2}+0.02}
            else:
                m2p_loss = torch.sqrt(m2p_loss[0][w] + 1e-8)

            cloest_indices = set(np.arange(len(p2m_loss[0]))[w2.cpu().numpy()]) | set(index[0].cpu().numpy())

            # cloest_points = np.concatenate((np.arange(len(cloest_indices))[:, None], 
            #                   points[point_rest[list(cloest_indices)]].cpu().numpy()), axis=1)
            
            # np.savetxt(f'{part}.txt', cloest_points, fmt=['%d', '%.3f', '%.3f', '%.3f'])
            
            rest_points_ind = set(point_rest) - set(point_rest[list(cloest_indices)])
            
            return m2p_loss * weight, np.asarray(list(rest_points_ind))
        else:
            return torch.tensor([]).cuda(), point_rest

    for i, verts in enumerate(smpl_verts):
        if i + start not in frames:            
            continue
        visible_vert_idx = vis_smpl_idx[i][0][vis_smpl_idx[i][1]]
        torso  = (set(BP['torso']) & set(visible_vert_idx))
        head   = (set(BP['head'])  & set(visible_vert_idx))
        feet   = (set(BP['feet'])  & set(visible_vert_idx))
        hands  = (set(BP['hands']) & set(visible_vert_idx))
        arms   = (set(BP['arms'])  & set(visible_vert_idx))
        legs   = (set(BP['legs'])  & set(visible_vert_idx))

        main   = list(torso | head | legs)
        # legs   = list((set(vis_smpl_idx[i]) - set(main)) & set(legs))
        arms   = list((set(visible_vert_idx) - set(main) ) & set(arms))
        ends   = list(feet | hands)
        ends   = list((set(visible_vert_idx) - set(main) - set(arms)) & set(ends))
        
        points = torch.from_numpy(human_points[i+start]).type(torch.FloatTensor).cuda()

        loss_main, rest_indices = parts_cd_dist(verts, points, main, np.arange(len(points)), 1.0)
        loss_arms, rest_indices = parts_cd_dist(verts, points, arms, rest_indices, 1.0)
        loss_ends, rest_indices = parts_cd_dist(verts, points, ends, rest_indices, 0.2)
        
        loss = torch.cat((loss_main, loss_arms, loss_ends), dim=0)

        if len(loss>0) and loss.mean():
            losses.append(loss.mean() if len(loss) > min_loss_num else loss.sum()/min_loss_num)
            num_m2pl.append(i)

    return losses, num_m2pl

def index_face_by_vindex(faces, index):
    """
    The function selects faces from a mesh based on a given vertex index.
    
    Args:
      faces: The `faces` parameter is a tensor representing the faces of a 3D mesh. Each row of the
    tensor contains the indices of the vertices that make up a single face.
      index: The `index` parameter is a list or array of vertex indices that we want to use to select
    faces from a mesh.
    
    Returns:
      The function `index_face_by_vindex` returns a subset of the input `faces` tensor, which contains
    only the faces that contain at least one vertex with the index specified in the `index` argument.
    The returned tensor has the same shape as the input `faces` tensor, but with fewer rows (i.e.,
    faces).
    """
    selected_vertices = torch.tensor(index, dtype=torch.long, device=faces.device) 
    
    selected_faces = torch.any(torch.eq(faces.unsqueeze(0), 
                                        selected_vertices.unsqueeze(1).unsqueeze(1)), dim=0).all(dim=1)
    return faces[selected_faces]

def points2smpl_loss(smpl_verts, 
                     human_points, 
                     vis_smpl_idx, 
                     smpl_faces  = None, 
                     start       = 0, 
                     trunk_dist  = 0.2,
                     is_filter   = True,
                    min_loss_num = 300):

    frames = [k for k in human_points.keys()]
    losses = []
    num_m2pl = []
    for i, verts in enumerate(smpl_verts):
        if i + start not in frames:            
            continue
        points = torch.from_numpy(human_points[i+start]).type(torch.FloatTensor).cuda()

        selected_faces = index_face_by_vindex(smpl_faces, vis_smpl_idx[i][0])
        face_vertices = index_vertices_by_faces(verts.unsqueeze(0), selected_faces)

        if len(face_vertices) > 0:
            loss, _, dist_type = point_to_mesh_distance(points.unsqueeze(0), face_vertices)
            
            loss = loss[dist_type != 0]
            loss = loss[loss < trunk_dist ** 2]

            if is_filter:
                loss = filter_loss(loss, a=0.1, b=0.004)
            else:
                loss = torch.sqrt(loss + 1e-8)

            if len(loss) > 0 and not torch.any(torch.isnan(loss)):
                losses.append(loss.mean() if len(loss) > min_loss_num else loss.sum()/min_loss_num)
                num_m2pl.append(i)

    return losses, num_m2pl

def compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # print('X1', X1.shape)

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2)

    # print('var', var1.shape)

    # 3. The outer product of X1 and X2.
    K = X1.mm(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)
    # V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    # Construct R.
    R = V.mm(Z.mm(U.T))

    # print('R', X1.shape)

    # 5. Recover scale.
    scale = torch.trace(R.mm(K)) / var1
    # print(R.shape, mu1.shape)
    # 6. Recover translation.
    t = mu2 - scale * (R.mm(mu1))
    # print(t.shape)

    # 7. Error:
    S1_hat = scale * R.mm(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return R, t, scale
    # return S1_hat

def compute_vertex_normals(vertices, faces, unit=True):
    """
    from :https://github.com/ShichenLiu/SoftRas/blob/master/soft_renderer/functional/vertex_normals.py
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3 or faces.ndimension() == 2)
    if faces.ndimension() == 2:
        faces = faces.unsqueeze_(0).repeat([vertices.shape[0],1,1])
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)
    faces = faces + (torch.arange(bs).to(device) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(),
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                       vertices_faces[:, 0] - vertices_faces[:, 1], dim=-1))
    normals.index_add_(0, faces[:, 2].long(),
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                       vertices_faces[:, 1] - vertices_faces[:, 2], dim=-1))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                       vertices_faces[:, 2] - vertices_faces[:, 0], dim=-1))

    if unit:
        normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

def angle_between_vectors(a_norm, b_norm):
    """
    calculates the angle between two normalized vectors using cosine similarity
    
    Args:
      a_norm: a normalized vector representing vector a.
      b_norm: a PyTorch tensor representing a normalized vector.
    
    Returns:
      the angle in degrees between two normalized vectors
    """

    cos_sim = F.cosine_similarity(a_norm, b_norm, dim=-1)
    angle = torch.acos(cos_sim) * 57.29577951308238

    angle[torch.isnan(angle)] = 0

    return angle

def part_coll_dist(smpl_verts, 
                   verts_normals, 
                   partA, 
                   partB, 
                   weight=1., 
                   angle_thresh=140, 
                   both_side=False,
                   is_filter=True):
    """
    This function calculates the distance between two parts of a 3D model based on their vertices and
    normals, and returns a loss value for each vertex.
    
    Args:
      smpl_verts: A tensor of shape (N, V, 3) representing the 3D coordinates of N vertices for a mesh
    with V vertices.
      verts_normals: Normals of the vertices of a 3D mesh.
      partA: The indices of the vertices belonging to the mesh.
      partB: The indices of the vertices belonging to the same mesh.
      weight: Weight is a scalar value that determines the importance of the distance
      angle_thresh: The angle threshold in degrees used to determine if a vertex is considered to be
    part of the collision between partA and partB. If the angle between the normal of the vertex and the
    vector from the vertex to the closest point on the other part is greater than angle_thresh, the
    vertex is considered to. Defaults to 140
      both_side: A boolean parameter that determines whether to calculate the distance between partA and
    partB in both directions (partA to partB and partB to partA). Defaults to
    False
    
    Returns:
      a tensor of losses for each vertex in smpl_verts, 
    """
    losses  = torch.zeros(len(smpl_verts), device=smpl_verts.device)

    a2b, b2a, a2b_index, b2a_index = distChamfer(smpl_verts[:, partA], 
                                                smpl_verts[:, partB])

    indices  = a2b_index.to(torch.int64).unsqueeze(-1).repeat(1, 1, 3)
    position = torch.gather(smpl_verts[:, partB], 1, indices)
    normals  = torch.gather(verts_normals[:, partB], 1, indices)
    vectors  = smpl_verts[:, partA] - position
    angles   = angle_between_vectors(normals, F.normalize(vectors, eps=1e-6, dim=-1))

    cd_losses = filter_loss(a2b, 0.2, 0.01) if is_filter else torch.sqrt(a2b+1e-6)
    for i, (angle, loss) in enumerate(zip(angles, cd_losses)):
        if (angle>angle_thresh).sum() > 0:
            losses[i] += loss[angle>angle_thresh].mean() * weight 
    
    if both_side:
        indices  = b2a_index.to(torch.int64).unsqueeze(-1).repeat(1, 1, 3)
        position = torch.gather(smpl_verts[:, partA], 1, indices)
        normals  = torch.gather(verts_normals[:, partA], 1, indices)
        vectors  = smpl_verts[:, partB] - position
        angles   = angle_between_vectors(normals, F.normalize(vectors, eps=1e-6, dim=-1))

        cd_losses = filter_loss(b2a, 0.2, 0.01) if is_filter else torch.sqrt(b2a+1e-8)
        for i, (angle, loss) in enumerate(zip(angles, cd_losses)):
            if (angle>angle_thresh).sum() > 0:
                losses[i] += loss[angle>angle_thresh].mean() * weight 

    return losses

def collision_loss(smpl_verts, smpl_faces):
    """
    The function calculates collision loss between different body parts of a 3D model using vertex
    normals and distance calculations.
    
    Args:
      smpl_verts: The vertices of the SMPL model.
      smpl_faces: The faces of the SMPL model, which define the triangular mesh of the body surface.
    
    Returns:
      a tuple containing two elements: 
    1. A tensor of collision losses for different vertices of the SMPL model, calculated based on the
    distances and angles between different body parts (left arm, right arm, body, left leg, right leg). 
    2. A list of indices representing the vertices for which the collision losses were calculated.
    """

    verts_normals = compute_vertex_normals(smpl_verts, smpl_faces.clone(), unit=True)
    
    # left hand to right hand collision
    hh_losses = part_coll_dist(smpl_verts, verts_normals, 
                               BP['left_arm'], BP['right_arm'], both_side=True)
    # hand to body collision
    hb_losses = part_coll_dist(smpl_verts, verts_normals, 
                               BP['right_arm'] + BP['left_arm'], BP['body'], both_side=True)
    # left leg to right leg collisison
    ll_losses = part_coll_dist(smpl_verts, verts_normals, 
                               BP['left_leg'], BP['right_leg'], both_side=True)
    # hands to legs collisison
    hl_losses = part_coll_dist(smpl_verts, verts_normals, 
                               BP['right_arm'] + BP['left_arm'], 
                               BP['right_leg'] + BP['left_leg'], 
                               both_side=True)

    sum_loss = hh_losses + hb_losses + ll_losses + hl_losses
    valid_list = list((sum_loss>1e-5).nonzero(as_tuple=False))
    return sum_loss[sum_loss>1e-5], valid_list

if False:
    from selfcontact import SelfContact
    from selfcontact.utils.extremities import get_extremities
    class penetration_loss(torch.nn.Module):
        """
        This function calculates the collision loss between the SMPL model and the environment.
        
        Args:
            smpl_verts: A tensor representing the vertices of a SMPL model.
            smpl_faces: `smpl_faces` is a tensor containing the indices of the vertices that form each face of
        the SMPL mesh. It has shape `(num_faces, 3)` where `num_faces` is the number of faces in the mesh.
            coll_loss_weight: `coll_loss_weight` is a scalar weight used to adjust the contribution of the
        collision loss to the overall loss function. It determines how much importance is given to the
        collision loss compared to other losses in the model.
        
        """

        
        def __init__(self, 
                    seg_lenght=1,
                    sigma = 0.5,
                    point2plane = True,
                    vectorized = True,
                    part_segm_fn = '',
                    MODEL_TYPE='smpl',
                    essentials_folder='/home/dyd/selfcontact/selfcontact-essentials',
                    device='cuda',
                    test_segments=False):
        
            super(penetration_loss, self).__init__()
            # self.search_tree = BVH(max_collisions=8)
            # self.pen_distance = \
            #     collisions_loss.DistanceFieldPenetrationLoss(sigma=sigma,
            #                                                 point2plane=point2plane,
            #                                                 vectorized=vectorized)

            self.cm = SelfContact( 
                essentials_folder=essentials_folder,
                geothres=0.3, 
                euclthres=0.03, 
                model_type=MODEL_TYPE,
                test_segments=test_segments,
                compute_hd=False,
                buffer_geodists=True,
            ).to(device)
            
            self.geodist = self.cm.geodesicdists
            self.MODEL_TYPE = MODEL_TYPE
            if MODEL_TYPE == 'smplx':
                self.ds = get_extremities(
                    os.path.join(essentials_folder, f'models_utils/{MODEL_TYPE}/smplx_segmentation_id.npy'), 
                    False
                )
            else:
                downsample = set(BP['low_limbs'] + BP['hands'])  
                self.ds = torch.tensor(list(downsample))

            self.sl = seg_lenght
            self.a1 = 0.04
            self.a2 = 0.04
            self.b1 = 0.07
            self.b2 = 0.06
            self.c1 = 0.01
            self.c2 = 0.01
            self.d1 = 0.023
            self.d2 = 0.02

            self.inside_w                  = 30
            self.contact_w                 = 30
            self.outside_w                 = 0.01
            self.hand_contact_prior_weight = 1.0
            # self.pose_prior_weight         = 1.0 
            # self.hand_pose_prior_weight    = 0.01
            self.angle_weight              = 0.001 

            # load hand-on-body prior (smplx)
            if MODEL_TYPE == 'smplx':
                HCP_PATH = os.path.join(essentials_folder, 'hand_on_body_prior/smplx/smplx_handonbody_prior.pkl')
                with open(HCP_PATH, 'rb') as f:
                    hand_contact_prior = pickle.load(f)
                lefthandids  = torch.tensor(hand_contact_prior['left_hand_verts_ids'])
                righthandids = torch.tensor(hand_contact_prior['right_hand_verts_ids'])
                weights      = torch.tensor(hand_contact_prior['mean_dist_hands_scaled'])
                self.hand_contact_prior = torch.cat((lefthandids,righthandids)).to(device)
                self.hand_contact_prior_weights = torch.cat((weights, weights)).to(device)
            else:
                pass

        def configure(self, vertices):
            self.register_buffer('init_verts', vertices.clone().detach())
            # get verts in contact in initial mesh
            self.init_verts_in_contact = []

            with torch.no_grad():
                print('Initializing penetration loss...')
                for verts in tqdm(self.init_verts, desc='Collisions searching...'):
                    self.init_verts_in_contact_idx = \
                        self.cm.segment_vertices(verts.unsqueeze(0), 
                                                test_segments=False)[0][1][0]
                    self.init_verts_in_contact.append(torch.where(self.init_verts_in_contact_idx)[0].cpu().numpy())

        def forward(self, batch_verts, test_segments=False):
            """
            Returns:
                the collision loss, which is calculated by multiplying the collision loss weight with the penalty
            distance between the triangles and the collision indices.
            """
            device = batch_verts.device
            bs, nv = batch_verts.shape[:2]


            insideloss  = torch.tensor(0.0, device=device)
            contactloss = torch.tensor(0.0, device=device)
            outsideloss = torch.tensor(0.0, device=device)
            angle_loss  = torch.tensor(0.0, device=device)

            hand_contact_loss_inside  = torch.tensor(0.0, device=device)
            hand_contact_loss_outside = torch.tensor(0.0, device=device)
            hand_contact_loss         = torch.tensor(0.0, device=device)
            
            left_hand_contact_loss_inside   = torch.tensor(0.0, device=device)
            right_hand_contact_loss_inside  = torch.tensor(0.0, device=device)
            left_hand_contact_loss_outside  = torch.tensor(0.0, device=device)
            right_hand_contact_loss_outside = torch.tensor(0.0, device=device)

            losses = []
            num = []

            for i, verts in enumerate(batch_verts):
                if test_segments:
                    (v2v_min, v2v_min_idx, exterior) \
                        = self.cm.segment_vertices_scopti(
                            vertices=verts.unsqueeze(0),
                            test_segments=test_segments)
                    exterior = exterior[:,self.ds]
                else:
                    v2v_min, _, exterior = self.cm.segment_points_scopti(
                        points=verts[None, self.ds, :],
                        vertices=verts.unsqueeze(0)
                    )
                v2v_min  = v2v_min.squeeze()
                exterior  = exterior.squeeze()


                # only extremities intersect
                inside = torch.zeros(nv).to(device).to(torch.bool)
                true_tensor = torch.ones((~exterior).sum().item(), device=device, dtype=torch.bool)
                inside[self.ds[~exterior]] = true_tensor

                if (~inside).sum() > 0:
                    if len(self.init_verts_in_contact[i]) > 0:
                        gdi = self.geodist[:, self.init_verts_in_contact[i]].min(1)[0]
                        weights_outside = 1 / (5 * gdi + 1)
                    else:
                        weights_outside = torch.ones_like(self.geodist)[:,0].to(device)
                    attaction_weights = weights_outside[self.ds][~inside[self.ds]]
                    v2voutside  = v2v_min[self.ds][~inside[self.ds]]
                    v2voutside  = self.a1 * attaction_weights  * torch.tanh(v2voutside/self.a2)
                    contactloss = self.contact_w * v2voutside.mean()

                # push inside to surface
                if inside.sum() > 0:
                    v2vinside  = v2v_min[inside]
                    v2vinside  = self.b1 * torch.tanh(v2vinside / self.b2)
                    insideloss = self.inside_w * v2vinside.mean()

                # ==== hand-on-body prior loss ====
                if self.MODEL_TYPE == 'smplx':
                    ha = int(self.hand_contact_prior.shape[0] / 2)
                    hand_verts_inside = inside[self.hand_contact_prior]

                    if (~hand_verts_inside).sum() > 0:
                        left_hand_outside = v2v_min[self.hand_contact_prior[:ha]][(~hand_verts_inside)[:ha]]
                        right_hand_outside = v2v_min[self.hand_contact_prior[ha:]][(~hand_verts_inside)[ha:]]
                        # weights based on hand contact prior
                        left_hand_weights = -0.1 * self.hand_contact_prior_weights[:ha].view(-1,1)[(~hand_verts_inside)[:ha]].view(-1,1) + 1.0
                        right_hand_weights = -0.1 * self.hand_contact_prior_weights[ha:].view(-1,1)[(~hand_verts_inside)[ha:]].view(-1,1) + 1.0       
                        if left_hand_outside.sum() > 0:
                            left_hand_contact_loss_outside = self.c1 * torch.tanh(left_hand_outside/self.c2)
                        if right_hand_outside.sum() > 0:
                            right_hand_contact_loss_outside = self.c1 * torch.tanh(right_hand_outside/self.c2)
                        hand_contact_loss_outside = (left_hand_weights * left_hand_contact_loss_outside.view(-1,1)).mean() + \
                                                    (right_hand_weights * right_hand_contact_loss_outside.view(-1,1)).mean()

                    if hand_verts_inside.sum() > 0:
                        left_hand_inside  = v2v_min[self.hand_contact_prior[:ha]][hand_verts_inside[:ha]]
                        right_hand_inside = v2v_min[self.hand_contact_prior[ha:]][hand_verts_inside[ha:]]
                        if left_hand_inside.sum() > 0:
                            left_hand_contact_loss_inside = self.d1 * torch.tanh(left_hand_inside/self.d2)
                        if right_hand_inside.sum() > 0:
                            right_hand_contact_loss_inside = self.d1 * torch.tanh(right_hand_inside/self.d2)
                        hand_contact_loss_inside = left_hand_contact_loss_inside.mean() + right_hand_contact_loss_inside.mean()

                    hand_contact_loss = self.hand_contact_prior_weight * (hand_contact_loss_inside + hand_contact_loss_outside)

                # ==== align normals of verts in contact ====
                # verts_close = torch.where(v2v_min < 0.01)[0]
                # if len(verts_close) > 0:
                #     vertex_normals  = compute_vertex_normals(verts.unsqueeze(0), self.cm.faces.clone())
                #     dotprod_normals = torch.matmul(vertex_normals, torch.transpose(vertex_normals,1,2))[0]
                #     #normalsgather  = dotprod_normals.gather(1, v2v_min_idx.view(-1,1))
                #     dotprod_normals   = dotprod_normals[np.arange(nv), v2v_min_idx.view(-1,1)]
                #     angle_loss      = 1 + dotprod_normals[verts_close,:]
                #     angle_loss      = self.angle_weight * angle_loss.mean()

                # # ==== pose regression loss / outside loss ====
                # outsidelossv2v = torch.norm(self.init_verts-verts, dim=2)
                # if self.init_verts_in_contact.sum() > 0:
                #     gd = self.geodist[:, self.init_verts_in_contact].min(1)[0]
                #     outsidelossv2vweights = (2 * gd.view(verts.shape[0], -1))**2
                # else:
                #     outsidelossv2vweights = torch.ones_like(outsidelossv2v).to(device)
                # outsidelossv2v = (outsidelossv2v * outsidelossv2vweights).sum()
                # outsideloss = self.outside_w * outsidelossv2v

                loss = contactloss + insideloss
                if loss:
                    losses.append(loss)
                    num.append(i)

            return losses, num

