import os
import sys
import argparse

from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler

sys.path.append('.')
sys.path.append('..')

from smpl import convert_to_6D_rot, rot6d_to_rotmat
from tools import load_point_cloud, SLOPER4D_Loader, plot_points_on_img, load_mask


def extrinsic_to_cam(cam_ex):
    cam = torch.tensor([np.eye(4)] * len(cam_ex)).float().to(cam_ex.device)
    cam[..., :3, :3] = cam_ex[..., :3, :3].transpose(-1, -2) @ torch.tensor([[1.,0,0],[0,-1,0],[0,0,-1]]).to(cam.device)
    cam[..., :3, 3:] = -(cam_ex[..., :3, :3].transpose(-1, -2) @ cam_ex[..., :3, 3:])
    return cam

def cam_to_extrinsic(cam):
    cam_ex = torch.tensor([np.eye(4)] * len(cam)).float().to(cam.device)
    cam_ex[..., :3, :3] = torch.tensor([[1.,0,0],[0,-1,0],[0,0,-1]]).to(cam.device) @ cam[..., :3, :3].transpose(-1, -2)
    cam_ex[..., :3, 3:] = -(cam_ex[..., :3, :3] @ cam[..., :3, 3:])
    return cam_ex

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

class Camera(nn.Module):
    def __init__(self, cam_ex, 
                 cam_in, 
                 dist=torch.zeros((1, 5)), 
                 shape=(1080,1920), 
                 device='cpu'):
        """
        This function initializes camera parameters and sets them to be trainable.
        
        Args:
          cam_ex: the extrinsic camera parameters. shape (N, 3, 4) 
          cam_in: The camera intrinsic matrix, shape (3, 3)
          dist: the camera distortion, shape (5,)
          shape: the shape of the image, (height, width), default is (1080, 1920)
          device: specifies the device (CPU or GPU) on which the tensors will be
        stored and operated upon. Defaults to cpu
        """
        super().__init__()
        self.device = device
        
        cam = extrinsic_to_cam(cam_ex)

        self.cam_6d = convert_to_6D_rot(cam[...,:3,:3]).to(device)
        self.xyz = cam[...,:3, -1].to(device)
        
        # the original camera extrinsic matrix
        self.ori_ex = self.cam_6d.clone()
        self.ori_xyz = self.xyz.clone()

        # the initialized camera for every frame
        self.init_ex = [self.cam_6d.clone()]
        self.init_xyz = [self.xyz.clone()]

        self.cam_in = cam_in.to(device)
        self.dist   = dist.to(device)
        self.shape  = shape
        
    def reset_camera(self):
        self.init_ex.append(self.cam_6d.clone().detach())
        self.init_xyz.append(self.xyz.clone().detach())

        self.cam_6d = self.ori_ex.clone()
        self.xyz = self.ori_xyz.clone()

    def set_cam_ex(self, cam_ex):
        cam = extrinsic_to_cam(cam_ex)
        self.cam_6d = convert_to_6D_rot(cam[...,:3,:3]).to(self.device)
        self.xyz = cam[...,:3, -1].to(self.device)

        self.init_ex.append(self.cam_6d.clone().detach())
        self.init_xyz.append(self.xyz.clone().detach())

        self.ori_ex = self.cam_6d.clone().detach()
        self.ori_xyz = self.xyz.clone().detach()
        

    def get_optmizer(self, opt_type='all', learn_rate = 0.0001):
        if opt_type == 'all':
            param_list = [self.cam_6d, self.xyz]
        elif opt_type == 'rot':
            param_list = [self.cam_6d]
        elif opt_type == 'trans':
            param_list = [self.xyz]
        else:
            raise NotImplementedError
        
        for param in param_list:
            param.grad = None
            param.requires_grad = True
        optimizer = torch.optim.Adam(param_list, learn_rate, betas=(0.9, 0.999))
        return optimizer
        
    def _camera_to_pixel(self, X):
        f = self.cam_in[:2]
        c = self.cam_in[2:]
        k = torch.Tensor([self.dist[0],self.dist[1], self.dist[4]]).to(self.device)
        p = torch.Tensor([self.dist[2], self.dist[3]]).to(self.device)
        XX = X[..., :2] / (X[..., 2:])
        r2 = torch.sum(XX[..., :2]**2, axis=-1, keepdims=True)
        radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3),
                            axis=-1), axis=-1, keepdims=True)
        tan = 2 * torch.sum(p * torch.flip(XX, [-1,]), axis=-1, keepdims=True)
        XXX = XX * (radial + tan) + r2 * torch.flip(p, [-1,])
        return f * XXX + c
    
    def get_extrinsic(self, index=0):
        rots  = rot6d_to_rotmat(self.cam_6d)     # B, 3, 3
        trans = self.xyz.unsqueeze(-1)           # B, 3, 1
        ones  = torch.tensor([[[0, 0, 0, 1]]] * len(rots)).to(self.device)  # (B, 1, 4)
        cam = torch.cat((rots, trans), dim=-1)   # (B, 3, 4)
        return cam_to_extrinsic(torch.cat((cam, ones), dim=1))             # (B, 4, 4)

    def project_points(self, points):
        if points is not None and len(points) > 0:
            cam_ex_R = self.get_extrinsic()[0]
            points_cam = points @ cam_ex_R[:3, :3].T + cam_ex_R[:3, 3]
            points_cam = filter_points(points_cam)

            pixels = self._camera_to_pixel(points_cam)
            
            rule1 = pixels[:, 0] >= 0
            rule2 = pixels[:, 0] < self.shape[1]
            rule3 = pixels[:, 1] >= 0
            rule4 = pixels[:, 1] < self.shape[0]

            return pixels[rule1 & rule2 & rule3 & rule4]
        else:
            return torch.tensor([])
    
    def get_mask_loss(self, points, mask):
        loss = [0, 0, 0]
        if len(points) > 50 and len(mask) > 500:
            points_on_img = self.project_points(points)[None, :, [1, 0]]
            if len(points_on_img) > 0:
                loss = iou_loss(points_on_img, mask.unsqueeze(0))
        return loss
    
    def L1_loss_origin(self, ):
        trans_loss = torch.abs(self.xyz - self.ori_xyz).sum()
        ori_loss = torch.abs(self.cam_6d - self.ori_ex).sum()
        return ori_loss + trans_loss
    
    def L1_loss_init(self, ):
        trans_loss = torch.abs(self.xyz - self.init_xyz[-1]).sum()
        ori_loss = torch.abs(self.cam_6d - self.init_ex[-1]).sum()
        return ori_loss + trans_loss
    
    def L2_loss(self, ):
        trans_loss = torch.abs(self.xyz - self.ori_xyz).norm()
        ori_loss = torch.abs(self.cam_6d - self.ori_ex).norm(dim=0)
        return ori_loss + trans_loss

def compute_iou_loss(mask_pred, mask_gt, W, H):
    mask_pred, mask_gt = mask_pred.long(), mask_gt.long()
    pred, gt = torch.zeros(W, H), torch.zeros(W, H)
    pred[tuple(mask_pred.T)] = 1
    gt[tuple(mask_gt.T)] = 1
    addUp = pred + gt
    intersection = (addUp == 2).sum()
    union = (pred == 1).sum()
    iou_loss = 1 - intersection / union
    return iou_loss

def loss_filter(loss, a=50):
    """
    The function returns the loss value mappinp to (0-1).
    
    Args:
      loss: (n, )
      a: a constant that is used to adjust the strength of the filter. 
    """

    loss = 1 - 1 / (a * loss + 1) 
    return loss

def iou_loss(points_coord, mask_coord, min_pixel_dist = 1):
    """
    The function calculates the intersection over union (IOU) loss between two sets of points and masks.
    
    Args:
      points_coord: A tensor containing the coordinates of points in the predicted mask.
      mask_coord: The coordinates of the mask, which is typically a binary image indicating the region
    of interest.
      min_pixel_dist: The minimum distance (in pixels) between a point and a mask for them to be
    considered as intersecting. If the distance is less than this value, they are considered as
    non-intersecting. Defaults to 1
    
    Returns:
      two values: iou_loss1 and iou_loss2.
    """
    # chamLoss = cham.chamfer_2DDist()
    # dist1, dist2, idx1, idx2 = chamLoss(points_coord, mask_coord)

    p2m, m2p = chamfer_distance_x2y(points_coord, mask_coord)

    non_inter_points = loss_filter(torch.relu(p2m - (min_pixel_dist ** 2 + 1e-4))).sum()
    non_inter_mask = loss_filter(torch.relu(m2p - (5 ** 2 + 1e-4))).sum()

    iou_loss1 = non_inter_points / len(p2m)
    iou_loss2 = non_inter_mask / len(m2p)
    iou_loss = (non_inter_mask + non_inter_points) / (non_inter_points + non_inter_mask + len(m2p))
    return iou_loss1, iou_loss2, iou_loss

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
    
def save_results_to_img(img_bath_path, pc_base_path, frame, world2cam, new_world2cam, K, dist):
    
    # load image
    img_name = os.path.join(img_bath_path, frame['file_basename'])
    img = cv2.imread(img_name)

    # load mask
    mask_img = load_mask(frame['mask'].cpu().numpy()[None, :], False)
    cv2.add(img, mask_img)

    # load point cloud
    pcd_name = f"{frame['lidar_tstamps']:.03f}".replace('.', '_') + '.pcd'
    pc_path  = os.path.join(pc_base_path, pcd_name)
    sence_pc = load_point_cloud(pc_path)

    tmp_img  = plot_points_on_img(img.copy(), 
                                  np.asarray(sence_pc.points), 
                                  world2cam.cpu().numpy(), 
                                  intrinsic=K.cpu().numpy(), 
                                  dist=dist.cpu().numpy())
    
    cv2.imwrite("scene_test.jpg", tmp_img)

    tmp_img = plot_points_on_img(img, 
                                 np.asarray(sence_pc.points), 
                                 new_world2cam, 
                                 intrinsic=K.cpu().numpy(), 
                                 dist=dist.cpu().numpy())
    
    cv2.imwrite("scene_new_test.jpg", tmp_img)

def main(args, device, window, skip, index):

    if args.pkl_path is None:
        base_path = args.root_folder
        seq_name  = os.path.basename(args.root_folder)
        pkl_path  = os.path.join(base_path, seq_name+'_labels.pkl')
    else:
        pkl_path  = args.pkl_path
        base_path = os.path.dirname(pkl_path)
        seq_name  = os.path.basename(os.path.splitext(pkl_path)[0])[:-7]

    img_bath_path = os.path.join(base_path, 'rgb_data', seq_name+'_imgs')
    pc_base_path  = os.path.join(base_path, "lidar_data", "lidar_frames_rot")

    dataset = SLOPER4D_Loader(pkl_path, device=device, print_info=False, return_smpl=True)
    
    # camera initialization
    E, K, dist = dataset.get_cam_params()
    W, H       = dataset.get_img_shape()

    camera = Camera(cam_ex = E.unsqueeze(0).clone(),
                    cam_in = K.clone(), 
                    dist   = dist.clone(), 
                    shape  = (H, W), 
                    device = device)

    indices = np.arange(len(dataset))[max(0, args.start): min(args.end, len(dataset))]

    bar = tqdm(indices) if index is None else tqdm([index])
    
    try:
        for index in bar:
            mask_list   = []
            points_list = []
            verts_list  = []
            weights     = []

            # determine the start and end frame used for loss calculation
            w_start = index - (window)//2 * skip
            w_end = index  + (window+1)//2 * skip

            if w_start < 0:
                w_end += abs(w_start)
                w_start = 0
            elif w_end >= len(dataset):
                w_start -= w_end - len(dataset)
                w_end = len(dataset) - 1

            for i in range(w_start, w_end, skip):
                frame = dataset[i]
                human_mask   = frame['mask'].nonzero(as_tuple=False).float()
                human_points = frame['human_points']
                smpl_verts   = frame['smpl_verts']
                if len(human_points) > 50:
                    human_points = human_points @ frame['lidar_pose'][:3, :3].T + frame['lidar_pose'][:3, -1]
                if len(smpl_verts) > 6000:
                    smpl_verts = smpl_verts @ frame['lidar_pose'][:3, :3].T + frame['lidar_pose'][:3, -1]
                mask_list.append(human_mask)
                points_list.append(human_points)
                verts_list.append(smpl_verts)
                weights.append(max(pow(abs(index-i)+1, -0.5), 0.2))

            frame       = dataset[index]
            world2lidar = frame['lidar_pose'].clone()
            world2cam   = frame['cam_pose'].clone()
            
            # lidar2cam   = world2cam @ world2lidar.inverse()
            # camera.set_cam_ex(lidar2cam[None, ...])

            optimizer  = camera.get_optmizer(opt_type='all')
            scaler     = GradScaler()

            loss       = torch.tensor([0])
            pre_losss  = 0

            for iters in range(200):
                iou_loss  = 0
                iou_loss1 = 0
                iou_loss2 = 0

                for i, (points, mask, verts) in enumerate(zip(points_list, mask_list, verts_list)):
                    if args.use_smpl:
                        iou1, iou2, iou = camera.get_mask_loss(verts, mask)
                    else:
                        iou1, iou2, iou = camera.get_mask_loss(points, mask)
                    iou_loss  += weights[i] * iou
                    iou_loss1 += weights[i] * iou1
                    iou_loss2 += weights[i] * iou2

                iou_loss  = iou_loss / len(mask_list)
                iou_loss1 = iou_loss1 / len(mask_list)
                iou_loss2 = iou_loss2 / len(mask_list)
                cam_smth  = camera.L1_loss_init()
                l1_loss   = camera.L1_loss_origin()

                if iters < 30:
                    loss = 20 * iou_loss + 30 * iou_loss1 + 20 * iou_loss2
                else:
                    loss = 20 * iou_loss + 30 * iou_loss1 + 20 * iou_loss2
                    # loss = 20 * iou_loss + 30 * iou_loss1 + 20 * iou_loss2 + 5 * l1_loss + 2 * cam_smth

                bar.set_description(f"{iters} " +
                                    f"Loss {loss.item():.2} " + 
                                    f"iou {iou_loss.item():.3f} " + 
                                    f"iou1 {iou_loss1.item():.3f} " + 
                                    f"iou2 {iou_loss2.item():.3f} " + 
                                    f"smt {cam_smth.item():.3f} " +
                                    f"L1 {l1_loss.item():.3f}")

                if abs(loss - pre_losss) < 1e-5: 
                    break

                pre_losss = loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # optimized extrinsics
            _lidar2cam = camera.get_extrinsic().detach().clone()
            _world2cam = (_lidar2cam[0] @ world2lidar).cpu().numpy()

            if index % 200 == 0:
                camera.set_cam_ex(_lidar2cam)
                print(_lidar2cam.cpu().numpy())
                dataset.save_pkl(args.overwrite)
            else:
                camera.reset_camera()
                pass

            if not args.no_render:
                save_results_to_img(img_bath_path, pc_base_path, frame, world2cam, _world2cam, K, dist)
            

            dataset.updata_pkl(frame['file_basename'], cam_pose=_world2cam)
        dataset.save_pkl(args.overwrite)

    except Exception as e:
        dataset.save_pkl(args.overwrite)
        import traceback
        traceback.print_exc()

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--root_folder", type=str, default=None,
                        help="xxx")
    
    parser.add_argument("--pkl_path", type=str, default=None, help="Path to the pkl file")

    parser.add_argument("--device", type=str, default='cuda', help="cpu/cuda")

    parser.add_argument('--index', type=int, default=None,
                        help='the index frame to be saved to a image')
    
    parser.add_argument('--start', type=int, default=0,
                        help='the index frame to be saved to a image')
    
    parser.add_argument('--end', type=int, default=np.inf,
                        help='the index frame to be saved to a image')
    
    parser.add_argument('--window', type=int, default=4,
                        help='the total frames used to optimize the extrinsics')
    
    parser.add_argument('--skip', type=int, default=1,
                        help='the skip between two frames used to optimize the extrinsics')
    
    parser.add_argument('--no_render', action='store_true')
    
    parser.add_argument('--overwrite', action='store_true',
                        help='whether to overwrite the pkl file for new cam_pose')
    
    parser.add_argument('--use_smpl', action='store_true',
                        help='whether to use smpl projection for optimization')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    main(args, args.device, args.window, args.skip, args.index)
