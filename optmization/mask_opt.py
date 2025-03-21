################################################################################
# File: /scene_optmizer.py                                                     #
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


import os
import time
import sys
from glob import glob
from copy import deepcopy
import pickle as pkl

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.amp.grad_scaler import GradScaler
import torch.nn.functional as F

from kaolin.render.mesh import camera as opengl_camera
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.utils import cameras_from_opencv_projection
sys.path.append('.')
sys.path.append('..')

from wandb_tool import set_wandb, save_wandb, get_weight_loss

from tool_func import loadLogger, load_scene_for_opt, crop_scene, get_lidar_to_head as get_sensor_to_head
from tool_func import check_nan, log_dict, set_foot_states, cal_global_trans

from losses import *

from smpl import SMPL_Layer, BODY_WEIGHT, BODY_PRIOR_WEIGHT, BODY_PARTS
from smpl import convert_to_6D_rot, rot6d_to_rotmat, rotation_matrix_to_axis_angle, rot6d_to_axis_angle

from utils import read_json_file, poses_to_vertices_torch, select_visible_points, multi_func

def get_valid_faces(valid_verts, th_faces):

    valid_verts = torch.tensor(valid_verts, dtype=torch.long, device=th_faces.device)  # Shape: (n)
    valid_faces_mask = torch.isin(th_faces, valid_verts)  # Shape: (num_faces, 3)
    valid_faces = valid_faces_mask.all(dim=-1)  # Shape: (num_faces, ), True if all 3 vertices are valid
    valid_face_indices = torch.nonzero(valid_faces).squeeze()  # Shape: (num_valid_faces, )

    return valid_face_indices

def get_opt_params(s:int, e:int, params:list, delta_trans=None, return_cameras=True):
    """
    It takes in the start and end indices of a sequence, and returns the correspondingsensor
    translation (or from point cloud), mocap translation, mocap orientation,sensor orientation, andsensor trajectory
    
    Args:
        s: start index
        e: end of the sequence
        delta_trans: the translation offset between thesensor and the mocap.
    
    Returns:
       trans, trans, orientation, pose,sensor_traj
    """
    # before_head = convert_to_6D_rot(self.before_head[s:e].clone().view(-1, 3))
    # head_rot    = convert_to_6D_rot(self.head_rot[s:e].clone())
    # after_head  = convert_to_6D_rot(self.after_head[s:e].clone().view(-1, 3))
    
    delta_trans = 0 if delta_trans is None else delta_trans

    if return_cameras:
        # intrinsics = torch.tensor(camera_params['intrinsic']).float()   # (4, )
        ex = torch.tensor(params['cameras']['extrinsic'][s: e]).float()   # (B, 4, 4)
        image_size  = torch.tensor([[params['cameras']['h'], params['cameras']['w']]] * len(ex))
        f = torch.tensor([params['cameras']['intrinsic'][:2]] * len(ex)).float()
        c = torch.tensor([params['cameras']['intrinsic'][2:]] * len(ex)).float()
        K = torch.tensor([[
            [params['cameras']['intrinsic'][0], 0, params['cameras']['intrinsic'][2]],
            [0, params['cameras']['intrinsic'][1], params['cameras']['intrinsic'][3]],
            [0, 0, 1]
            ]] * len(ex))
        cameras = cameras_from_opencv_projection(R=ex[:, :3, :3].cuda(), tvec=ex[:, :3, 3].cuda(), 
                                                 image_size=image_size.cuda(), 
                                                 camera_matrix=K.cuda())
        
        # M_flip = torch.tensor([[
        #     [1., 0, 0, 0],
        #     [0, -1, 0, 0],   
        #     [0, 0, -1, 0],
        #     [0, 0, 0, 1]
        # ]]) 
        # cameras2 = opengl_camera.Camera.from_args(view_matrix=M_flip @ ex, 
        #                                          focal_x=f[0, 0], focal_y=f[1, 0],
        #                                         #  x0=c[0] - w/2, y0=c[1] - h/2,
        #                                          width=1024, height=1024).cuda()
        # cameras2.requires_grad_(False)
    else:
        cameras = None
        # cameras2 = None

    return {
        'trans':        params['trans'][s:e].clone() + delta_trans,
        'mocap_trans':  params['mocap_trans'][s:e].clone(),
        'ori':          convert_to_6D_rot(params['ori'][s:e].clone()),
        'pose':         convert_to_6D_rot(params['pose'][s:e].clone().view(-1, 3)),
        'sensor_traj':  params['sensor_traj'][s:e].clone(),
        'mask':         params['mask'][s:e].clone(),
        'cameras':      cameras
    }

class SmplParams:
    # Shared parameters across all instances
    _pose = np.array([])  
    _trans = np.array([])
    _betas = np.array([])
    _gender = 'Neutral'  # Default gender

    @classmethod
    def _ensure_list_or_array(cls, value, name):
        """Ensures the value is a list or NumPy array."""
        if isinstance(value, (list, np.ndarray)):
            return np.array(value) if isinstance(value, list) else value
        raise TypeError(f"{name} must be a list or NumPy array")

    @property
    def pose(self):
        return SmplParams._pose

    @pose.setter
    def pose(self, value):
        SmplParams._pose = self._ensure_list_or_array(value, "pose")

    @property
    def trans(self):
        return SmplParams._trans

    @trans.setter
    def trans(self, value):
        SmplParams._trans = self._ensure_list_or_array(value, "trans")

    @property
    def betas(self):
        return SmplParams._betas

    @betas.setter
    def betas(self, value):
        SmplParams._betas = self._ensure_list_or_array(value, "betas")

    @property
    def gender(self):
        return SmplParams._gender

    @gender.setter
    def gender(self, value):
        if value not in {'neutral', 'male', 'female'}:
            raise ValueError("gender must be 'neutral', 'male', or 'female'")
        SmplParams._gender = value

    def __repr__(self):
        return (f"SmplParams(pose={self.pose.tolist()}, trans={self.trans.tolist()}, "
                f"betas={self.betas.tolist()}, gender='{self.gender}')")

class Optimizer():
    def __init__(self,
                 is_cuda  = True,
                 person   = 'first_person'):
        """
        Args:
          is_cuda: Whether to use GPU or not. Defaults to True
          person: The person whose data we are optimizing. Defaults to first_person
        """

        self.is_cuda = is_cuda
        self.person  = person
        self.w       = {}      # weights dict

        self.sensor_pos  = 15   # head joint position
        self.data_length = 0
        self.sensor_frame_rate = 30.0 # frame rate, default is 30.0 fps

        self.SMPL   = SmplParams()
        self.valid_face = []    # valid face positions of the SMPL faces 
        self.synced_data_file = ''
        self.logger = None
        self.mask   = None
        self.human_points = None

        self.stage = {
            "start": 0,        # iters 0-ort,      optimize translation only (contact loss)
            "cont": 20,        # iters cont-ort,   + m2p, sld, trans loss
            "ort": 50,         # iters ort->pose,  optimize +orientation (+rot, joints loss)
            "pose": 90,        # iters > pose,     optimize +pose (+pose, coll, pen, prior)
            "all_loss": 140,   # iters > all_loss, all loss are activated( + p2m)
            "proj_only": 200,   # iters > all_loss, all loss are activated( + p2m)
        }

    def set_args(self, args, opt_file=None, logger_file=None):
        print('Setting arguments...')

        self.root_folder   = args.root_folder
        self.window_frames = args.window_frames
        self.imu_smt_mode  = args.imu_smt_mode
        self.radius        = args.radius
        self.shoes_height  = args.shoes_height
        self.name          = args.name
        self.SMPL.gender   = args.gender
        self.scene         = args.scene
        self.mask_path     = args.mask

        # optimization arguments
        self.learn_rate = args.learn_rate
        self.iterations = args.iterations

        self.w['ft_sld']     = args.wt_ft_sliding
        self.w['ft_cont']    = args.wt_ft_cont
        self.w['m2p']        = args.wt_mesh2point
        self.w['p2m']        = args.wt_point2mesh
        self.w['trans_smth'] = args.wt_trans_smth
        self.w['rot_smth']   = args.wt_rot_smth
        self.w['pose_prior'] = args.wt_pose_prior
        self.w['jts_smth']   = args.wt_joints_smth
        self.w['pen_loss']   = args.wt_pen_loss
        self.w['mask_loss']  = args.wt_mask_loss
        self.w['l2h']        = args.wt_sensor2head
        self.w['coll']       = args.wt_coll_loss
            
        self.w['cat_sld']    = args.cat_sliding_weight
        self.w['cat_rot']    = args.cat_rot_weight
        self.w['cat_pose']   = args.cat_pose_weight
        self.w['cat_jts']    = args.cat_joints_weight
        self.w['cat_trans']  = args.cat_trans_weight

        self.logger, logger_time, logger_file = loadLogger(os.path.join(
            self.root_folder, 'log'), work_file=logger_file, name=self.name)

        if opt_file is not None:
            self.synced_data_file = opt_file
        else:
            try:
                self.synced_data_file = glob(os.path.join(
                    self.root_folder, 'synced_data') + '/*_param_manual.pkl')[0]
            except:
                self.synced_data_file = glob(os.path.join(
                    self.root_folder, 'synced_data') + '/*_param.pkl')[0]
                
            opt_file = os.path.join(
                self.root_folder, 'log', f'{logger_time}_{self.name}.pkl')
        
        # load scene for optimization
        if self.scene is None:
            try:
                self.scene = glob(self.root_folder + '/synced_data/*frames.ply')[0]
            except:
                self.logger.error('No default scene file!!!')
                exit(0)

        elif not os.path.exists(self.scene):
            self.scene = os.path.join(self.root_folder, 'sensor_data', self.scene)

        if not os.path.exists(self.scene):
            self.logger.error('No scene file!!!')
            exit(0)

        self.logger.debug("Loss weight")
        for k, v in self.w.items():
            self.logger.debug(f"{k.ljust(12)}: {v:.1f}".rstrip('0').rstrip('.'))

        return opt_file, logger_file

    def update_pkl_data(self, opt_data_file, trans, pose, start, end):
        """
        `update_pkl_data` updates the `synced_data` dictionary with the optimized pose and translation
        parameters, and then saves the updated dictionary to a pickle file

        Args:
          trans: the optimized translations
          pose: the optimized pose parameters
          start: the start index of the data to be updated
          end: the index of the last frame to be optimized
        """
        self.logger.info(f"Updated in: {os.path.basename(opt_data_file)}")

        # update original data
        self.SMPL.trans[start: end] = trans.detach().cpu().numpy()
        if len(pose.shape) >= 3:
            pose = pose.reshape(-1, 3, 3)
            pose = rotation_matrix_to_axis_angle(pose).reshape(-1, 72)

        self.SMPL.pose[start: end] = pose.detach().cpu().numpy()

        person_data = self.synced_data[self.person]

        if 'opt_pose' not in person_data:
            person_data['opt_pose']  = self.SMPL.pose
            person_data['opt_trans'] = self.SMPL.trans
        else:
            person_data['opt_pose'][start: end]  = pose.detach().cpu().numpy()
            person_data['opt_trans'][start: end] = trans.detach().cpu().numpy()

        with open(opt_data_file, 'wb') as f:
            pkl.dump(self.synced_data, f)

    def initialize_data(self, person='first person'):
        """
        It loads the data from the pickle file, sets the start and end frames for optimization, sets the
       sensor offset, creates the SMPL layer, and sets the foot states

        Args:
          person: the person you want to optimize.
        """

        dataset_params = read_json_file(os.path.join(self.root_folder, 'dataset_params.json'))

        with open(self.synced_data_file, 'rb') as f:
            synced_data = pkl.load(f)
            self.logger.info(f'Load data in {self.synced_data_file}')

        with open(self.mask_path, 'rb') as f:
            self.mask = pkl.load(f)
            self.logger.info(f'Load mask data in {self.mask_path}')

        human_data             = synced_data[person]
        camera_params          = synced_data[person]['cam_head']
        sensor_traj            = synced_data[person]['lidar_traj'].copy()
        self.data_length       = len(synced_data['frame_num'])
        # self.frame_time        = synced_data['device_ts']    
        self.sensor_frame_rate = dataset_params['lidar_framerate']

        # ==================== the params used to optimize ====================
        if 'manual_pose' in human_data:
            self.SMPL.pose = human_data['manual_pose'].copy()    # (n, 24, 3)
            self.logger.info(f'manual_pose is detected {self.person}')
        else:
            self.SMPL.pose = human_data['pose'].copy()           # (n, 24, 3)
        self.synced_imu_trans = human_data['mocap_trans'].copy()        # (n, 3)

        if 'T_sensor_head' in human_data:
            self.head2sensor = torch.tensor(human_data['T_sensor_head']).float()

        # Transformation from the sensor (camera of lidar)
        self.SMPL.trans = human_data['trans'].copy()  # (n, 3)   

        if 'beta' in human_data:
            self.betas = torch.from_numpy(np.array(human_data['beta'])).float()
        else:
            self.betas = torch.zeros(10).float()

        if 'gender' in human_data:
            self.SMPL.gender = human_data['gender']

        # ============= define the start and end frames to optimize =============
        self.opt_start  = args.opt_start
        self.opt_end    = args.opt_end
        
        if self.opt_end < 0 or self.opt_end >= self.data_length:
            self.opt_end = self.data_length
        # =======================================================================

        init_params = self.get_init_params(self.opt_start, self.opt_end, sensor_traj, camera_params)

        self.foot_states, \
        self.lfoot_move, \
        self.rfoot_move = set_foot_states(
            human_data['pose'][self.opt_start: self.opt_end],
            human_data['mocap_trans'][self.opt_start: self.opt_end].copy(),
            1/self.sensor_frame_rate,
            self.betas,
            self.SMPL.gender)

        self.logger.info('[Init] Mocap pose, mocap trans andsensor trajectory ready.')

        self.scene_ground = load_scene_for_opt(self.scene)
        self.sub_segment_idx, self.jump_list = self.divide_traj(person, dataset_params)
        self.synced_data = synced_data

        # ============= record the arguments =============
        self.logger.info(f'Optimizing from index [{self.opt_start}] to [{self.opt_end}]\n')
        
        return init_params

    def get_init_params(self, start, end, sensor_traj, camera_params=None):
        """
        It takes in the start and end indices of the data, 
        and returns thesensor and mocap data in the form
        of torch tensors

        Args:
          start: the index of the first frame to be used for training
          end: the end index of the data,
          camera_params: the camera parameters
        """
        if start > self.data_length:
            self.logger.error('======================================================')
            self.logger.error('Start idx is larger than data lenght\n')
            self.logger.error('======================================================')
            exit(0)

        if end > self.data_length:
            end = self.data_length

        #sensor_ori_params = torch.from_numpy(self.SMPL.trans[start: end, 4:8])
        trans = torch.from_numpy(self.SMPL.trans[start: end])

        sensor_t  = np.array([np.eye(4)] * self.data_length)
        sensor_t[:, :3, :3] = R.from_quat(sensor_traj[:, 4:8]).as_matrix()
        sensor_t[:, :3, 3:] = sensor_traj[:, 1:4].reshape(-1, 3, 1)
        sensor_t = torch.from_numpy(sensor_t[start: end])
            
        mocap_trans_params = torch.from_numpy(self.synced_imu_trans[start: end])

        ori_params  = torch.from_numpy(self.SMPL.pose[start: end])[:, :3]
        pose_params = torch.from_numpy(self.SMPL.pose[start: end])[:, 3:]
        
        def mask_dict_to_tensor(mask_dict, w, h, scale=0.5):
            mask = torch.zeros((len(mask_dict), 2, h, w)).type(torch.bool)
            for idx, (_, v) in enumerate(mask_dict.items()):
                if 1 in v:  # left hand
                    mask[idx, 0] = torch.from_numpy(v[1])
                if 2 in v:  # right hand
                    mask[idx, 1] = torch.from_numpy(v[2])
            return F.interpolate(mask.float(), scale_factor=scale, mode="nearest")

        mask_tensor = mask_dict_to_tensor(self.mask, camera_params['w'], camera_params['h'])

        trans.requires_grad              = False
        sensor_t.requires_grad           = False
        mocap_trans_params.requires_grad = False
        ori_params.requires_grad         = False
        pose_params.requires_grad        = False
        self.betas.requires_grad         = False
        mask_tensor.requires_grad        = False
        self.head2sensor.requires_grad   = False

        self.smpl_layer = SMPL_Layer(gender=self.SMPL.gender)
        self.valid_face = get_valid_faces(BODY_PARTS['arms'] + BODY_PARTS['hands'], self.smpl_layer.th_faces).tolist()

        if self.is_cuda:
            self.smpl_layer.cuda()
            self.betas         = self.betas.unsqueeze(0).type(torch.FloatTensor).cuda()
            mask_tensor        = mask_tensor.type(torch.FloatTensor).cuda()
            self.head2sensor   = self.head2sensor.type(torch.FloatTensor).cuda()
            #sensor_ori_params =sensor_ori_params.type(torch.FloatTensor).cuda()
            trans              = trans.type(torch.FloatTensor).cuda()
            sensor_t           = sensor_t.type(torch.FloatTensor).cuda()
            mocap_trans_params = mocap_trans_params.type(torch.FloatTensor).cuda()
            ori_params         = ori_params.type(torch.FloatTensor).cuda()
            pose_params        = pose_params.type(torch.FloatTensor).cuda()

        #sensor_ori_params
        return {'trans': trans,   # translation from the camera
                'mocap_trans': mocap_trans_params,  # translation from the imu
                'ori': ori_params,
                'pose': pose_params,
                'sensor_traj':sensor_t,
                'mask': mask_tensor,
                'cameras': {'w': camera_params['w'],
                            'h': camera_params['h'],
                            'intrinsic': camera_params['intrinsic'],
                            'extrinsic': camera_params['extrinsic'][start:end],
                            } if camera_params is not None else None
        }

    def divide_traj(self, person, dataset_params, skip=100):

        sensor_fr   = dataset_params['lidar_framerate']

        jump_list = []

        sub_segment_idx = np.arange(
            self.opt_start, self.opt_end, self.window_frames).tolist()
        
        if self.opt_end - sub_segment_idx[-1] < self.window_frames//3:
            sub_segment_idx[-1] = self.opt_end
        else:
            sub_segment_idx.append(self.opt_end)

        sub_segment_idx = [i - self.opt_start for i in sub_segment_idx]

        return sub_segment_idx, jump_list

    def cal_loss(self, params):
        """
        It takes in the predicted vertices, the predicted global transformation, and the type of loss (Sensor
        or MoCap) and returns the contact loss, the sliding loss, the number of frames that have contact
        loss, the number of frames that have sliding loss, and the localization loss
        
        Args:
          params: the parameters of the SMPL model
        
        Returns:
          The contact loss, the sliding loss, the number of frames that have contact loss, the number of
        frames that have sliding loss, and the localization loss
        """
        start = 0
        end   = self.opt_end - self.opt_start
        if self.human_points:
            self.vis_smpl_idx = [[] * (end - start)]

        seg_params = get_opt_params(start, end, params, return_cameras=False)
        init_chart = {}
        jump_list  = []
        [jump_list.append(jl - start) if jl >= start and jl <
         end else None for jl in self.jump_list]

        foot_states = self.foot_states[start:end]
        lfoot_move  = self.lfoot_move[start:end]
        rfoot_move  = self.rfoot_move[start:end]

        # ground = self.scene_ground

        ground = crop_scene(self.scene_ground,
                            seg_params['trans'].cpu().numpy(), self.radius)

        smpl_rots = torch.cat([rot6d_to_rotmat(seg_params['ori']).view(-1, 1, 9),
                                rot6d_to_rotmat(seg_params['pose']).view(-1, 23, 9)], dim=1)

        def process(verts, trans, ltype='SENSOR'):
            """
            > This function takes in the predicted vertices, the predicted global transformation, and the type
            of loss (SENSOR or MoCap) and returns the contact loss, the sliding loss, the number of frames that
            have contact loss, the number of frames that have sliding loss, and the localization loss

            Args:
              verts: the vertices of the mesh
              trans: the translation of thesensor in the world frame
              ltype: the type of loss, either 'SENSOR' or 'mocap'. Defaults tosensor
            """
            loss_trans = cal_global_trans(smpl_rots, trans, is_cuda=True)
            contac_loss, contac_num   = contact_constraint(verts, contact_info, self.shoes_height)
            sliding_loss, sliding_num = sliding_constraint(verts, foot_states, jump_list, lfoot_move, rfoot_move)

            contac_num  = [n + self.opt_start for n in contac_num]
            sliding_num = [n + self.opt_start for n in sliding_num]

            if ltype.lower() == 'sensor':
                if self.person == 'first_person':
                    self.logger.info(
                        f'Sensor to head: {1000*torch.norm(self.head2sensor[:3, 3]).item():.1f} (mm)')
                self.logger.info(f'Type\tContact\tSliding\tLocali')

            print_str = f'{ltype}\t' + \
                f'{1000 * torch.tensor(contac_loss).mean().item():.1f}\t' + \
                f'{1000 * torch.tensor(sliding_loss).mean().item():.1f}\t' + \
                f'{loss_trans:.1f}\t' + \
                f'| num frames: {len(contac_num)}/{len(sliding_num)}'

            self.logger.info(print_str)
            init_chart[f'{ltype.lower()}_contact'] = [[i.item()
                                                       for i in contac_loss], contac_num]
            init_chart[f'{ltype.lower()}_sliding'] = [[i.item()
                                                       for i in sliding_loss], sliding_num]

        with torch.no_grad():
            smpl_verts, joints, global_rots = poses_to_vertices_torch(
                smpl_rots, seg_params['trans'], betas=self.betas, gender=self.SMPL.gender)
            lowest_height = torch.min(smpl_verts[:20,:, -1], dim=-1)[0].mean() - self.shoes_height

            contact_info = get_contacinfo(
                foot_states, jump_list, ground, smpl_verts)

            #sensor
            process(smpl_verts, seg_params['trans'], 'Sensor')
            # Mocap
            smpl_verts += seg_params['mocap_trans'].unsqueeze(1) - seg_params['trans'].unsqueeze(1)
            process(smpl_verts, seg_params['mocap_trans'], 'IMU')

            del smpl_verts, joints, global_rots, contact_info

        self.logger.info('===============================================')

        del seg_params
        torch.cuda.empty_cache()

        return init_chart

    def get_losses(self,
                   scene_grids,
                   jump_list,
                   loss_dict,
                   indexes,
                   opt_params,
                   init_params):

        sub_seg, iters = indexes  # i-th segment of the data

        start, end  = self.sub_segment_idx[sub_seg: sub_seg+2]
        foot_states = self.foot_states[start: end]
        lfoot_move  = self.lfoot_move[start: end]
        rfoot_move  = self.rfoot_move[start: end]

        smpl_rots  = torch.cat([rot6d_to_rotmat(opt_params['ori']).view(-1, 1, 9),
                                 rot6d_to_rotmat(opt_params['pose']).view(-1, 23, 9)], dim=1)

        # todo: 改成 limbs和orientation可以求导，orientation仅绕Z轴方向求导
        smpl_verts, joints, global_rots = self.smpl_layer(
            smpl_rots, opt_params['trans'], self.betas)

        sum_loss, print_str, loss_chart = 0, [], {}

        def add_loss_item(ll, weight, category, is_concat=False):
            return get_weight_loss(ll, weight, category, print_str, loss_dict, indexes, loss_chart, is_concat)

        # =============  0. re-projecting term =====================
        if self.w['mask_loss'] > 0:
            ll = cam_loss(opt_params['mask'], smpl_verts, self.smpl_layer.th_faces, cameras=opt_params['cameras'], face_indices=self.valid_face)
            sum_loss += add_loss_item(ll, self.w['mask_loss'], 'mask')

        return sum_loss, ''.join(print_str), loss_chart

    def run(self, opt_data_file):
            
        init_params = self.initialize_data(self.person)

        raw_loss    = self.cal_loss(init_params)
        loss_dict   = {'start': [], 'end': [], 'time': []}
        delta_trans = None

        scaler = GradScaler("cuda")

        # Optmize all windows
        for i in range(len(self.sub_segment_idx) - 1):

            start, end    = self.sub_segment_idx[i:i+2]
            frameid_start = int(start + self.opt_start)    # to check!
            frameid_end   = int(end + self.opt_start)

            loss_dict['start'] += [frameid_start]
            loss_dict['end']   += [frameid_end]

            # define optimization params
            opt_params = get_opt_params(start, end, init_params, delta_trans)

            # find the jumping list in this window
            jump_list = []
            for jl in self.jump_list:
                if jl >= start and jl < end:
                    jump_list.append(jl - start)

            scene_grids = crop_scene(
                self.scene_ground, opt_params['trans'].cpu().numpy(), self.radius)

            self.logger.info(
                f'=========[Segments {i+1}/{len(self.sub_segment_idx) - 1}], ' + 
                f'frame {frameid_start:.0f} to {frameid_end - 1:.0f} unit (cm)=========')

            pre_loss   = -1
            params     = {}
            iters      = 0
            time_start = time.time()

            while True:
                if iters == 0:
                    optimizer = get_optmizer('pose only, for reprojection term only', 
                                            [
                                             opt_params['pose']], 
                                            self.learn_rate)
                # elif iters == self.stage['proj_only']: # Now optimize the pose with all losses
                #     optimizer = get_optmizer('trans / orit / pose  with all loss functions', 
                #                              [opt_params['trans'], 
                #                               opt_params['ori'], 
                #                               opt_params['pose']], 
                #                               self.learn_rate)
                
                sum_loss, info, loss_chart = self.get_losses(
                    scene_grids,
                    jump_list,
                    loss_dict,
                    [i, iters],
                    opt_params,
                    init_params)

                # Record the loss before optimization for visualization comparison
                init_loss_chart = deepcopy(loss_chart) if iters == 0 else init_loss_chart   

                if sum_loss > 0:
                    scaler.scale(sum_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    if abs(pre_loss - sum_loss.item()) < 1e-7:
                        count += 1
                    else:
                        count = 0
                    pre_loss = sum_loss.item()
                else:
                    count = 0
                
                info = f'{iters} {info} Loss {sum_loss:.3f} Time {(time.time() - time_start):.1f}s'

                if iters in self.stage.values():
                    self.logger.info(info)
                else:
                    self.logger.debug(info)

                if check_nan(ori=opt_params['ori'], 
                             pose=opt_params['pose'], 
                             trans=opt_params['trans']):
                    count = 100
                else:
                    params['trans'] = opt_params['trans'].clone().detach()
                    params['mocap'] = opt_params['mocap_trans'].clone().detach()
                    params['ori']   = opt_params['ori'].clone().detach()
                    params['pose']  = opt_params['pose'].clone().detach()

                iters += 1

                if count > 10 or iters >= self.iterations:

                    save_wandb(raw_loss,
                               init_loss_chart,
                               loss_chart,
                               frameid_start,
                               end-start,
                               self.person)

                    smpl_rots = torch.cat([rot6d_to_rotmat(params['ori']).view(-1, 1, 9),
                                            rot6d_to_rotmat(params['pose']).view(-1, 23, 9)], dim=1)

                    self.pre_sensor_t = params['trans'][-2:]
                    self.pre_ori     = params['ori'][-2:]
                    self.pre_pose    = params['pose'][-2*23:]
                    self.pre_verts, self.pre_joints, _ \
                        = self.smpl_layer(smpl_rots[-2:], self.pre_sensor_t, self.betas)

                    self.logger.info(info)

                    break

            loss_dict['time'] += [f'{time.time() - time_start:.1f}']

            delta_trans, _ = log_dict(self.logger,
                                      loss_dict,
                                      params['trans'],
                                      init_params['trans'][start:end],
                                      rot6d_to_axis_angle(params['ori']),
                                      init_params['ori'][start:end])

            self.update_pkl_data(opt_data_file, opt_params['trans'], smpl_rots, start + self.opt_start, end + self.opt_start)

            self.logger.info(
                '================================================================')

        if self.person == 'first_person':
            try:
                loss_trans = cal_global_trans(
                    self.SMPL.pose, self.SMPL.trans)
                self.logger.info(f'opt localization loss: {loss_trans:.3f}')
            except Exception as e:
                self.logger.warning(f'{e.args[0]}')


def config_parser(is_optimization=False):
    import configargparse
    parser = configargparse.ArgumentParser()

    # Experiment Setup
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    if is_optimization:
        # scene parameters
        parser.add_argument("--radius",              type=float, default=0.6)
        parser.add_argument("--shoes_height",        type=float, default=0.00)
        parser.add_argument("-PS", "--plane_thresh", type=float, default=0.01)
        parser.add_argument("--window_frames",       type=int, default=1,
                            help="window of the frame to be used")

        # Optimization parameters - global
        parser.add_argument("--iterations",     type=int,   default=250)
        parser.add_argument("--learn_rate",     type=float, default=0.0001)

        parser.add_argument("--wt_ft_sliding",  type=float, default=400)
        parser.add_argument("--wt_ft_cont",     type=float, default=400)
        parser.add_argument("--wt_mesh2point",  type=float, default=1000)
        parser.add_argument("--wt_point2mesh",  type=float, default=1000)
        parser.add_argument("--wt_rot_smth",    type=float, default=300)
        parser.add_argument("--wt_trans_smth",  type=float, default=100)
        parser.add_argument("--wt_joints_smth", type=float, default=100)
        parser.add_argument("--wt_pose_prior",  type=float, default=500)
        parser.add_argument("--wt_sensor2head", type=float, default=500)
        parser.add_argument("--wt_coll_loss",   type=float, default=200)
        parser.add_argument("--wt_mask_loss",   type=float, default=10,
                            help= "body-scene collision loss weight")
        parser.add_argument("--wt_pen_loss",    type=float, default=60, 
                            help= 'self-penetration loss')

        parser.add_argument("--imu_smt_mode",   type=str, default='XYZ', 
                            help='optmization mode use XY or XYZ')

        # Optimization parameters - connecting
        parser.add_argument("--cat_sliding_weight", type=float, default=300/100)
        parser.add_argument("--cat_trans_weight",   type=float, default=400/100)
        parser.add_argument("--cat_rot_weight",     type=float, default=400/100)
        parser.add_argument("--cat_pose_weight",    type=float, default=200/100)
        # cat_joints_weight must < cat_pose_weight/4
        parser.add_argument("--cat_joints_weight",  type=float, default=100/200)    

    return parser


if __name__ == '__main__':

    parser = config_parser(True)

    parser.add_argument('--root_folder', type=str, default="/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd", 
                        help="data folder's directory path")

    parser.add_argument('--scene', type=str, default=None, 
                        help="scene path for optimization")

    parser.add_argument('--sensor_pos', type=str, default='Head', 
                        help="the sensor to the nearest body position to the SMPL Model")

    parser.add_argument('--opt_file', type=str, default=None,
                        # default='/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd/log/2025-03-18T17:52:07__wandb.pkl', 
                        help="the path of the optimization pkl file")

    parser.add_argument('--logger_file', type=str, default=None,
                        # default='/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd/log/2025-03-18T17:52:07__wandb.log', 
                        help="the path of optimization logger file")

    parser.add_argument('--name', type=str, default='',
                        help="Specify the name of the optimization file")

    parser.add_argument('--gender', type=str, default='neutral',
                        help="gender of the optimization person")

    parser.add_argument("--wandb", action='store_true', 
                        help="Open wandb to store the losses")
    
    parser.add_argument("--debug", action='store_true', 
                        help="Use debug mode to close wandb")

    parser.add_argument('--offline', action='store_true',
                        help="Use offline mode to wandb")

    parser.add_argument("-OS", "--opt_start", type=int, default=1049,
                        help='Optimization start frame in the original trajectory')

    parser.add_argument("-OE", "--opt_end", type=int, default=1990,
                        help='Optimization end frame in the original trajectory')
    
    parser.add_argument("--mask", type=str, default='/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd/recording_head/mask_imgs_1049_1990.pkl', 
                        help="Path to the mask file")

    args = parser.parse_args()

    config = set_wandb(args)

    print('File path: ', args.root_folder)

    if args.opt_file is None or args.logger_file is None:
        print('====== Run first person optimization ======')
        optimizer = Optimizer(person='first_person')
        opt_file, logger_file = optimizer.set_args(args)
        optimizer.run(opt_file)
    else:
        opt_file = os.path.join(args.root_folder, 'log', args.opt_file)
        logger_file = os.path.join(args.root_folder, 'log', args.logger_file)

        print('====== Run second person optimization ======')
        optimizer = Optimizer(person='first_person')
        opt_file, _ = optimizer.set_args(args, opt_file, logger_file)
        optimizer.run(opt_file)
