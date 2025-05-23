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

from pytorch3d.utils import cameras_from_opencv_projection
sys.path.append('.')
sys.path.append('..')

from wandb_tool import set_wandb, save_wandb, get_weight_loss

from tool_func import *
from losses import *

from smpl import SMPL_Layer, BODY_WEIGHT, BODY_PRIOR_WEIGHT, BODY_PARTS, SmplParams, load_body_models
from smpl import convert_to_6D_rot, rot6d_to_rotmat, rotation_matrix_to_axis_angle, rot6d_to_axis_angle

from utils import read_json_file, poses_to_vertices_torch, select_visible_points, multi_func

def get_valid_faces(valid_verts, th_faces):
    valid_verts = list(set(valid_verts))
    valid_verts = torch.tensor(valid_verts, dtype=torch.long, device=th_faces.device)  # Shape: (n)
    valid_faces_mask = torch.isin(th_faces, valid_verts)  # Shape: (num_faces, 3)
    valid_faces = valid_faces_mask.all(dim=-1)  # Shape: (num_faces, ), True if all 3 vertices are valid
    valid_face_indices = torch.nonzero(valid_faces).squeeze()  # Shape: (num_valid_faces, )

    return valid_verts, valid_face_indices.tolist()

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
        'hand':         convert_to_6D_rot(params['hand'][s:e].clone().view(-1, 3)),
        'hand_kpt':     params['hand_kpt'][s:e].clone(),
        'betas':        params['betas'][s:e].clone(),
        'sensor_traj':  params['sensor_traj'][s:e].clone(),
        'mask':         params['mask'][s:e].clone(),
        'cameras':      cameras
    }

def make_smpl_params(human_data, prefix=''):
    SMPL = SmplParams()
    if f'{prefix}pose' not in human_data or f'{prefix}hand_pose' not in human_data or f'{prefix}trans' not in human_data:
        prefix = ''
        print("opt params doesn't exist!!")

    SMPL.pose = human_data[f'{prefix}pose'].copy()           # (n, 24, 3)
    SMPL.trans = human_data[f'{prefix}trans'].copy()  # (n, 3) 
    if f'{prefix}hand_pose' in human_data:
        SMPL.hand_pose = human_data[f'{prefix}hand_pose'].copy().reshape(-1, 30, 3).astype(np.float32)    # (n, 30, 3)
    else:
        SMPL.hand_pose = np.zeros((len(human_data['trans']), 30, 3)).astype(np.float32)

    if 'beta' in human_data:
        SMPL.betas = np.array(human_data['beta']).reshape(1, -1).astype(np.float32)
    else:
        SMPL.betas = np.zeros(10).reshape(1, -1).astype(np.float32)

    if 'gender' in human_data:
        SMPL.gender = human_data['gender']
    
    return SMPL
    

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

        self.valid_face = []    # valid face positions of the SMPL faces 
        self.synced_data_file = ''
        self.logger = None
        self.mask   = None
        self.human_points = None
        self.ori_SMPL = None
        self.opt_SMPL = None

        self.stage = {
            "start": 0,        # iters 0-ort,      optimize translation only (contact loss)
            "cont": 20,        # iters cont-ort,   + m2p, sld, trans loss
            "ort": 50,         # iters ort->pose,  optimize +orientation (+rot, joints loss)
            "pose": 90,        # iters > pose,     optimize +pose (+pose, coll, pen, prior)
            "all_loss": 140,   # iters > all_loss, all loss are activated( + p2m)
            "proj_only": 250,   # iters > all_loss, all loss are activated( + p2m)
        }

    def set_args(self, args, opt_file=None, logger_file=None):
        print('Setting arguments...')

        self.root_folder   = args.root_folder
        self.window_frames = args.window_frames
        self.imu_smt_mode  = args.imu_smt_mode
        self.radius        = args.radius
        self.shoes_height  = args.shoes_height
        self.name          = args.name
        self.scene         = args.scene

        self.logger, logger_time, logger_file = loadLogger(os.path.join(
            self.root_folder, 'log'), work_file=logger_file, name=self.name)
        
        with open(args.mask, 'rb') as f:
            self.mask = pkl.load(f)
            self.logger.info(f'Load mask data in {args.mask}')

        with open(args.mano_file, 'rb') as f:
            self.mano = pkl.load(f)
            self.logger.info(f'Load mano data in {args.mano_file}')

        with open(args.kpt_file, 'rb') as f:
            self.det_dict = pkl.load(f)
            self.logger.info(f'Load hand kpt data in {args.kpt_file}')

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
        self.w['kpt_loss']   = args.wt_kpt_loss
        self.w['l2h']        = args.wt_sensor2head
        self.w['coll']       = args.wt_coll_loss
            
        self.w['cat_sld']    = args.cat_sliding_weight
        self.w['cat_rot']    = args.cat_rot_weight
        self.w['cat_pose']   = args.cat_pose_weight
        self.w['cat_jts']    = args.cat_joints_weight
        self.w['cat_trans']  = args.cat_trans_weight


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

    def update_pkl_data(self, opt_data_file, trans, pose, hand_pose, start, end):
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
        if len(pose.shape) >= 3:
            pose = pose.reshape(-1, 3, 3)
            pose = rotation_matrix_to_axis_angle(pose).reshape(-1, 72)
        if len(hand_pose.shape) >= 3:
            hand_pose = hand_pose.reshape(-1, 3, 3)
            hand_pose = rotation_matrix_to_axis_angle(hand_pose).reshape(-1, 30, 3)

        person_data = self.synced_data[self.person]
        if 'hand_pose' not in person_data:
            person_data['hand_pose'] = self.ori_SMPL.hand_pose.copy()
        if 'opt_pose' not in person_data:
            person_data['opt_pose']  = self.opt_SMPL.pose
            person_data['opt_trans'] = self.opt_SMPL.trans
            person_data['opt_hand_pose'] = self.opt_SMPL.hand_pose.copy()

        person_data['opt_pose'][start: end]  = pose.detach().cpu().numpy()
        person_data['opt_trans'][start: end] = trans.detach().cpu().numpy()
        person_data['opt_hand_pose'][start: end] = hand_pose.detach().cpu().numpy()

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

        human_data             = synced_data[person]
        camera_params          = synced_data[person]['cam_head']
        sensor_traj            = synced_data[person]['sensor_traj'].copy()
        self.data_length       = len(synced_data['frame_num'])
        # self.frame_time        = synced_data['device_ts']    
        self.sensor_frame_rate = dataset_params['lidar_framerate']

        # ==================== the params used to optimize ====================
        self.ori_SMPL = make_smpl_params(human_data)
        self.opt_SMPL = make_smpl_params(human_data, 'opt_')
        self.synced_imu_trans = human_data['mocap_trans'].copy()        # (n, 3)
        if 'T_sensor_head' in human_data:
            self.head2sensor = torch.tensor(human_data['T_sensor_head']).float()

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
            self.ori_SMPL.betas,
            self.ori_SMPL.gender)

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

        mask_tensor = mask_dict_to_tensor(self.mask, camera_params['w'], camera_params['h'])
        if self.mano is not None:
            self.ori_SMPL.hand_pose[start: end] = get_hand_pose(self.mano, len(mask_tensor))  
            self.opt_SMPL.hand_pose[start: end] = fill_and_smooth_mano_poses(self.ori_SMPL.hand_pose[start: end])
        hand_kpt, _ = get_hand_kpt(self.det_dict, threshold=0.5, enable_filling=False)
        self.empty_mask = create_distance_mask(camera_params['w'] //2, camera_params['h']//2)  # the corner of the image is always empty

        sensor_t  = np.array([np.eye(4)] * self.data_length)
        sensor_t[:, :3, :3] = R.from_quat(sensor_traj[:, 4:8]).as_matrix()
        sensor_t[:, :3, 3:] = sensor_traj[:, 1:4].reshape(-1, 3, 1)
        sensor_t = torch.from_numpy(sensor_t[start: end])
            
        mocap_trans_params = torch.from_numpy(self.synced_imu_trans[start: end])

        ori_params  = torch.from_numpy(self.opt_SMPL.pose[start: end])[:, :3]   # (B, 1*3 )
        pose_params = torch.from_numpy(self.opt_SMPL.pose[start: end])[:, 3:]   # (B, 23*3)
        hand_params = torch.from_numpy(self.opt_SMPL.hand_pose[start: end])     # (B, 30, 3)
        trans       = torch.from_numpy(self.opt_SMPL.trans[start: end])         # (B, 3)
        betas       = torch.from_numpy(self.opt_SMPL.betas).float().repeat(len(trans), 1)
        hand_kpt    = torch.from_numpy(hand_kpt)         # (B, 3)
        
        trans.requires_grad              = False
        sensor_t.requires_grad           = False
        mocap_trans_params.requires_grad = False
        ori_params.requires_grad         = False
        pose_params.requires_grad        = False
        mask_tensor.requires_grad        = False
        self.head2sensor.requires_grad   = False

        self.smpl_layer = SMPL_Layer(gender=self.opt_SMPL.gender)
        self.body_model = load_body_models(gender=self.opt_SMPL.gender)
        self.valid_face = get_valid_faces(BODY_PARTS['arms'] + BODY_PARTS['hands'], self.smpl_layer.th_faces)
        self.valid_face_l = get_valid_faces(BODY_PARTS['left_arm'], self.smpl_layer.th_faces)
        self.valid_face_r = get_valid_faces(BODY_PARTS['right_arm'], self.smpl_layer.th_faces)

        if self.is_cuda:
            self.smpl_layer.cuda()
            self.body_model.cuda()
            mask_tensor        = mask_tensor.type(torch.FloatTensor).cuda()
            self.head2sensor   = self.head2sensor.type(torch.FloatTensor).cuda()
            #sensor_ori_params =sensor_ori_params.type(torch.FloatTensor).cuda()
            trans              = trans.type(torch.FloatTensor).cuda()
            sensor_t           = sensor_t.type(torch.FloatTensor).cuda()
            mocap_trans_params = mocap_trans_params.type(torch.FloatTensor).cuda()
            ori_params         = ori_params.type(torch.FloatTensor).cuda()
            pose_params        = pose_params.type(torch.FloatTensor).cuda()
            betas              = betas.type(torch.FloatTensor).cuda()
            hand_params        = hand_params.type(torch.FloatTensor).cuda()
            hand_kpt           = hand_kpt.type(torch.FloatTensor).cuda()

        #sensor_ori_params
        return {'trans': trans,   # translation from the camera
                'mocap_trans': mocap_trans_params,  # translation from the imu
                'ori': ori_params,
                'pose': pose_params,
                'hand': hand_params,
                'betas': betas,
                'hand_kpt': hand_kpt,
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
                smpl_rots, seg_params['trans'], betas=self.ori_SMPL.betas, gender=self.ori_SMPL.gender)
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

        # smpl_rots  = torch.cat([rot6d_to_rotmat(opt_params['ori']).view(-1, 1, 9),
        #                          rot6d_to_rotmat(opt_params['pose']).view(-1, 23, 9)], dim=1)

        # smpl_verts, joints, global_rots = self.smpl_layer(
        #     smpl_rots, opt_params['trans'], self.betas)
        
        body_pose_world = self.body_model(root_orient = rot6d_to_rotmat(opt_params['ori']).view(-1, 1 * 9), 
                                          pose_body = rot6d_to_rotmat(opt_params['pose']).view(-1, 23 * 9)[:, :21*9],
                                          trans=opt_params['trans'],
                                          pose_hand=rot6d_to_rotmat(opt_params['hand']).view(-1, 30 * 9),
                                          betas=opt_params['betas'])
        smpl_verts = body_pose_world.v
        joints = body_pose_world.Jtr

        sum_loss, print_str, loss_chart = 0, [], {}

        def add_loss_item(ll, weight, category, is_concat=False):
            return get_weight_loss(ll, weight, category, print_str, loss_dict, indexes, loss_chart, is_concat)

        if iters % 10 == 0 and iters < self.stage['all_loss']:
            self.contact_info = get_contacinfo(foot_states, jump_list, scene_grids, smpl_verts)

        # =============  0. re-projecting term =====================
        # if self.w['mask_loss'] > 0 and iters >= self.stage['all_loss']:
        #     ll = cam_loss(opt_params['mask'][:, :1], smpl_verts, self.smpl_layer.th_faces, 
        #                   cameras=opt_params['cameras'], empty_mask=self.empty_mask, face_indices=self.valid_face_l)
        #     sum_loss += add_loss_item(ll, self.w['mask_loss'], 'lmask')
        #     ll = cam_loss(opt_params['mask'][:, 1:2], smpl_verts, self.smpl_layer.th_faces, 
        #                   cameras=opt_params['cameras'], empty_mask=self.empty_mask, face_indices=self.valid_face_r)
        #     sum_loss += add_loss_item(ll, self.w['mask_loss'], 'rmask')

        if self.w['kpt_loss'] > 0 and iters >= self.stage['pose']:
            ll = reprojection_loss(joints[:, 20:22], opt_params['hand_kpt'][:, 2:4], opt_params['cameras'], max_margin=100)
            sum_loss += add_loss_item(ll, self.w['kpt_loss'] * 0.05, 'kpt_hand')
            ll = reprojection_loss(joints[:, 34:37], opt_params['hand_kpt'][:, 6:9], opt_params['cameras'], max_margin=100)
            sum_loss += add_loss_item(ll, self.w['kpt_loss'], 'kpt_lt') # left thumb
            ll = reprojection_loss(joints[:, 22:25], opt_params['hand_kpt'][:, 9:12], opt_params['cameras'], max_margin=100)
            sum_loss += add_loss_item(ll, self.w['kpt_loss'], 'kpt_lif') # left index finger
            ll = reprojection_loss(joints[:, 49:52], opt_params['hand_kpt'][:, 21:24], opt_params['cameras'], max_margin=100)
            sum_loss += add_loss_item(ll, self.w['kpt_loss'], 'kpt_rt') # right thumb
            ll = reprojection_loss(joints[:, 37:40], opt_params['hand_kpt'][:, 24:27], opt_params['cameras'], max_margin=100)
            sum_loss += add_loss_item(ll, self.w['kpt_loss'], 'kpt_rif') # right index finger
            # ll = reprojection_loss(joints[:, 18:20], opt_params['hand_kpt'][:, 0:2], opt_params['cameras'], max_margin=30, thresh_ratio=0.7)
            # sum_loss += add_loss_item(ll, self.w['kpt_loss'] * 0.2, 'kpt_el')

        # =============  1. scene-aware terms =====================
        # foot contact loss
        if self.w['ft_cont'] > 0 and iters >= self.stage['start'] and iters < self.stage['proj_only']:
            ll   = contact_constraint(smpl_verts, self.contact_info, self.shoes_height)
            loss = add_loss_item(ll, self.w['ft_cont'], 'cont')
            sum_loss += loss

        # body-scene collision loss
        if self.w['coll'] > 0 and iters >= self.stage['start'] and iters < self.stage['proj_only']:
            sum_loss += add_loss_item(foot_collision(smpl_verts,
                                      self.contact_info), 
                                      self.w['coll'], 'coll')
        # =============  2. scene-aware terms =====================
                
        # sensor_center_to_head CD loss
        if self.w['l2h'] > 0 and self.person == 'first_person' and iters < self.stage['proj_only']:
            ll = (opt_params['sensor_traj'] @ self.head2sensor)[:, :3, 3] - joints[:, 15]
            loss = torch.abs(ll).sum(-1)
            loss = add_loss_item([torch.abs(ll).sum(-1), 
                                  np.arange(end - start).tolist()], 
                                  self.w['l2h'], 'sensor')

            if self.w['l2h'] > 0:
                sum_loss += loss

        # =============  3. self-constraint loss =====================
        # self penetration loss
        if self.w['pen_loss'] > 0 and iters >= self.stage['pose']:
            loss = add_loss_item(collision_loss(smpl_verts, self.smpl_layer.th_faces),
                                 self.w['pen_loss'], 'pen')

            if iters >= self.stage['pose'] and self.w['pen_loss'] > 0:
                sum_loss += loss

        # pose prior (from IMU) loss
        if self.w['pose_prior'] > 0 and iters >= self.stage['pose'] and iters < self.stage['proj_only']:
            loss   = torch.sum(torch.abs(opt_params['pose'] - convert_to_6D_rot(
                init_params['pose'][start:end].view(-1, 3))).view(-1, 23, 6), dim=-1)
            
            # loss = torch.mean(loss, dim=-1)
            loss   = (loss @ torch.from_numpy(BODY_PRIOR_WEIGHT[1:]).to(loss.device)) / 23

            # weight = 1 if 'first' in self.person else 0.97 ** (iters - self.stage['pose'])
            weight = 0.99 ** (iters - self.stage['pose'])
            loss   = add_loss_item(
                [loss, np.arange(end - start).tolist()], self.w['pose_prior'] * weight, 'prior')

            if loss and self.w['pose_prior'] > 0:
                sum_loss += loss

        # foot sliding loss
        if self.w['ft_sld'] > 0 and iters < self.stage['proj_only']:
            ll = sliding_constraint(smpl_verts, 
                                    foot_states, 
                                    jump_list, lfoot_move, rfoot_move)
            loss = add_loss_item(ll, self.w['ft_sld'], 'sld')

            if iters > self.stage['cont'] and self.w['ft_sld'] > 0:
                sum_loss += loss
        # =============  self-constraint loss =====================

        # =============  4. temporal smoothness loss =====================
        # translation smooth loss
        if self.w['trans_smth'] > 0 and iters < self.stage['proj_only']:
            loss = trans_imu_smooth(
                opt_params['trans'].squeeze(1),
                jump_list,
                self.imu_smt_mode,
                0.02/self.sensor_frame_rate)

            loss = add_loss_item(loss, self.w['trans_smth'], 'trans')

            if iters > self.stage['cont'] and self.w['trans_smth'] > 0:
                sum_loss += loss

        #  body joints trans smoothness
        if self.w['jts_smth'] > 0 and iters >= self.stage['ort'] and iters < self.stage['proj_only']:
            loss = joints_smooth(joints, self.imu_smt_mode, BODY_WEIGHT[1:])
            loss = add_loss_item(
                [loss, np.arange(2, end - start).tolist()], self.w['jts_smth'], 'jts')
            sum_loss += loss

        # global rotation smoothness
        if self.w['rot_smth'] > 0 and iters >= self.stage['ort'] and iters < self.stage['proj_only']:
            # loss = joint_orient_error(ori_params[1:], ori_params[:-1])
            loss = torch.mean(
                torch.abs(opt_params['ori'][0:-1] - opt_params['ori'][1:]), dim=-1)
            loss = add_loss_item(
                [loss, np.arange(1, end - start).tolist()], self.w['rot_smth'], 'rot')

            if iters >= self.stage['ort'] and self.w['rot_smth'] > 0:
                sum_loss += loss

        # body pose rotation smoothness
        if self.w['rot_smth'] > 0 and iters >= self.stage['ort'] and iters < self.stage['proj_only']:
            # loss = joint_orient_error(ori_params[1:], ori_params[:-1])
            loss = torch.abs(opt_params['pose'].view(-1, 23, 6)
                             [0:-1]-opt_params['pose'].view(-1, 23, 6)[1:]).sum(-1).mean(-1)
            loss = add_loss_item(
                [loss, np.arange(1, end - start).tolist()], self.w['rot_smth'], 'pose')

            if iters > self.stage['cont'] and self.w['rot_smth'] > 0:
                sum_loss += loss

        # hand pose rotation smoothness
        if self.w['rot_smth'] > 0 and iters >= self.stage['ort'] and iters < self.stage['proj_only']:
            # loss = joint_orient_error(ori_params[1:], ori_params[:-1])
            loss = torch.abs(opt_params['hand'].view(-1, 30, 6)
                             [0:-1]-opt_params['hand'].view(-1, 30, 6)[1:]).sum(-1).mean(-1)
            loss = add_loss_item(
                [loss, np.arange(1, end - start).tolist()], self.w['rot_smth'], 'hpose')

            if iters > self.stage['cont'] and self.w['rot_smth'] > 0:
                sum_loss += loss
        # =============  temporal smoothness loss =====================

        # concancatation loss between two optmization segments
        if sub_seg > 0:
            # concancatation translation loss
            if self.w['cat_trans'] > 0 and iters < self.stage['proj_only']:
                acc = opt_params['trans'][0] - 2 * \
                    self.pre_sensor_t[1] + self.pre_sensor_t[0]
                loss1 = torch.nn.functional.relu(torch.norm(acc) - 1e-4)
                acc = opt_params['trans'][1] - 2 * opt_params['trans'][0] + self.pre_sensor_t[1]
                loss2 = torch.nn.functional.relu(torch.norm(acc) - 1e-4)
                # trans = sensor_t[0] - self.pre_sensor_t[1]
                # loss = torch.nn.functional.relu(torch.norm(trans) - 1e-4)
                loss = add_loss_item(
                    [[loss1, loss2], [0, 1]], self.w['cat_trans'], 'trans', True)

                if loss and iters > self.stage['cont']:
                    sum_loss += loss

            # concancatation sliding loss
            if self.w['cat_sld'] > 0 and iters < self.stage['proj_only']:
                init_verts = torch.cat((self.pre_verts[-1:], smpl_verts[0:1]))

                loss, _ = sliding_constraint(
                    init_verts,
                    self.foot_states[start - 1:start + 1],
                    jump_list,
                    self.lfoot_move[start - 1: start + 1],
                    self.rfoot_move[start-1:start+1])

                loss = add_loss_item(
                    [loss, [0]], self.w['cat_sld'], 'sld', True)

                if loss and iters > self.stage['cont']:
                    sum_loss += loss

            # concancatation rotation smoothness loss
            if self.w['cat_rot'] > 0 and iters >= self.stage['ort'] and iters < self.stage['proj_only']:
                loss = torch.mean(
                    torch.abs(opt_params['ori'][:1] - self.pre_ori[-1:]), dim=-1)
                loss = add_loss_item(
                    [loss, [0]], self.w['cat_rot'], 'rot', True)
                if loss and iters > self.stage['cont']:
                    sum_loss += loss

            # concancatation pose smoothness loss
            if self.w['cat_pose'] > 0 and iters >= self.stage['ort'] and iters < self.stage['proj_only']:
                loss = torch.abs(
                    opt_params['pose'][:1*23] - self.pre_pose[-1*23:]).sum(-1).mean()
                loss = add_loss_item(
                    [[loss], [0]], self.w['cat_pose'], 'pose', True)

                if loss and iters > self.stage['cont']:
                    sum_loss += loss

            # concancatation joints smoothness loss
            if self.w['cat_jts'] > 0 and iters >= self.stage['ort'] and iters < self.stage['proj_only']:
                loss = joints_smooth(torch.cat(
                    [joints[:2, ], self.pre_joints[-2:]], dim=0), self.imu_smt_mode, BODY_WEIGHT[1:])
                loss = add_loss_item(
                    [loss, [0, 1]], self.w['cat_jts'], 'jts', True)

                if iters > self.stage['cont'] and self.w['cat_jts'] > 0:
                    sum_loss += loss

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
                if iters == 0:  # Optimize global translation
                    optimizer = get_optmizer('trans', 
                                             [opt_params['trans']], 
                                            #   [opt_params['hand'],
                                            #    opt_params['ori'],
                                            #   opt_params['pose']], 
                                              self.learn_rate)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
                    
                elif iters == self.stage['ort']: # Now optimize global orientation
                    optimizer = get_optmizer('trans / orit',
                                             [opt_params['trans'], 
                                              opt_params['ori']],
                                             self.learn_rate)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

                elif iters == self.stage['pose']: # Now optimize full SMPL pose
                    optimizer = get_optmizer('trans / orit / pose',
                                             [opt_params['trans'], 
                                              opt_params['ori'], 
                                              opt_params['hand'], 
                                              opt_params['pose']],
                                             self.learn_rate)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

                elif iters == self.stage['all_loss']: # Now optimize the pose with all losses
                    optimizer = get_optmizer('trans / orit / pose  with all loss functions', 
                                             [opt_params['trans'], 
                                              opt_params['ori'], 
                                              opt_params['hand'], 
                                              opt_params['pose']], 
                                             self.learn_rate)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
                    
                elif iters == self.stage['proj_only']: # Now optimize the pose with all losses
                    optimizer = get_optmizer('pose only, for reprojection term only', 
                                            [opt_params['pose'],
                                              opt_params['ori'],
                                             opt_params['hand']], 
                                            self.learn_rate)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
                
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
                    optimizer.zero_grad()  # Reset gradients
                    scaler.scale(sum_loss).backward()  # Compute gradients
                    
                    # Unscale before updating optimizer
                    scaler.unscale_(optimizer) 

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    current_loss = sum_loss.item()

                    if abs(pre_loss - current_loss) < 1e-7:
                        count += 1
                    else:
                        count = 0
                    # if count >= 5:  # Adjust learning rate if loss stagnates
                        # prev_lr = optimizer.param_groups[0]['lr']
                        # for param_group in optimizer.param_groups:
                        #     param_group['lr'] *= 0.8  # Custom strategy (halving the learning rate)
                        # new_lr = optimizer.param_groups[0]['lr']
                        # print(f"Iteration {iters} - Adjusting LR: {prev_lr} -> {new_lr}")
                        # count = 0  # 重置 count，避免频繁触发

                    pre_loss = current_loss
                else:
                    count = 0
                
                info = f'{iters} {info} Loss {sum_loss:.3f} Time {(time.time() - time_start):.1f}s'

                if iters in self.stage.values():
                    self.logger.info(info)
                else:
                    self.logger.debug(info)

                if check_nan(ori=opt_params['ori'], 
                             pose=opt_params['pose'], 
                             hand=opt_params['hand'], 
                             trans=opt_params['trans']):
                    count = 100
                else:
                    params['trans'] = opt_params['trans'].clone().detach()
                    params['mocap'] = opt_params['mocap_trans'].clone().detach()
                    params['ori']   = opt_params['ori'].clone().detach()
                    params['pose']  = opt_params['pose'].clone().detach()
                    params['hand']  = opt_params['hand'].clone().detach()

                iters += 1

                if count > 5 or iters >= self.iterations:

                    save_wandb(raw_loss,
                               init_loss_chart,
                               loss_chart,
                               frameid_start,
                               end-start,
                               self.person)

                    smpl_rots = torch.cat([rot6d_to_rotmat(params['ori']).view(-1, 1, 9),
                                            rot6d_to_rotmat(params['pose']).view(-1, 23, 9)], dim=1)
                    out_hand_pose = rot6d_to_rotmat(params['hand']).view(-1, 30, 9)

                    self.pre_sensor_t = params['trans'][-2:]
                    self.pre_ori      = params['ori'][-2:]
                    self.pre_pose     = params['pose'][-2*23:]
                    # self.pre_verts, self.pre_joints, _ \
                    #     = self.smpl_layer(smpl_rots[-2:], self.pre_sensor_t, opt_params['betas'][0].clone().detach())
                    
                    bp = self.body_model(root_orient = rot6d_to_rotmat(params['ori']).view(-1, 1*9)[-2:], 
                                          pose_body = rot6d_to_rotmat(params['pose']).view(-1, 23 * 9)[-2:, :21*9],
                                          trans=self.pre_sensor_t,
                                          pose_hand=out_hand_pose.view(-1, 30 * 9)[-2:],
                                          betas=opt_params['betas'][-2:].clone().detach())
                    self.pre_verts = bp.v
                    self.pre_joints = bp.Jtr

                    self.logger.info(info)

                    break

            loss_dict['time'] += [f'{time.time() - time_start:.1f}']

            delta_trans, _ = log_dict(self.logger,
                                      loss_dict,
                                      params['trans'],
                                      init_params['trans'][start:end],
                                      rot6d_to_axis_angle(params['ori']),
                                      init_params['ori'][start:end])

            self.update_pkl_data(opt_data_file, params['trans'], smpl_rots, out_hand_pose, start + self.opt_start, end + self.opt_start)

            self.logger.info('================================================================')

        if self.person == 'first_person':
            try:
                loss_trans = cal_global_trans(
                    self.ori_SMPL.pose, self.ori_SMPL.trans)
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
        parser.add_argument("--window_frames",       type=int, default=250,
                            help="window of the frame to be used")

        # Optimization parameters - global
        parser.add_argument("--iterations",     type=int,   default=250)
        parser.add_argument("--learn_rate",     type=float, default=0.005)

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
        parser.add_argument("--wt_mask_loss",   type=float, default=100,
                            help= "body-scene collision loss weight")
        parser.add_argument("--wt_kpt_loss",   type=float, default=1,
                            help= "body-scene collision loss weight")
        parser.add_argument("--wt_pen_loss",    type=float, default=300, 
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
                        help="the path of the optimization pkl file")

    parser.add_argument('--logger_file', type=str, default=None, 
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
    
    parser.add_argument("--mano_file", type=str, default='/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd/recording_head/mano_imgs_1049_1990.pkl', 
                        help="Path to the mask file")
    
    parser.add_argument("--kpt_file", type=str, default='/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd/recording_head/kpts_imgs_1049_1990.pkl', 
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
