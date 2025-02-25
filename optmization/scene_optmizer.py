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

sys.path.append('.')
sys.path.append('..')

from wandb_tool import set_wandb, save_wandb, get_weight_loss

from tool_func import loadLogger, load_scene_for_opt, crop_scene, get_lidar_to_head as get_sensor_to_head
from tool_func import check_nan, log_dict, set_foot_states, cal_global_trans

from losses import *

from smpl import SMPL_Layer, BODY_WEIGHT, BODY_PRIOR_WEIGHT
from smpl import convert_to_6D_rot, rot6d_to_rotmat, rotation_matrix_to_axis_angle, rot6d_to_axis_angle

from utils import read_json_file, poses_to_vertices_torch, select_visible_points, multi_func

def get_opt_params(s:int, e:int, params:list, delta_trans=None):
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

    return {
        'trans':        params['trans'][s:e].clone() + delta_trans,
        'mocap_trans':  params['mocap_trans'][s:e].clone(),
        'ori':          convert_to_6D_rot(params['ori'][s:e].clone()),
        'pose':         convert_to_6D_rot(params['pose'][s:e].clone().view(-1, 3)),
        'sensor_traj':  params['sensor_traj'][s:e].clone(),
    }

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

        self.stage = {
            "start": 0,        # iters 0-ort,      optimize translation only (contact loss)
            "cont": 20,        # iters cont-ort,   + m2p, sld, trans loss
            "ort": 50,         # iters ort->pose,  optimize +orientation (+rot, joints loss)
            "pose": 90,        # iters > pose,     optimize +pose (+pose, coll, pen, prior)
            "all_loss": 140,   # iters > all_loss, all loss are activated( + p2m)
        }

    def set_args(self, args, opt_file=None, logger_file=None):
        print('Setting arguments...')

        self.sensor_pos = 15 if args.sensor_pos == 'Head' and self.person.lower(
        ) == 'first_person' else 0
        self.root_folder   = args.root_folder
        self.window_frames = args.window_frames
        self.imu_smt_mode  = args.imu_smt_mode
        self.radius        = args.radius
        self.shoes_height  = args.shoes_height
        self.name          = args.name
        self.gender        = args.gender
        self.scene         = args.scene

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

    def update_pkl_data(self, opt_data_file, sensor_t, mocap_rots, start, end):
        """
        `update_pkl_data` updates the `synced_data` dictionary with the optimized pose and translation
        parameters, and then saves the updated dictionary to a pickle file

        Args:
          sensor_t: the translation parameters of thesensor
          mocap_rots: the optimized mocap rotations
          start: the start index of the data to be updated
          end: the index of the last frame to be optimized
        """
        self.logger.info(f"Updated in: {os.path.basename(opt_data_file)}")

        # update original data
        self.synced_trans[start: end] = sensor_t.detach().cpu().numpy()
        if len(mocap_rots.shape) >= 3:
            mocap_rots = mocap_rots.reshape(-1, 3, 3)
            mocap_rots = rotation_matrix_to_axis_angle(mocap_rots).reshape(-1, 72)

        self.synced_smpl_pose[start: end] = mocap_rots.detach().cpu().numpy()

        person_data = self.synced_data[self.person]

        if 'opt_pose' not in person_data:
            person_data['opt_pose']  = self.synced_smpl_pose
            person_data['opt_trans'] = self.synced_trans
        else:
            person_data['opt_pose'][start: end]  = mocap_rots.detach().cpu().numpy()
            person_data['opt_trans'][start: end] = sensor_t.detach().cpu().numpy()

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
        self.frame_id          = np.array(synced_data['frame_num'])
        self.data_length       = len(self.frame_id)
        self.frame_time        = synced_data['device_ts']    
        self.sensor_frame_rate = dataset_params['lidar_framerate']
        self.sensor_traj       = synced_data['first_person']['lidar_traj'].copy()

        # ==================== the params used to optimize ====================
        if 'manual_pose' in human_data:
            self.synced_smpl_pose = human_data['manual_pose'].copy()    # (n, 24, 3)
            self.logger.info(f'manual_pose is detected {self.person}')
        else:
            self.synced_smpl_pose = human_data['pose'].copy()           # (n, 24, 3)
        self.synced_imu_trans = human_data['mocap_trans'].copy()        # (n, 3)

        if 'T_sensor_head' in human_data:
            self.head2sensor = torch.tensor(human_data['T_sensor_head']).float()

        # Transformation drived from the sensor (camera of lidar)
        self.synced_trans = human_data['trans'].copy()  # (n, 3)   

        if 'beta' in human_data:
            self.betas = torch.from_numpy(np.array(human_data['beta'])).float()
        else:
            self.betas = torch.zeros(10).float()

        if 'gender' in human_data:
            self.gender = human_data['gender']

        # ============= define the start and end frames to optimize =============
        self.opt_start  = int(max(args.opt_start - self.frame_id[0], 0))
        self.opt_end    = args.opt_end - args.opt_start + self.opt_start
        # self.opt_end   += self.frame_id[0]
        # self.opt_start += self.frame_id[0]
        
        if self.opt_end < 0 or self.opt_end >= self.data_length:
            self.opt_end = self.data_length
        s, e = self.opt_start, self.opt_end
        # =======================================================================

        init_params = self.get_init_params(s, e)

        self.foot_states, \
        self.lfoot_move, \
        self.rfoot_move = set_foot_states(
            human_data['pose'][s:e],
            human_data['mocap_trans'][s:e].copy(),
            1/self.sensor_frame_rate,
            self.betas,
            self.gender)

        self.logger.info('[Init] Mocap pose, mocap trans andsensor trajectory ready.')

        try:
            # pc = o3d.geometry.PointCloud()
            self.human_points = {}
            id_list = self.frame_id.tolist()[s:e]
            for i, p in enumerate(human_data['point_frame']):
                if p in id_list:
                    self.human_points[id_list.index(p)] = human_data['point_clouds'][i]
                    
        except:
            self.logger.warning(f'[Warning]: No human points data in {self.person}')
            self.human_points = None

        self.scene_ground = load_scene_for_opt(self.scene)
        self.sub_segment_idx, self.jump_list = self.divide_traj(person, dataset_params)
        self.synced_data = synced_data

        # ============= record the arguments =============
        self.logger.info(f"Optimizing from frame [{self.opt_start + self.frame_id[0]:.0f}] to " + 
                         f"[{min(self.opt_end + self.frame_id[0], self.frame_id[-1]):.0f}]\n")
        
        return init_params

    def get_init_params(self, start, end):
        """
        It takes in the start and end indices of the data, 
        and returns thesensor and mocap data in the form
        of torch tensors

        Args:
          start: the index of the first frame to be used for training
          end: the end index of the data
        """
        if start > self.data_length:
            self.logger.error('======================================================')
            self.logger.error('Start idx is larger than data lenght\n')
            self.logger.error('======================================================')
            exit(0)

        if end > self.data_length:
            end = self.data_length

        #sensor_ori_params = torch.from_numpy(self.synced_trans[start: end, 4:8])
        trans = torch.from_numpy(self.synced_trans[start: end])

        sensor_t  = np.array([np.eye(4)] * self.data_length)
        sensor_t[:, :3, :3] = R.from_quat(self.sensor_traj[:, 4:8]).as_matrix()
        sensor_t[:, :3, 3:] = self.sensor_traj[:, 1:4].reshape(-1, 3, 1)
        sensor_t = torch.from_numpy(sensor_t[start: end])

        mocap_trans_params = torch.from_numpy(
            self.synced_imu_trans[start: end])

        smpl_params = torch.from_numpy(self.synced_smpl_pose[start: end])
        ori_params  = smpl_params[:, :3]
        pose_params = smpl_params[:, 3:]

        #sensor_ori_params.requires_grad = False
        trans.requires_grad           = False
        sensor_t.requires_grad           = False
        # smpl_params.requires_grad      = False
        mocap_trans_params.requires_grad = False
        ori_params.requires_grad         = False
        pose_params.requires_grad        = False
        self.betas.requires_grad         = False
        self.head2sensor.requires_grad   = False

        self.smpl_layer = SMPL_Layer(gender=self.gender)

        if self.is_cuda:
            self.smpl_layer.cuda()
            self.betas         = self.betas.unsqueeze(0).type(torch.FloatTensor).cuda()
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
                'sensor_traj':sensor_t}

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

        seg_params = get_opt_params(start, end, params)
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

        mocap_rots = torch.cat([rot6d_to_rotmat(seg_params['ori']).view(-1, 1, 9),
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
            loss_trans = cal_global_trans(mocap_rots, trans, is_cuda=True)
            contac_loss, contac_num   = contact_constraint(verts, contact_info, self.shoes_height)
            sliding_loss, sliding_num = sliding_constraint(verts, foot_states, jump_list, lfoot_move, rfoot_move)

            contac_num  = [n + self.frame_id[0] +
                           self.opt_start for n in contac_num]
            sliding_num = [n + self.frame_id[0] +
                           self.opt_start for n in sliding_num]

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
                mocap_rots, seg_params['trans'], betas=self.betas, gender=self.gender)
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


        return init_chart

    def set_losses(self,
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

        mocap_rots  = torch.cat([rot6d_to_rotmat(opt_params['ori']).view(-1, 1, 9),
                                 rot6d_to_rotmat(opt_params['pose']).view(-1, 23, 9)], dim=1)

        # todo: 改成 limbs和orientation可以求导，orientation仅绕Z轴方向求导
        smpl_verts, joints, global_rots = self.smpl_layer(
            mocap_rots, opt_params['trans'], self.betas)

        sum_loss, print_str, loss_chart = 0, [], {}

        def add_loss_item(ll, weight, category, is_concat=False):
            return get_weight_loss(ll, weight, category, print_str, loss_dict, indexes, loss_chart, is_concat)

        if iters % 10 == 0 and iters < self.stage['all_loss']:
            self.contact_info = get_contacinfo(foot_states, jump_list, scene_grids, smpl_verts)

        # =============  0. re-projecting term =====================

        # =============  1. scene-aware terms =====================
        # foot contact loss
        if self.w['ft_cont'] >= 0 and iters >= self.stage['start']:
            ll   = contact_constraint(smpl_verts, self.contact_info, self.shoes_height)
            loss = add_loss_item(ll, self.w['ft_cont'], 'cont')
            sum_loss += loss

        # body-scene collision loss
        if self.w['coll'] >= 0 and iters >= self.stage['start']:
            sum_loss += add_loss_item(foot_collision(smpl_verts,
                                      self.contact_info), 
                                      self.w['coll'], 'coll')
        # =============  scene-aware terms =====================
                
        # sensor_center_to_head CD loss
        if self.w['l2h'] >= 0 and self.person == 'first_person':
            ll = (opt_params['sensor_traj'] @ self.head2sensor)[:, :3, 3] - joints[:, 15]
            loss = torch.abs(ll).sum(-1)
            loss = add_loss_item([torch.abs(ll).sum(-1), 
                                  np.arange(end - start).tolist()], 
                                  self.w['l2h'], 'sensor')

            if self.w['l2h'] > 0:
                sum_loss += loss
        # =============  (mesh to point) / (sensor to body) loss =====================

        # =============  3. self-constraint loss =====================
        # too slow
        # if self.w['pen_loss'] >= 0 and iters >= self.stage['pose']:
            
        #     if iters == self.stage['pose'] or iters == self.stage['all_loss']:
        #         self.pen_loss.configure(smpl_verts)
                
        #     ll = self.pen_loss(smpl_verts)
        #     loss = add_loss_item(ll, self.w['pen_loss'], 'pen')

        #     if iters > self.stage['pose'] and self.w['pen_loss'] > 0:
        #         sum_loss += loss

        # self penetration loss
        if self.w['pen_loss'] >= 0 and iters >= self.stage['pose']:
            loss = add_loss_item(collision_loss(smpl_verts, 
                                                self.smpl_layer.th_faces),
                                 self.w['pen_loss'], 'pen')

            if iters >= self.stage['pose'] and self.w['pen_loss'] > 0:
                sum_loss += loss

        # pose prior (from IMU) loss
        if self.w['pose_prior'] >= 0 and iters >= self.stage['pose']:
            loss   = torch.sum(torch.abs(opt_params['pose'] - convert_to_6D_rot(
                init_params['pose'][start:end].view(-1, 3))).view(-1, 23, 6), dim=-1)
            
            # loss = torch.mean(loss, dim=-1)
            loss   = (loss @ torch.from_numpy(BODY_PRIOR_WEIGHT[1:]).to(loss.device)) / 23

            weight = 1 if 'first' in self.person else 0.97 ** (iters - self.stage['pose'])
            loss   = add_loss_item(
                [loss, np.arange(end - start).tolist()], self.w['pose_prior'] * weight, 'prior')

            if loss and self.w['pose_prior'] > 0:
                sum_loss += loss

        # foot sliding loss
        if self.w['ft_sld'] >= 0:
            ll = sliding_constraint(smpl_verts, 
                                    foot_states, 
                                    jump_list, lfoot_move, rfoot_move)
            loss = add_loss_item(ll, self.w['ft_sld'], 'sld')

            if iters > self.stage['cont'] and self.w['ft_sld'] > 0:
                sum_loss += loss
        # =============  self-constraint loss =====================

        # =============  4. temporal smoothness loss =====================
        # translation smooth loss
        if self.w['trans_smth'] >= 0:
            loss = trans_imu_smooth(
                opt_params['trans'].squeeze(1),
                jump_list,
                self.imu_smt_mode,
                0.02/self.sensor_frame_rate)

            loss = add_loss_item(loss, self.w['trans_smth'], 'trans')

            if iters > self.stage['cont'] and self.w['trans_smth'] > 0:
                sum_loss += loss

        #  body joints trans smoothness
        if self.w['jts_smth'] >= 0 and iters >= self.stage['ort']:
            loss = joints_smooth(joints, self.imu_smt_mode, BODY_WEIGHT[1:])
            loss = add_loss_item(
                [loss, np.arange(2, end - start).tolist()], self.w['jts_smth'], 'jts')
            sum_loss += loss

        # global rotation smoothness
        if self.w['rot_smth'] >= 0 and iters >= self.stage['ort']:
            # loss = joint_orient_error(ori_params[1:], ori_params[:-1])
            loss = torch.mean(
                torch.abs(opt_params['ori'][0:-1] - opt_params['ori'][1:]), dim=-1)
            loss = add_loss_item(
                [loss, np.arange(1, end - start).tolist()], self.w['rot_smth'], 'rot')

            if iters >= self.stage['ort'] and self.w['rot_smth'] > 0:
                sum_loss += loss

        # body pose rotation smoothness
        if self.w['rot_smth'] >= 0 and iters >= self.stage['ort']:
            # loss = joint_orient_error(ori_params[1:], ori_params[:-1])
            loss = torch.abs(opt_params['pose'].view(-1, 23, 6)
                             [0:-1]-opt_params['pose'].view(-1, 23, 6)[1:]).sum(-1).mean(-1)
            loss = add_loss_item(
                [loss, np.arange(1, end - start).tolist()], self.w['rot_smth'], 'pose')

            if iters > self.stage['cont'] and self.w['rot_smth'] > 0:
                sum_loss += loss
        # =============  temporal smoothness loss =====================

        # concancatation loss between two optmization segments
        if sub_seg > 0:
            # concancatation translation loss
            if self.w['cat_trans'] >= 0:
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
            if self.w['cat_sld'] >= 0:
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
            if self.w['cat_rot'] >= 0 and iters >= self.stage['ort']:
                loss = torch.mean(
                    torch.abs(opt_params['ori'][:1] - self.pre_ori[-1:]), dim=-1)
                loss = add_loss_item(
                    [loss, [0]], self.w['cat_rot'], 'rot', True)
                if loss and iters > self.stage['cont']:
                    sum_loss += loss

            # concancatation pose smoothness loss
            if self.w['cat_pose'] >= 0 and iters >= self.stage['ort']:
                loss = torch.abs(
                    opt_params['pose'][:1*23] - self.pre_pose[-1*23:]).sum(-1).mean()
                loss = add_loss_item(
                    [[loss], [0]], self.w['cat_pose'], 'pose', True)

                if loss and iters > self.stage['cont']:
                    sum_loss += loss

            # concancatation joints smoothness loss
            if self.w['cat_jts'] >= 0 and iters >= self.stage['ort']:
                loss = joints_smooth(torch.cat(
                    [joints[:2], self.pre_joints[-2:]], dim=0), self.imu_smt_mode, BODY_WEIGHT[1:])
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
            frameid_start = int(start + self.opt_start + self.frame_id[0])
            frameid_end   = int(end + self.opt_start + self.frame_id[0])

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
                                             self.learn_rate)
                    
                elif iters == self.stage['ort']: # Now optimize global orientation
                    optimizer = get_optmizer('trans / orit',
                                             [opt_params['trans'], 
                                              opt_params['ori']],
                                             self.learn_rate)

                elif iters == self.stage['pose']: # Now optimize full SMPL pose
                    optimizer = get_optmizer('trans / orit / pose',
                                             [opt_params['trans'], 
                                              opt_params['ori'], 
                                              opt_params['pose']],
                                             self.learn_rate)

                elif iters == self.stage['all_loss']: # Now optimize the pose with all losses
                    optimizer = get_optmizer('trans / orit / pose  with all loss functions', 
                                             [opt_params['trans'], 
                                              opt_params['ori'], 
                                              opt_params['pose']], 
                                             self.learn_rate)
                
                sum_loss, info, loss_chart = self.set_losses(
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
                    if abs(pre_loss - sum_loss.item()) < 1e-6:
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

                    mocap_rots = torch.cat([rot6d_to_rotmat(params['ori']).view(-1, 1, 9),
                                            rot6d_to_rotmat(params['pose']).view(-1, 23, 9)], dim=1)

                    self.pre_sensor_t = params['trans'][-2:]
                    self.pre_ori     = params['ori'][-2:]
                    self.pre_pose    = params['pose'][-2*23:]
                    self.pre_verts, self.pre_joints, _ \
                        = self.smpl_layer(mocap_rots[-2:], self.pre_sensor_t, self.betas)

                    self.logger.info(info)

                    break

            loss_dict['time'] += [f'{time.time() - time_start:.1f}']

            delta_trans, _ = log_dict(self.logger,
                                      loss_dict,
                                      params['trans'],
                                      init_params['trans'][start:end],
                                      rot6d_to_axis_angle(params['ori']),
                                      init_params['ori'][start:end])

            self.update_pkl_data(opt_data_file, opt_params['trans'], mocap_rots, start, end)

            self.logger.info(
                '================================================================')

        if self.person == 'first_person':
            try:
                loss_trans = cal_global_trans(
                    self.synced_smpl_pose, self.synced_trans)
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
        parser.add_argument("--window_frames",       type=int, default=500,
                            help="window of the frame to be used")

        # Optimization parameters - global
        parser.add_argument("--iterations",     type=int,   default=200)
        parser.add_argument("--learn_rate",     type=float, default=0.005)

        parser.add_argument("--wt_ft_sliding",  type=float, default=400)
        parser.add_argument("--wt_ft_cont",     type=float, default=400)
        parser.add_argument("--wt_mesh2point",  type=float, default=1000)
        parser.add_argument("--wt_point2mesh",  type=float, default=1000)
        parser.add_argument("--wt_rot_smth",    type=float, default=300)
        parser.add_argument("--wt_trans_smth",  type=float, default=100)
        parser.add_argument("--wt_joints_smth", type=float, default=100)
        parser.add_argument("--wt_pose_prior",  type=float, default=500)
        parser.add_argument("--wt_sensor2head",  type=float, default=500)
        parser.add_argument("--wt_coll_loss",   type=float, default=200,
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

    parser.add_argument("-OS", "--opt_start", type=int, default=0,
                        help='Optimization start frame in the original trajectory')

    parser.add_argument("-OE", "--opt_end", type=int, default=-1,
                        help='Optimization end frame in the original trajectory')
    
    parser.add_argument("--mask", type=str, default='/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd/recording_head/mask_dict.pkl', 
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
        optimizer = Optimizer(person='second_person')
        opt_file, _ = optimizer.set_args(args, opt_file, logger_file)
        optimizer.run(opt_file)
