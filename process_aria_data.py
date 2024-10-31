import sys
import os
from glob import glob
import pickle
from os.path import dirname, split, abspath
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import configargparse
from projectaria_tools.core.sophus import SE3

sys.path.append(dirname(split(abspath( __file__))[0]))

from nymeria.data_provider import NymeriaDataProvider
from nymeria.handeye import HandEyeSolver
from utils import poses_to_joints, mocap_to_smpl_axis, sync_lidar_mocap, poses_to_vertices_torch
from utils import save_json_file, read_json_file, load_csv_data, print_table, compute_similarity

field_fmts = ['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.3f']

def save_trajs(save_dir, rot, pos, mocap_id, comments='first', framerate=100):
    # lidar_file = os.path.join(save_dir, 'lidar_traj.txt')
    save_file = os.path.join(save_dir, f'{comments}_person_traj.txt')
    imu_to_world = np.array([[-1,0,0],[0,0,1],[0,1,0]])
    # xsense_to_world = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) 
    pos = pos[:, 0][:, [1,2,0]]
    t = pos @ imu_to_world.T   # (R * pos.T).T
    r = R.from_rotvec(rot[:, :3]).as_quat()
    time = np.array(mocap_id).reshape(-1, 1)/framerate    # -->second
    
    save_data = np.concatenate((np.array(mocap_id).reshape(-1, 1), t, r, time), axis=1)
    np.savetxt(save_file, save_data, fmt=field_fmts)

    print('Save traj in: ', save_file)

    return t

def finetune_first_person_data(synced_data_file, sensor_traj):
    """
    > It takes the head sensor trajectory and the parameter file, 
      and then it saves the LiDAR trajectory in the parameter file
    """
    
    with open(synced_data_file, "rb") as f:
        save_data = pickle.load(f)
    
    if 'first_person' in save_data and 'pose' in save_data['first_person']:
        fp_data = save_data['first_person']
        first_pose = fp_data['pose'].copy()
        mocap_tran = fp_data['mocap_trans'].copy()

        fp_data['lidar_traj'] = sensor_traj

        try:
            _, j, glo_rot = poses_to_vertices_torch(first_pose, mocap_tran)
        except:
            _, j, glo_rot = poses_to_vertices_torch(first_pose, mocap_tran, is_cuda=False)

        # save smpl head trajectory
        smpl_head_traj = np.concatenate((save_data['frame_num'].reshape(-1, 1), 
                                            j[:, 15]), axis=1)
        smpl_head_traj_path = os.path.join(os.path.dirname(synced_data_file), 'smpl_head_trajectory.txt')
        np.savetxt(smpl_head_traj_path, smpl_head_traj)
        print(f'Saved initial smpl head trajectory to {smpl_head_traj_path}')
        
        head_to_root = (j[:, 0] - j[:, 15]).cpu().numpy()

        ROT, _, _ = compute_similarity(j[:len(mocap_tran)//2, 15, :2].numpy(), 
                                       sensor_traj[:len(mocap_tran)//2, 1:3])

        delta_degree = np.linalg.norm(R.from_matrix(ROT).as_rotvec()) * 180 / np.pi
        print(f'[First person] {delta_degree:.1f}° around Z-axis from the mocap to LiDAR data.')

        scaled_trans = mocap_tran @ ROT.T
        scaled_trans = (scaled_trans - scaled_trans[0]) + (sensor_traj[0, 1:4] + head_to_root[0]) # move to the sensor start position

        first_pose[:, :3] = (R.from_matrix(ROT) * R.from_rotvec(first_pose[:, :3])).as_rotvec()

        fp_data['mocap_trans'] = scaled_trans
        fp_data['mocap_pose'] = first_pose.copy()

        # joints, verts = poses_to_joints(first_pose[:1], return_verts=True, is_cuda=False) 
        # feet_center = (joints[0, 7] + joints[0, 8])/2
        # feet_center[2] = verts[..., 2].min()
        # scaled_trans -= scaled_trans[0] + feet_center  # 

        # solve handeye (eye in the hand)
        handeye = HandEyeSolver(
            stride=2,
            smooth=False,
            skip=30 * 5,
            window=30 * 120,
        )
        
        aria_se3 = SE3.from_quat_and_translation(
            sensor_traj[:, 7], 
            sensor_traj[:, 4:7], 
            sensor_traj[:, 1:4]
        )
        
        smpl_head_se3 = SE3.from_quat_and_translation(
            R.from_matrix(ROT @ glo_rot[:, 15].numpy()).as_quat()[:, 3], 
            R.from_matrix(ROT @ glo_rot[:, 15].numpy()).as_quat()[:, :3], 
            scaled_trans
        )

        Ts_Hd_Hx: list[SE3] = handeye(
            T_Wa_A=aria_se3,
            T_Wb_B=smpl_head_se3,
        )
        print(f"from xsense (smpl head) to aria head:\n {Ts_Hd_Hx[0].to_matrix()}")

        head_to_aria = Ts_Hd_Hx[0].to_matrix()  # 4*4


        # head_world_rot = R.from_matrix(ROT @ glo_rot[:, 15].numpy())
        # root_world_rot = R.from_matrix(ROT @ glo_rot[:, 0].numpy())
        # new_root_rot = R.from_matrix(new_head_trans[:, :3, :3]) * head_world_rot.inv() * root_world_rot 
        # first_pose[:, :3] = new_root_rot.as_rotvec()
        new_head_trans = aria_se3.to_matrix() @ head_to_aria
        fp_data['trans'] = new_head_trans[:, :3, 3] + head_to_root
        fp_data['pose'] = first_pose
        fp_data['mocap_trans'] = (scaled_trans - scaled_trans[0]) + (new_head_trans[0, :3, 3] + head_to_root[0])

        fp_data['head_to_aria'] = head_to_aria.tolist()

    else:
        save_data['first_person'] = {'lidar_traj': sensor_traj}

    with open(synced_data_file, "wb") as f:
        pickle.dump(save_data, f)
        print(f"File saved in {synced_data_file}")

def make_sensor_params(sensor, gap_frame):
    info = f"{sensor['syncid'][gap_frame]} ({sensor['times'][sensor['syncid'][gap_frame]]}s) " if gap_frame>0 else gap_frame
    return {"Framerate"       : sensor['framerate'],
            "Total frames"    : len(sensor['times']),
            "Start frame"     : f"{sensor['syncid'][0]} ({sensor['times'][sensor['syncid'][0]]:.6f}s)",
            "- Keyframe"      : f"{sensor['keyid'][0]} ({sensor['keytime'][0]:.6f}s)",
            "- Gapframe"      : info,
            "- Keyframe2"     : f"{sensor['keyid'][-1]} ({sensor['keytime'][-1]:.6f}s)",
            "End frame"       : f"{sensor['syncid'][-1]} ({sensor['times'][sensor['syncid'][-1]]:.6f}s)",
            "Relative keytime": f"{sensor['keytime'][-1] - sensor['keytime'][0]:.6f}"}

def update_data(save_data, person, save_trans, save_pose):
    betas = np.array([0.0] * 10)
    if 'betas' in dataset_params[f'{person}_person']:
        betas = dataset_params[f'{person}_person']['betas']
    if 'gender' in dataset_params[f'{person}_person']:
        gender = dataset_params[f'{person}_person']['gender']
    else:
        gender = 'male'
    if f'{person}_person' not in save_data:
        save_data[f'{person}_person'] = {}

    save_data[f'{person}_person'].update({
                                'beta': betas, 
                                'gender': gender, 
                                'pose': save_pose, 
                                'mocap_trans': save_trans})

def save_sync_data(root_folder, head_traj, body_data, start=0, end=np.inf, gap_frame=-1):
    first_data  = True
    second_data = True

    json_path   = os.path.join(root_folder, 'dataset_params.json')

    # lidar_trajectory = glob(root_folder + '/*lidar_trajectory.txt')[0]
    save_dir = os.path.join(root_folder, 'synced_data')
    param_file = os.path.join(save_dir, 'humans_param.pkl')
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(json_path):
        dataset_params = read_json_file(json_path)
    else:
        dataset_params = {}
    sensor = {}
    mocap = {}

    sensor['times'] = head_traj[:, -1]
    sensor['framerate'] = 1 / np.diff(sensor['times']).mean()
    dataset_params['lidar_framerate'] = sensor['framerate']
    save_json_file(json_path, dataset_params)

    try:
        if os.path.exists(param_file):
            with open(param_file, "rb") as f:
                save_data = pickle.load(f)
            print(f"Synced data loaded in: {param_file}")
        else:
            save_data = {}
    except:
        save_data = {}

    def load_csv(search_str='*'):
        rot_csv = glob(root_folder + f'/mocap_data/*{search_str}_rot*.csv')[0]
        pos_csv = glob(root_folder + f'/mocap_data/*{search_str}_*pos.csv')[0]
        pos, rot, col = load_csv_data(rot_csv, pos_csv)
        mocap['times'] = rot[:, 0]
        return pos, rot, col
    
    try:
        imu_start, imu_end = body_data['index_span']
        first_pos = body_data['segment_tXYZ'].reshape(-1, 23, 3)[imu_start:imu_end]
        first_ori = body_data['segment_qWXYZ'][imu_start:imu_end, :4]       # global orientation in Xsense coordinations (w,x,y,z)
        first_rot = body_data['joint_angleEulerZXY'][imu_start:imu_end]

        dataset_params['mocap_framerate'] = int(body_data['frameRate'])

        mocap['framerate'] = dataset_params['mocap_framerate']
        mocap['times']     = ((body_data['timestamps_us'][imu_start:imu_end] + body_data['t_diff']/1e3)/1e6)
        # first_pos, first_rot, col_names = load_csv('first')
    except BaseException:
        print('=======================')
        print('No first person data!!')
        print('=======================')
        first_data = False
        
    try:
        second_pos, second_rot, col_names = load_csv('second')
    except:
        try:
            second_pos, second_rot, col_names = load_csv('*')
        except:
            print('=======================')
            print('No second person data!!')
            print('=======================')
            second_data = False

    if not first_data and not second_data:
        print("No mocap data in './mocap_data/'!!!")
        exit(0)
    

    def get_keyid(sensor: dict):
        keyid = []
        for kt in sensor['keytime']:
            keyid.append(np.where(abs(sensor['times'] - kt) < 0.85/sensor['framerate'])[0][-1])
        return keyid
    
    if True:
    
        sensor['syncid'], mocap['syncid'] = sync_lidar_mocap(sensor['times'], mocap['times'], 0.5/mocap['framerate'])
        if 'start' in dataset_params:
            start =  dataset_params['start']
        else:
            start = 0

        start = sensor['syncid'].index(start) if start in sensor['syncid'] else 0
        end   = sensor['syncid'].index(end) if end in sensor['syncid'] else len(sensor['syncid']) - 1

        sensor['syncid'] = sensor['syncid'][start:end+1]
        mocap['syncid'] = mocap['syncid'][start:end+1]
        
        sensor['keyid'] =  [sensor['syncid'][0]]
        mocap['keyid'] =  [mocap['syncid'][0]]
        
        sensor['keytime'] =  [sensor['times'][sensor['syncid'][0]]]
        mocap['keytime'] =  [mocap['times'][mocap['syncid'][0]]]

        save_data['frame_num'] = sensor['syncid']

        if sensor['syncid'][0] > 1000:
            print("Note that the starting frame for the LiDAR is > 1000.")
            print("It is recommended to manually set the starting frame number.")

    # 4. print the necessary information
    dataset_params['lidar'] = make_sensor_params(sensor, gap_frame)
    dataset_params['mocap'] = make_sensor_params(mocap, gap_frame)

    save_json_file(json_path, dataset_params)
    print_table("Synchronization Parameters", 
                ["Parameter", "LiDAR", "Mocap(IMU)"],
                [dataset_params['lidar'], dataset_params['mocap']])

    # start to sync all the data
    if first_data:
        if 'syncid' not in mocap:
            mocap['syncid'] = np.arange(len(first_rot))[1000:-1000:int(mocap['framerate']/sensor['framerate'])]
        sync_pose, _  = mocap_to_smpl_axis(first_rot[mocap['syncid']], 
                                           first_ori[mocap['syncid']],
                                           fix_orit = False, )
        # 2. save synced mocap trans
        trans = save_trajs(save_dir, sync_pose, first_pos[mocap['syncid']], mocap['syncid'], framerate=mocap['framerate'])

        j, v  = poses_to_joints(sync_pose[:1], return_verts=True, is_cuda=False) 
        feet_center    = (j[0, 7] + j[0, 8])/2
        feet_center[2] = v[..., 2].min()
        trans      -= trans[0] + feet_center

        # 3. save synced data
        update_data(save_data, 'first', trans, sync_pose)

    if second_data:
        fix_orit = True if not first_data else False

        sync_pose_2nd, delta_r = mocap_to_smpl_axis(second_rot[mocap['syncid']], 
                                           fix_orit=fix_orit,
                                           col_name=col_names)

        second_trans = save_trajs(save_dir, sync_pose_2nd, 
                                  second_pos[mocap['syncid']] @ delta_r.T, 
                                  mocap['syncid'], 'second', mocap['framerate'])
        update_data(save_data, 'second', second_trans, sync_pose_2nd)
        if not first_data:
            trans = second_trans
            sync_pose = sync_pose_2nd

    save_data['framerate'] = sensor['framerate']

    with open(param_file, "wb") as f:
        pickle.dump(save_data, f)
        print(f"File saved in {param_file}")
    
    return sync_pose, trans, sensor['syncid'], param_file   # type: ignore

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()

    parser.add_argument("--root_folder", type=str, 
                        default="D:\\Data\\20231222_s1_kenneth_fischer_act7_56uvqd",
                        help="The data's root directory")

    parser.add_argument("--traj_file", type=str, default='lidar_trajectory.txt')

    parser.add_argument("--params_file", type=str, default='dataset_params.json')

    parser.add_argument('-S', "--start_idx", type=int, default=0,
                        help='The start frame index in LiDAR for processing, specified when sychronization time is too late')

    parser.add_argument('-E', "--end_idx", type=int, default=np.inf,
                        help='The end frame index in LiDAR for processing, specified when sychronization time is too early.')

    parser.add_argument("-VS", "--voxel_size", type=float, default=0.06, 
                        help="The voxel filter parameter for TSDF fusion")

    parser.add_argument("--skip_frame", type=int, default=8, 
                        help='The everay n frame used for mapping')
    
    parser.add_argument('--tsdf', action='store_true',
                        help="Use VDB fusion to build the scene mesh")
    
    parser.add_argument('--sdf_trunc', type=float, default=0.25,
                        help="The trunction distance for SDF funtion")

    parser.add_argument('--sync', action='store_true', 
                        help='Synced all data and save a pkl based on the params_file')

    args, opts    = parser.parse_known_args()
    root_folder   = args.root_folder

    head_f        = glob(root_folder + '/*head')[0]
    traj_file     = os.path.join(head_f, 'mps', 'slam', 'closed_loop_trajectory.csv')

    params_file   = os.path.join(root_folder, args.params_file)

    lidar_dir     = os.path.join(root_folder, 'lidar_data', 'lidar_frames_rot')

    if os.path.exists(params_file):
        dataset_params = read_json_file(params_file)
        if 'start' in dataset_params:
            args.start_idx = dataset_params['start']
        if 'end' in dataset_params:
            args.end_idx   = dataset_params['end']

    # --------------------------------------------
    # 1. Load the aria_head trajectory.
    # --------------------------------------------
    columns_to_read = ['tracking_timestamp_us', 'utc_timestamp_ns', 
                        'tx_world_device', 'ty_world_device', 'tz_world_device', 
                        'qx_world_device', 'qw_world_device', 'qy_world_device', 'qz_world_device']
    
    nd_lodaer = NymeriaDataProvider(sequence_rootdir=Path(root_folder), trajectory_sample_fps=30, return_diff=True)

    aria_trajs = nd_lodaer.get_all_trajectories(return_time=True)

    for traj_name, traj in aria_trajs.items():
        print(f'{traj_name}: {len(traj)}')
        diff = np.diff(traj[:, -1])
        traj = traj[1:][diff>0]
        aria_trajs[traj_name] = np.concatenate((np.arange(len(traj)).reshape(-1, 1),
                                traj[:, [3, 7, 11]],
                                R.from_matrix(traj[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]].reshape(-1, 3, 3)).as_quat(),
                                traj[:, -1:]/1e9), axis=1)   # head_traj: num, x, y, z, qx, qy, qz, qw, device_time(s)
        print(f'{len(aria_trajs[traj_name])}')

    aria_points = nd_lodaer.get_all_pointclouds()
    head_traj = aria_trajs['recording_head']
    head_pc = aria_points['recording_head']
    for tag, pc in aria_points.items():
        save_path = os.path.join(root_folder, 'body', f'pc_{tag}.txt')
        np.savetxt(save_path, pc)
        
    # aria_poses = nd_lodaer.get_synced_poses()
    save_path = os.path.join(os.path.dirname(traj_file), 'head_trajectory.txt')
    np.savetxt(save_path, head_traj)
    print(f"Aria glassed Head trajectory saved in {save_path}")

    start = np.searchsorted(nd_lodaer.body_dp.xsens_data['timestamps_us'], nd_lodaer.timespan_ns[0]/1e3) + 240
    end = np.searchsorted(nd_lodaer.body_dp.xsens_data['timestamps_us'], nd_lodaer.timespan_ns[1]/1e3) - 240

    nd_lodaer.body_dp.xsens_data['index_span'] = [start, end]
    nd_lodaer.body_dp.xsens_data['t_diff'] = nd_lodaer.t_diff       # from xsense to head glasses
    # nd_lodaer.body_dp.xsens_data['Ts_Hd_Hx'] = nd_lodaer.Ts_Hd_Hx[0].to_matrix()       # from xsense to head glasses

    args.end_idx = min(args.end_idx, head_traj.shape[0])

    # --------------------------------------------
    # 2. Synchronize the LiDAR and mocap data.
    # --------------------------------------------
    _, _, frameids, synced_data_file = save_sync_data(
        root_folder, 
        head_traj,
        nd_lodaer.body_dp.xsens_data,
        start = args.start_idx, 
        end   = args.end_idx)
    
    finetune_first_person_data(synced_data_file, head_traj[frameids])