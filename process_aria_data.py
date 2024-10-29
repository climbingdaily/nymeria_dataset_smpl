import sys
import os
from glob import glob
import pickle
from os.path import dirname, split, abspath
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

import configargparse
from sympy import true

sys.path.append(dirname(split(abspath( __file__))[0]))

from nymeria.data_provider import NymeriaDataProvider
from utils import poses_to_joints, mocap_to_smpl_axis, sync_lidar_mocap
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

def finetune_first_person_data(synced_data_file):
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
        sensor_traj = fp_data['lidar_traj'].copy() 

        try:
            j = poses_to_joints(first_pose)
        except:
            j = poses_to_joints(first_pose, is_cuda=False, batch_size=1024)

        smpl_head_traj = np.concatenate((save_data['frame_num'].reshape(-1, 1), 
                                            j[:, 15] + mocap_tran), axis=1)
        smpl_head_traj_path = os.path.join(os.path.dirname(synced_data_file), 'smpl_head_trajectory.txt')
        np.savetxt(smpl_head_traj_path, smpl_head_traj)
        print(f'Saved initial smpl head trajectory to {smpl_head_traj_path}')
        
        root_to_head = j[:, 15] - j[:, 0]

        data_lenght = len(mocap_tran)

        ROT, _, _ = compute_similarity(
            mocap_tran[:data_lenght//2, :2] + root_to_head[:data_lenght//2, :2], 
            sensor_traj[:data_lenght//2, 1:3])
        
        delta_degree = np.linalg.norm(R.from_matrix(ROT).as_rotvec()) * 180 / np.pi
        print(f'[First person] {delta_degree:.1f}° around Z-axis from the mocap to LiDAR data.')

        scaled_trans = mocap_tran @ ROT.T
        scaled_trans -= scaled_trans[0] - mocap_tran[0] # move to the start position

        first_pose[:, :3] = (R.from_matrix(
            ROT) * R.from_rotvec(first_pose[:, :3])).as_rotvec()

        joints, verts = poses_to_joints(first_pose[:1], return_verts=True, is_cuda=False) 
        feet_center = (joints[0, 7] + joints[0, 8])/2
        feet_center[2] = verts[..., 2].min()
        scaled_trans -= scaled_trans[0] + feet_center  # 使得第一帧位于原点

        fp_data['origin_trans'] = fp_data['mocap_trans'].copy()
        fp_data['origin_pose'] = fp_data['pose'].copy()
        fp_data['lidar_traj'] = sensor_traj
        fp_data['mocap_trans'] = scaled_trans
        fp_data['pose'] = first_pose

    else:
        save_data['first_person'] = {'lidar_traj': sensor_traj}

    with open(synced_data_file, "wb") as f:
        pickle.dump(save_data, f)
        print(f"File saved in {synced_data_file}")


def check_sync_valid(keytime_a, keytime_b, min_time_gap = 50, offset=1.5):
    gap_time = abs(keytime_a[-1] - keytime_a[0]) - abs(keytime_b[-1] - keytime_b[0])
    if keytime_a[-1] - keytime_a[0] > min_time_gap and abs(gap_time) < offset:
        return True, gap_time
    else:
        return False, 0

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
    
    # if len(dataset_params['lidar_sync']) > 0 and len(dataset_params['mocap_sync']) > 0:
    if True:
        # sensor['keytime']  = dataset_params['lidar_sync']   # lidar keytime for synchronization
        # mocap['keytime']  = dataset_params['mocap_sync']   # mocap keytime for synchronization
        # sensor['keyid'] =  get_keyid(lidar)
        # mocap['keyid'] =  get_keyid(mocap)

        # double_sync, gap_time = check_sync_valid(sensor['keytime'], mocap['keytime'])
        # print(f'Reletive keytime offset: {-gap_time:.3f}\n')
    
        # 1. sync. data
        sensor['syncid'], mocap['syncid'] = sync_lidar_mocap(sensor['times'], mocap['times'], 0.5/mocap['framerate'])
        # 2 choose start time
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
        trans      -= trans[0] + feet_center  # 使得第一帧位于原点

        # 3. save synced data
        update_data(save_data, 'first', trans, sync_pose)

        # 4. save the head trajectory if it doesn't exist (for comparison)
        if not os.path.exists(os.path.join(root_folder, 'smpl_head_trajectory.txt')):
            j = poses_to_joints(sync_pose, return_verts=False, is_cuda=False) 
            smpl_head_traj = np.concatenate((mocap['syncid'].reshape(-1, 1), 
                                             j[:, 15] + trans, 
                                             mocap['times'][mocap['syncid']].reshape(-1, 1)), axis=1)
            np.savetxt(os.path.join(root_folder, 'smpl_head_trajectory.txt'), smpl_head_traj)
            print(f'Saved initial smpl head trajectory to {os.path.join(root_folder, "smpl_head_trajectory.txt")}')


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

def add_lidar_traj(synced_data_file, lidar_traj):
    """
    It takes a synced data file and a lidar trajectory, and saves the lidar trajectory in the synced
    data file
    
    Args:
      synced_data_file: the file that contains the synced data
      lidar_traj: a list of lidar poses, each of which is a list of [ID, x, y, z, qx, qy, qz, qw, timestamp]
    """

    with open(synced_data_file, "rb") as f:
        save_data = pickle.load(f)

    save_data['first_person'] = {'lidar_traj': lidar_traj}

    with open(synced_data_file, "wb") as f:
        pickle.dump(save_data, f)
        print(f"Lidar traj saved in {synced_data_file}")


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

    # aria_poses = nd_lodaer.get_synced_poses()
    save_path = os.path.join(os.path.dirname(traj_file), 'head_pc.txt')
    save_path = os.path.join(os.path.dirname(traj_file), 'head_trajectory.txt')
    np.savetxt(save_path, head_traj)
    print(f"Aria glassed Head trajectory saved in {save_path}")

    start = np.searchsorted(nd_lodaer.body_dp.xsens_data['timestamps_us'], nd_lodaer.timespan_ns[0]/1e3) + 240
    end = np.searchsorted(nd_lodaer.body_dp.xsens_data['timestamps_us'], nd_lodaer.timespan_ns[1]/1e3) - 240

    nd_lodaer.body_dp.xsens_data['index_span'] = [start, end]
    nd_lodaer.body_dp.xsens_data['t_diff'] = nd_lodaer.t_diff       # from xsense to head glasses
    nd_lodaer.body_dp.xsens_data['Ts_Hd_Hx'] = nd_lodaer.Ts_Hd_Hx[0].to_matrix()       # from xsense to head glasses

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
    
    finetune_first_person_data(synced_data_file)

    # ----------------------------------------------------------------
    # compute hand-eye coordinates for     
    
    # --------------------------------------------
    # 3. Use TSDF fusion to build the scene mesh
    # --------------------------------------------
    # mapping_start, mapping_end = frameids[0] + 100, frameids[-1] - 100
    # if args.tsdf:
    #     kitti_poses = traj_to_kitti_main(lidar_traj, args.start_idx, args.end_idx)
    #     vdbf = VDBFusionPipeline(lidar_dir, 
    #                             traj_to_kitti_main(lidar_traj, mapping_start, mapping_end), 
    #                             start         = mapping_start,
    #                             end           = mapping_end,
    #                             map_name      = os.path.basename(root_folder),
    #                             sdf_trunc     = args.sdf_trunc,
    #                             voxel_size    = args.voxel_size, 
    #                             space_carving = True)
    #     vdbf.run(skip=args.skip_frame)
    #     try:
    #         vdbf.segment_ground()
    #     except Exception as e:
    #         print(e.args[0])

# python process_raw_data.py --root_folder <root folder> [--tsdf] [--sync]
# --tsdf,         building the scene mesh
# --sync,         synchronize lidar and imu
# --voxel_size,   The voxel filter parameter for TSDF fusion
