import sys
import os
from glob import glob
import pickle
from os.path import dirname, split, abspath
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import torch
from tqdm import tqdm
from loguru import logger

import configargparse
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core import calibration
from projectaria_tools.core.sensor_data import (
    ImageData,
    ImageDataRecord,
    TimeDomain,
    TimeQueryOptions,
)
sys.path.append(dirname(split(abspath( __file__))[0]))

from nymeria.data_provider import NymeriaDataProvider
# from nymeria.handeye import HandEyeSolver
from utils import poses_to_joints, mocap_to_smpl_axis, sync_lidar_mocap, poses_to_vertices_torch
from utils import save_json_file, read_json_file, print_table, compute_similarity, Renderer
from utils.cam_tool import *
from smpl import SMPL_Layer, BODY_PARTS

field_fmts = ['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.3f']
LIGHT_RED=(0.31372549, 0.35686275, 1.0)
LIGHT_BLUE=(1.0, 0.6, 0.2)

def save_trajs(save_dir, rot, pos, mocap_id, comments='first', framerate=100):
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

def finetune_first_person_data(synced_data_file, sensor_traj, T_sensor_head, return_verts=False):
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

        fp_data['sensor_traj'] = sensor_traj

        _, j, glo_rot = poses_to_vertices_torch(first_pose, mocap_tran, betas=np.array(fp_data['beta']), gender=fp_data['gender'], is_cuda=False)

        # save smpl head trajectory
        smpl_head_traj = np.concatenate((save_data['frame_num'].reshape(-1, 1), j[:, 15].cpu().numpy()), axis=1)
        smpl_head_traj_path = os.path.join(os.path.dirname(synced_data_file), 'smpl_head_trajectory.txt')
        np.savetxt(smpl_head_traj_path, smpl_head_traj)
        print(f'Saved initial smpl head trajectory to {smpl_head_traj_path}')
        
        head_to_root = (j[:, 0] - j[:, 15]).cpu().numpy() # head joint to root joint
        root_to_trans = mocap_tran - j[:, 0].cpu().numpy() # root joint to smpl translation 

        ROT, _, _ = compute_similarity(j[:len(mocap_tran)//2, 15, :2].numpy(), 
                                       sensor_traj[:len(mocap_tran)//2, 1:3])

        delta_degree = np.linalg.norm(R.from_matrix(ROT).as_rotvec()) * 180 / np.pi
        print(f'[First person] {delta_degree:.1f}° around Z-axis from the mocap to Sensor data.')

        T_w_sensor = SE3.from_quat_and_translation(
            sensor_traj[:, 7], 
            sensor_traj[:, 4:7], 
            sensor_traj[:, 1:4]
        )

        New_translation = (T_w_sensor.to_matrix() @ T_sensor_head)[:, :3, 3] + head_to_root @ ROT.T + root_to_trans

        first_pose[:, :3] = (R.from_matrix(ROT) * R.from_rotvec(first_pose[:, :3])).as_rotvec()

        fp_data['trans'] = New_translation
        fp_data['pose'] = first_pose
        fp_data['mocap_pose'] = first_pose.copy()
        fp_data['mocap_trans'] = (mocap_tran - mocap_tran[0]) @ ROT.T + New_translation[0]
        fp_data['T_sensor_head'] = T_sensor_head.tolist()

    else:
        save_data['first_person'] = {'sensor_traj': sensor_traj}

    with open(synced_data_file, "wb") as f:
        pickle.dump(save_data, f)
        print(f"File saved in {synced_data_file}")

    return fp_data['pose'], fp_data['trans'], fp_data['mocap_trans']

def draw_bbox_on_image(img, bbox):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
    color = (0, 255, 0)  # 绿色
    thickness = 2  # 线条粗细

    img_with_bbox = img.copy()
    cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow("BBox Image", img_with_bbox)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def save_sync_data(root_folder, time_stamps, body_data, start=0, end=np.inf, gap_frame=-1):
    first_data  = True
    second_data = True

    json_path   = os.path.join(root_folder, 'dataset_params.json')

    save_dir = os.path.join(root_folder, 'synced_data')
    param_file = os.path.join(save_dir, 'humans_param.pkl')
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(json_path):
        dataset_params = read_json_file(json_path)
    else:
        dataset_params = {}
    sensor = {}
    mocap = {}

    sensor['times'] = time_stamps
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
        
    print('=======================')
    print('No second person data!!')
    print('=======================')
    second_data = False

    if not first_data and not second_data:
        print("No mocap data in './mocap_data/'!!!")
        exit(0)

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
                ["Parameter", "Sensor", "Mocap(IMU)"],
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

    save_data['framerate'] = sensor['framerate']
    save_data['device_ts'] = sensor['times'][sensor['syncid']]

    with open(param_file, "wb") as f:
        pickle.dump(save_data, f)
        print(f"File saved in {param_file}")
    
    return save_data['device_ts'], sensor['syncid'], param_file   # type: ignore


def project_pts_img(img, points, linear_calib, Tx, size=3):
    if isinstance(Tx, np.ndarray):
        T_cam_world = Tx    # T_world_device
    else:
        T_cam_world   = (Tx @ linear_calib.get_transform_device_camera()).inverse().to_matrix()
    u, v, depth = project_point_cloud_to_image(points, linear_calib, T_cam_world)
    depth_by_projection = remove_further_points_and_color(u,v, depth, 
                                                        linear_calib.get_image_size()[0], 
                                                        linear_calib.get_image_size()[1])   # (w, h)
    if depth_by_projection.max() > 0.1:
        return overlay_depth_on_image(img, 
                                    depth_by_projection, 
                                    min_d=0.1,
                                    size=size)[0] # rgb
    else:
        return img.copy()
    
def get_bbox_by_proj(points, linear_calib, T_cam_world):
    u, v, _ = project_point_cloud_to_image(points, linear_calib, T_cam_world)
    if len(u) == 0 or len(v) == 0:
        return None

    x1, y1 = min(u), min(v)
    x2, y2 = max(u), max(v)

    return round(x1), round(y1), round(x2), round(y2)

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()

    parser.add_argument("--root_folder", type=str, 
                        default="/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd",
                        help="The data's root directory")

    parser.add_argument("--params_file", type=str, default='dataset_params.json')

    parser.add_argument('-S', "--start_idx", type=int, default=0,
                        help='The start frame index in LiDAR for processing, specified when sychronization time is too late')

    parser.add_argument('-E', "--end_idx", type=int, default=np.inf,
                        help='The end frame index in LiDAR for processing, specified when sychronization time is too early.')

    parser.add_argument("--skip_frame", type=int, default=8, 
                        help='The everay n frame used for mapping')
    
    parser.add_argument('--vis_cam', action='store_true', help='Visualize human in the camera')

    args, opts  = parser.parse_known_args()
    root_folder = args.root_folder
    head_f      = glob(root_folder + '/*head')[0]
    params_file = os.path.join(root_folder, args.params_file)
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
    
    nd_lodaer = NymeriaDataProvider(sequence_rootdir=Path(root_folder), trajectory_sample_fps=100, return_diff=True)

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
    for tag, pc in aria_points.items():
        save_path = os.path.join(root_folder, 'synced_data', f'pc_{tag}.txt')
        np.savetxt(save_path, pc)

    # get camera_traj based on the online_calibs times
    T_c_w = []
    head_traj = []
    head_ang_vel = []
    head_lin_vel = []
    lwrist_traj = []
    rwrist_traj = []

    calib_head = nd_lodaer.recording_head.vrs_dp.get_device_calibration().get_camera_calib("camera-rgb")

    pre_time = 0
    count = 0
    for idx, t_s in enumerate(aria_trajs['recording_head'][:, -1]):
        calib = nd_lodaer.recording_head.mps_dp.get_online_calibration(int(t_s*1e9))
        tracking_timestamp = calib.tracking_timestamp.total_seconds()
        if tracking_timestamp == pre_time:
            continue
        pre_time = tracking_timestamp

        timecode_ns = nd_lodaer.recording_head.vrs_dp.convert_from_device_time_to_timecode_ns(int(tracking_timestamp*1e9))

        # head
        pose, _ = nd_lodaer.recording_head.get_pose(int(tracking_timestamp*1e9), TimeDomain.DEVICE_TIME)
        q_xyzw = R.from_matrix(pose.transform_world_device.to_matrix()[:3, :3]).as_quat()
        xyz = pose.transform_world_device.to_matrix()[:3, 3]
        head_traj.append(np.hstack([count, xyz, q_xyzw, tracking_timestamp])) # online calib. resuts
        T_c_w.append((pose.transform_world_device @ calib_head.get_transform_device_camera()).inverse().to_matrix())
        head_ang_vel.append(pose.angular_velocity_device)
        head_lin_vel.append(pose.device_linear_velocity_device)

        # left wrist
        pose, _ = nd_lodaer.recording_lwrist.get_pose(timecode_ns, TimeDomain.TIME_CODE)
        q_xyzw = R.from_matrix(pose.transform_world_device.to_matrix()[:3, :3]).as_quat()
        xyz = pose.transform_world_device.to_matrix()[:3, 3]
        lwrist_traj.append(np.hstack([count, xyz, q_xyzw, pose.tracking_timestamp.total_seconds()])) # online calib. resuts

        # right wrist
        pose, _ = nd_lodaer.recording_rwrist.get_pose(timecode_ns, TimeDomain.TIME_CODE)
        q_xyzw = R.from_matrix(pose.transform_world_device.to_matrix()[:3, :3]).as_quat()
        xyz = pose.transform_world_device.to_matrix()[:3, 3]
        rwrist_traj.append(np.hstack([count, xyz, q_xyzw, pose.tracking_timestamp.total_seconds()])) # online calib. resuts

        count += 1

    head_traj = np.stack(head_traj, axis=0)
    lwrist_traj = np.stack(lwrist_traj, axis=0)
    rwrist_traj = np.stack(rwrist_traj, axis=0)
    T_c_w = np.stack(T_c_w, axis=0)
    head_ang_vel = np.stack(head_ang_vel, axis=0)
    head_lin_vel = np.stack(head_lin_vel, axis=0)
    head_pc = aria_points['recording_head']

    start = np.searchsorted(nd_lodaer.body_dp.xsens_data['timestamps_us'], nd_lodaer.timespan_ns[0]/1e3) + 240
    end = np.searchsorted(nd_lodaer.body_dp.xsens_data['timestamps_us'], nd_lodaer.timespan_ns[1]/1e3) - 240

    nd_lodaer.body_dp.xsens_data['index_span'] = [start, end]
    nd_lodaer.body_dp.xsens_data['t_diff'] = nd_lodaer.t_diff       # from xsense to head glasses
    # nd_lodaer.body_dp.xsens_data['Ts_Hd_Hx'] = nd_lodaer.Ts_Hd_Hx[0].to_matrix()       # from xsense to head glasses

    args.end_idx = min(args.end_idx, head_traj.shape[0])

    # --------------------------------------------
    # 2. Synchronize the Aria and mocap data.
    # --------------------------------------------
    device_time, frameids, synced_data_file = save_sync_data(
        root_folder, 
        head_traj[:, -1],
        nd_lodaer.body_dp.xsens_data,
        start = args.start_idx, 
        end   = args.end_idx)
    
    if (np.diff(lwrist_traj[frameids, -1]) < 0.01).sum() > 0:
        print("Warning: The left wrist trajectory is not continuous.")
        exit(0)
    if (np.diff(rwrist_traj[frameids, -1]) < 0.01).sum() > 0:
        print("Warning: The right wrist trajectory is not continuous.")
        exit(0)

    np.savetxt(os.path.join(root_folder, 'synced_data', 'sensor_head_trajectory.txt'), head_traj[frameids])
    np.savetxt(os.path.join(root_folder, 'synced_data', 'sensor_lwrist_trajectory.txt'), lwrist_traj[frameids])
    np.savetxt(os.path.join(root_folder, 'synced_data', 'sensor_rwrist_trajectory.txt'), rwrist_traj[frameids])
    print(f"Aria glasses trajectory saved in {os.path.join(root_folder, 'synced_data')}")

    # define T_head_aria
    T_head_sensor = nd_lodaer.recording_head.vrs_dp.get_device_calibration().get_transform_cpf_sensor("camera-slam-left").to_matrix()
    T_head_sensor[:3, 3] = np.array([0.055, 0.096, 0.076])    # neutral 
    T_sensor_head = np.linalg.inv(T_head_sensor)  # transformation from head to aria
    print(f"Transformation from head to aria: \n{T_sensor_head}")

    first_pose, first_tran, trans2 = finetune_first_person_data(synced_data_file, head_traj[frameids], T_sensor_head)

    T_lwrist_sensor = nd_lodaer.recording_lwrist.vrs_dp.get_device_calibration().get_transform_cpf_sensor("camera-slam-left")
    P_lwrist_rgb = nd_lodaer.recording_lwrist.vrs_dp.get_device_calibration().get_transform_cpf_sensor("camera-rgb").translation()
    T_w_sensor = SE3.from_quat_and_translation(
        lwrist_traj[frameids, 7], 
        lwrist_traj[frameids, 4:7], 
        lwrist_traj[frameids, 1:4],
    )
    T_w_lwrist = T_w_sensor @ T_lwrist_sensor.inverse()
    T_lwrist_smpl = np.array([[0, 0, -1, 0],
                              [0, 1, 0, -0.025],
                              [1, 0, 0, P_lwrist_rgb[0,2]],
                              [0, 0, 0, 1]])
    T_rwrist_sensor = nd_lodaer.recording_rwrist.vrs_dp.get_device_calibration().get_transform_cpf_sensor("camera-slam-left")
    P_rwrist_rgb = nd_lodaer.recording_lwrist.vrs_dp.get_device_calibration().get_transform_cpf_sensor("camera-rgb").translation()

    T_w_sensor = SE3.from_quat_and_translation(
        rwrist_traj[frameids, 7], 
        rwrist_traj[frameids, 4:7], 
        rwrist_traj[frameids, 1:4]
    )    
    T_w_rwrist = T_w_sensor @ T_rwrist_sensor.inverse()
    T_rwrist_smpl = np.array([[0, 0, 1, 0],
                              [0, 1, 0, -0.025],
                              [-1, 0, 0, P_rwrist_rgb[0,2]],
                              [0, 0, 0, 1]])


    # --------------------------------------------
    # 4. Synchronize the camera data
    # --------------------------------------------
    if True:
        with open(synced_data_file, "rb") as f:
            save_data = pickle.load(f)
        
            fp_data = save_data['first_person']
            first_pose, first_tran, trans2, gender = fp_data['pose'], fp_data['trans'], fp_data['mocap_trans'], fp_data['gender']

        new_calib_head = calibration.get_linear_camera_calibration(
            1024, 
            1024, 
            400, "camera-rgb",calib_head.get_transform_device_camera())
        
        vertices, j, _ = poses_to_vertices_torch(first_pose, first_tran, gender='neutral', is_cuda=True)                            
        # bboxes = [[]] * len(vertices)
        bboxes = {"full": [[]] * len(vertices),
                  "leftHand": [[]] * len(vertices),
                  "rightHand": [[]] * len(vertices),
                  "left_arm": [[]] * len(vertices),
                  "right_arm": [[]] * len(vertices)}
        

        with open(synced_data_file, "wb") as f:
            save_data['first_person']['bboxes'] = bboxes
            T_w_lwristSmpl = T_w_lwrist.to_matrix() @ T_lwrist_smpl
            T_w_rwristSmpl = T_w_rwrist.to_matrix() @ T_rwrist_smpl
            u_left, v_left, _, l_valid = project_point_cloud_to_image((T_c_w[frameids] @ T_w_lwristSmpl)[:, :3, 3], new_calib_head, np.eye(4), True)
            u_right, v_right, _, r_valid = project_point_cloud_to_image((T_c_w[frameids] @ T_w_rwristSmpl)[:, :3, 3], new_calib_head, np.eye(4), True,)
            
            w, h  = new_calib_head.get_image_size()
            intrinsics = new_calib_head.projection_params().tolist()
            ex = T_c_w[frameids].tolist()

            save_data['first_person']['hand_gt'] = {
                'T_world_lsmpl': T_w_lwristSmpl.copy().tolist(),
                'T_world_rsmpl': T_w_rwristSmpl.copy().tolist(),
                'left_valid': l_valid.tolist(),
                'right_valid': r_valid.tolist(),
                'T_lwrist_smpl': T_lwrist_smpl.tolist(),    # from smpl wrist joints to wrist sensor
                'T_rwrist_smpl': T_rwrist_smpl.tolist(),    # from smpl wrist joints to wrist sensor
            }
            save_data['first_person']['cam_head'] = {'w': w, 
                                                     'h': h,
                                                     'intrinsic': intrinsics,
                                                     'extrinsic': ex,
                                                     'lin_vel': head_lin_vel[frameids].tolist(),
                                                     'ang_vel': head_ang_vel[frameids].tolist()}

            pickle.dump(save_data, f)
            print(f"Boxes saved in {synced_data_file}") 

        os.makedirs(os.path.join(head_f, 'imgs'), exist_ok=True)
        
        for idx, t_s in tqdm(enumerate(device_time), total=len(device_time), desc="Processing"):
            out_path = os.path.join(head_f, 'imgs',  f"{t_s:.6f}.jpg")
            if not os.path.exists(out_path):
                result = nd_lodaer.recording_head.get_rgb_image(int(t_s*1e9), TimeDomain.DEVICE_TIME)
                if abs(result[-1] / 1e6) > 33:
                    logger.warning(f"time difference for image query: {result[-1]/ 1e6} ms")
                rgb_img = cv2.cvtColor(result[0].to_numpy_array(), cv2.COLOR_BGR2RGB)
                undistorted_img = undistort_image(rgb_img, new_calib_head, calib_head, "rgb")
                # save the undistorted image
                cv2.imwrite(out_path, undistorted_img)
                
            # bbox = get_bbox_by_proj(vertices[idx].cpu().numpy(), 
            #                         new_calib_head, 
            #                         T_c_w[frameids[idx]])
            # if len(u_left) > 0 and len(v_left) > 0 or len(u_right) > 0 and len(v_right) > 0:
            #     bboxes['full'][idx] = bbox
            #     for parts in ['leftHand', 'rightHand', 'left_arm', 'right_arm']:
            #         bbox = get_bbox_by_proj(vertices[idx][BODY_PARTS[parts]].cpu().numpy(), 
            #                                 new_calib_head, 
            #                                 T_c_w[frameids[idx]])
            #         if bbox is not None:
            #             bboxes[parts][idx] = bbox

                                    
    if args.vis_cam:
        new_calib_head = calibration.get_linear_camera_calibration(
            512, 
            512, 
            200, "camera-rgb",calib_head.get_transform_device_camera())
        
        calib_obs   = nd_lodaer.recording_observer.vrs_dp.get_device_calibration().get_camera_calib("camera-rgb")
        new_calib_obs = calibration.get_linear_camera_calibration(
            512, 
            512, 
            200, "camera-rgb",calib_obs.get_transform_device_camera())

        intrinsics = new_calib_head.projection_params().tolist()

        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        # fourcc = cv2.VideoWriter_fourcc(*'X264')
        fps = 30  
        w, h  = new_calib_obs.get_image_size()
        video_writer = cv2.VideoWriter(os.path.join(head_f, "proj_smpl_verts.avi"), fourcc, fps, (w*2, h*2))
        
        vertices, j, _ = poses_to_vertices_torch(first_pose, first_tran, gender=gender, is_cuda=True)
        vertices2, j2, _ = poses_to_vertices_torch(first_pose, trans2, gender=gender, is_cuda=True)

        render = Renderer(resolution=(w, h), wireframe=True)
        
        faces = SMPL_Layer(gender=gender).th_faces
        ex = torch.tensor(ex).float()   # (B, 4, 4)
        K = torch.tensor([[
            [intrinsics[0], 0, intrinsics[2]],
            [0, intrinsics[1], intrinsics[3]],
            [0, 0, 1]
            ]] * len(frameids))
        
        for idx, t_s in tqdm(enumerate(device_time), total=len(device_time), desc="Processing"):
            result = nd_lodaer.recording_head.get_rgb_image(int(t_s*1e9), TimeDomain.DEVICE_TIME)
            if abs(result[-1] / 1e6) > 33:  # 33ms
                logger.warning(f"time difference for image query: {result[-1]/ 1e6} ms")
            rgb_img = cv2.cvtColor(result[0].to_numpy_array(), cv2.COLOR_BGR2RGB)
            undistorted_img  = undistort_image(rgb_img, new_calib_head, calib_head, "rgb")

            ex = T_c_w[frameids[idx]]
            img = render.render(
                undistorted_img,
                smpl_model    = (vertices[idx].cpu(), faces),
                cam           = (K[idx], ex),
                color         = LIGHT_RED,
                a=1)
            img2 = render.render(
                undistorted_img,
                smpl_model    = (vertices2[idx].cpu(), faces),
                cam           = (K[idx], ex),
                color         = LIGHT_BLUE,
                a=1)
            
            tc = nd_lodaer.recording_head.vrs_dp.convert_from_device_time_to_timecode_ns(int(t_s*1e9))
            pose, _ = nd_lodaer.recording_observer.get_pose(tc, TimeDomain.TIME_CODE)
            result = nd_lodaer.recording_observer.get_rgb_image(tc, TimeDomain.TIME_CODE)
            if abs(result[-1] / 1e6) > 33:  # 33ms
                logger.warning(f"time difference for image query: {result[-1]/ 1e6} ms")
            rgb_img = cv2.cvtColor(result[0].to_numpy_array(), cv2.COLOR_BGR2RGB)
            undistorted_img  = undistort_image(rgb_img, new_calib_obs, calib_obs, "rgb")

            ex = (pose.transform_world_device @ new_calib_obs.get_transform_device_camera()).inverse().to_matrix()
            img3 = render.render(
                undistorted_img,
                smpl_model    = (vertices[idx].cpu(), faces),
                cam           = (K[idx], ex),
                color         = LIGHT_RED,
                a=1)
            
            # img4 = project_pts_img(undistorted_img, vertices2[idx].cpu().numpy(), new_calib_obs, ex, size=1)
            img4 = render.render(
                undistorted_img,
                smpl_model    = (vertices2[idx].cpu(), faces),
                cam           = (K[idx], ex),
                color         = LIGHT_BLUE,
                a=1)

            img_a = cv2.hconcat([np.rot90(img2, -1), np.rot90(img, -1)])
            img_b = cv2.hconcat([np.rot90(img4, -1), np.rot90(img3, -1)])
            
            final_image = cv2.vconcat([img_a, img_b])

            # add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            color = (255, 255, 255)  
            shadow_color = (0, 0, 0)  

            h, w, _ = final_image.shape
            positions = [
                (int(w * 0.25) - 80, 30),  # ego: before 左上
                (int(w * 0.75) - 80, 30),  # ego: after  右上
                (int(w * 0.25) - 80, int(0.5 * h + 30)),  # third: before 左下
                (int(w * 0.75) - 80, int(0.5 * h + 30))   # third: after  右下
            ]
            labels = ["ego: origin", "ego: fit_sensor", "third: origin", "third: fit_sensor"]

            for (x, y), label in zip(positions, labels):
                # cv2.putText(final_image, label, (x+1, y+1), font, font_scale, shadow_color, thickness+2, cv2.LINE_AA)
                cv2.putText(final_image, label, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            video_writer.write(final_image[:, :, :3])
        video_writer.release()

