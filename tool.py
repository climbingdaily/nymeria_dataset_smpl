import sys
import os
from glob import glob
import pickle
from os.path import dirname, split, abspath
from pathlib import Path

import numpy as np
import configargparse
from PIL import Image
import cv2
import open3d as o3d

import projectaria_tools.core.mps as mps
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.data_provider import VrsDataProvider
from projectaria_tools.core.mps.utils import get_nearest_pose, filter_points_from_confidence, bisection_timestamp_search
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import (
    ImageData,
    ImageDataRecord,
    TimeDomain,
    TimeQueryOptions,
)

def get_stereo_image(vrs_dp: VrsDataProvider, t_ns: int, time_domain: TimeDomain = TimeDomain.TIME_CODE
) -> tuple[ImageData, ImageData, ImageData]:
    assert time_domain in [
        TimeDomain.DEVICE_TIME,
        TimeDomain.TIME_CODE,
    ], "unsupported time domain"

    if time_domain == TimeDomain.TIME_CODE:
        t_ns_device = vrs_dp.convert_from_timecode_to_device_time_ns(
            timecode_time_ns=t_ns
        )
    else:
        t_ns_device = t_ns

    # left image
    left_image_data, left_image_meta = vrs_dp.get_image_data_by_time_ns(
        StreamId("1201-1"),
        time_ns=t_ns_device,
        time_domain=TimeDomain.DEVICE_TIME,
        time_query_options=TimeQueryOptions.CLOSEST,
    )
    t_diff = t_ns_device - left_image_meta.capture_timestamp_ns

    right_image_data, right_image_meta = vrs_dp.get_image_data_by_time_ns(
        StreamId("1201-2"),
        time_ns=t_ns_device,
        time_domain=TimeDomain.DEVICE_TIME,
        time_query_options=TimeQueryOptions.CLOSEST,
    )


    image_data, image_meta = vrs_dp.get_image_data_by_time_ns(
        StreamId("214-1"),
        time_ns=t_ns_device,
        time_domain=TimeDomain.DEVICE_TIME,
        time_query_options=TimeQueryOptions.CLOSEST,
    )

    t_diff = t_ns_device - right_image_meta.capture_timestamp_ns

    return left_image_data, right_image_data, image_data

def get_intrinsic_from_calib(calib):
    cam = np.eye(3)
    cam[0, 0] = calib.get_focal_lengths()[0]
    cam[1, 1] = calib.get_focal_lengths()[1]
    cam[0, 2] = calib.get_principal_point()[0]
    cam[1, 2] = calib.get_principal_point()[1]
    return cam

def compute_depth_from_undistorted(left_image, right_image, left_calib, right_calib, label: str ="",is_show: bool=True):
    """
    Computes a depth map from already-undistorted stereo images using the extrinsic parameters from the right to the left camera.

    Parameters:
    - left_image (np.array): Undistorted left grayscale image.
    - right_image (np.array): Undistorted right grayscale image.
    - left_calib (calibration.Calibration): Calibration object for the left camera.
    - right_calib (calibration.Calibration): Calibration object for the right camera.
    - is_show (bool): Whether to show the rectified images. Defaults to True.

    Returns:
    - depth_map (np.array): The computed depth map.
    """

    K_1 = get_intrinsic_from_calib(left_calib)
    K_2 = get_intrinsic_from_calib(right_calib)
    R = right_calib.get_transform_device_camera().to_matrix()[:3, :3]
    T = right_calib.get_transform_device_camera().to_matrix()[:3, 3]
    f = K_1[0, 0]  # focal length
    B = np.linalg.norm(T)
    units = 0.512     # depth units, adjusted for the output to fit in one byte

    R1 = np.eye(3)
    P1 = K_1
    P2 = np.array([
            [K_2[0,0], 0, K_2[0,2], 0],
            [0, K_2[1,1], K_2[1,2], -f * B],
            [0, 0, 1, 0]
        ])
    # P2 = K_2
    R2 = R
    # R1, R2, P1, P2, Q, roi1, roi2  = cv2.stereoRectify(K_1, None, K_2, None, np.rot90(left_image, -1).shape[:2], R, T)
    map1, map2 = cv2.initUndistortRectifyMap(K_1, None, R1, P1, np.rot90(left_image, -1).shape[:2], cv2.CV_32FC1)
    map3, map4 = cv2.initUndistortRectifyMap(K_2, None, R2, P2, np.rot90(right_image, -1).shape[:2], cv2.CV_32FC1)

    # after rectifying
    rectified_left = cv2.remap(left_image, map1, map2, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_image, map3, map4, cv2.INTER_LINEAR)

    if is_show:
        cv2.imshow("Rectified main", np.rot90(rectified_left, -1))
        cv2.imshow(f"Rectified {label}", np.rot90(rectified_right, -1))

    # minDisparity = 0
    numDisparities = 64  # 适应较远的物体
    blockSize = 5        # 合适的匹配块大小
    # P1 = 8 * blockSize**2
    # P2 = 32 * blockSize**2

    # stereo = cv2.StereoSGBM_create(
    #     minDisparity=minDisparity,
    #     numDisparities=numDisparities,
    #     blockSize=blockSize,
    #     P1=P1,
    #     P2=P2,
    # )

    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity_map = stereo.compute(cv2.cvtColor(rectified_left, cv2.COLOR_RGB2GRAY),
                                    cv2.cvtColor(rectified_right, cv2.COLOR_RGB2GRAY))
    
    disparity_map = np.maximum(disparity_map, 0)
    # disparity_map = cv2.medianBlur(disparity_map, 3)
    disparity_map = cv2.GaussianBlur(disparity_map, (3, 3), 0)

    valid_pixels = disparity_map > 0
    depth_map = np.zeros(shape=rectified_left.shape[:2]).astype("uint8")
    depth_map[valid_pixels] = (f * B * 1e3) / (units * disparity_map[valid_pixels] + 1e-5)
    # disparity_map = cv2.GaussianBlur(disparity_map, (5, 5), 0)
    depth_map = cv2.equalizeHist(depth_map)
    colorized_depth = np.zeros((rectified_left.shape[0], rectified_left.shape[1], 3), dtype="uint8")
    temp = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    colorized_depth[valid_pixels] = temp[valid_pixels]

    # depth_map[depth_map > 10] = 10
    # depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    # depth_normalized = np.uint8(depth_normalized)
    # depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)


    disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_normalized = np.uint8(disparity_map_normalized)

    if is_show:
        cv2.imshow(f"Disparity Map {label}", np.rot90(cv2.applyColorMap(disparity_map_normalized, cv2.COLORMAP_INFERNO), -1))
        cv2.imshow(f"Depth Map {label}", np.rot90(colorized_depth, -1))

    return depth_map

def get_updated_calib(input_calib, label=""):
    return calibration.get_linear_camera_calibration(
        input_calib.get_image_size()[0], 
        input_calib.get_image_size()[1], 
        input_calib.get_focal_lengths()[0],
        label,
        input_calib.get_transform_device_camera())

def undistort_image(input_img, new_calib, input_calib, label=""):
    undistorted_img = calibration.distort_by_calibration(input_img, new_calib, input_calib)
    cv2.imshow(label, np.rot90(input_img, -1))
    cv2.imshow(f"Undistored {label}", np.rot90(undistorted_img, -1))
    return undistorted_img

def load_point_cloud(points_path:str, save_pc_path:str, inv_dist_std: float=0.0006, dist_std: float=0.01, voxel_size=0.01):
    raw_pts = mps.read_global_point_cloud(points_path)
    filtered_points = filter_points_from_confidence(raw_pts, inv_dist_std, dist_std)
    pts = []
    for p in filtered_points:
        pts.append(p.position_world.astype(np.float32))
    pts = np.stack(pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)  # remove the duplicate points
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
    o3d.io.write_point_cloud(save_pc_path, pcd)

    print(f"{len(pcd.points)} points after filtering.")
    return pcd, raw_pts

def assign_frame_indices(observations, closed_loop_traj):
    """
    为observations中的每个点分配其在closed_loop_traj中的帧索引
    """
    frame_index = []
    for obs in observations:
        idx = bisection_timestamp_search(closed_loop_traj, obs.frame_capture_timestamp.total_seconds() * 1e9)
        frame_index.append(idx)
    return frame_index
    
def process_large_observations(observations, points, closed_loop_traj, online_calibs):
    """
    优化处理大规模观测点，将UV点按序号归类到每帧，并投影计算深度。
    """
    frame_indices  = assign_frame_indices(observations, closed_loop_traj)

    # 初始化帧结果存储
    frame_uv_points = {idx: [] for idx in range(len(closed_loop_traj))}

    # 按帧分组UV点
    for obs, frame_idx in zip(observations, frame_indices):
        frame_uv_points[frame_idx].append(obs)

    # 逐帧处理
        # 逐帧处理
    for frame_idx, frame_points in frame_uv_points.items():
        if not frame_points:
            continue  # 跳过没有观测点的帧
        
        for obs in frame_points:
            uid = obs.point_uid
            point_3d = find_point_3d(uid, points)
            if point_3d is not None:
                projected_uv, depth = project_3d_to_2d(point_3d, camera_intrinsics, camera_extrinsics)
                observed_uv = obs.uv.flatten()  # 获取观测的UV
                print(f"  UID {uid}: Observed UV: {observed_uv}, Projected UV: {projected_uv}, Depth: {depth}")
            else:
                print(f"  UID {uid}: Point not found in 3D points.")

def find_point_3d(uid, points):
    """
    根据uid从points中查找3D点（已排序情况下优化查找）。
    """
    for pt in points:
        if pt.uid == uid:
            return pt.position_world.flatten()  # 返回3D点的numpy数组
    return None

def project_3d_to_2d(point_3d, camera_intrinsics, camera_extrinsics):
    """
    将3D点投影到2D像素平面
    """
    point_3d_h = np.append(point_3d, 1.0)
    camera_coord = camera_extrinsics @ point_3d_h
    depth = camera_coord[2]
    pixel_coord_h = camera_intrinsics @ camera_coord[:3]
    u = pixel_coord_h[0] / pixel_coord_h[2]
    v = pixel_coord_h[1] / pixel_coord_h[2]
    return (u, v), depth

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()

    parser.add_argument("--root_folder", type=str, 
                        # default="/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd",
                        default="D:\\Data\\20231222_s1_kenneth_fischer_act7_56uvqd",
                        help="The data's root directory")

    args, opts    = parser.parse_known_args()
    root_folder   = args.root_folder
    head_f        = glob(root_folder + '/*head')[0]

    online_calib_path = os.path.join(head_f, 'mps', 'slam', 'online_calibration.jsonl')
    closed_loop_path  = os.path.join(head_f, 'mps', 'slam', 'closed_loop_trajectory.csv')
    glo_points_path   = os.path.join(head_f, 'mps', 'slam', 'semidense_points.csv.gz')
    observations_path = os.path.join(head_f, 'mps', 'slam', 'semidense_observations.csv.gz')
    
    vrsfile           = os.path.join(head_f, 'data', 'data.vrs')

    save_pc_path      = os.path.join(root_folder, 'body', 'pc_head.ply')


    # Goal: To have the UV pixel and corresponding points and frame number
    # todo: 1. read uid, camera, and uv of this observation
    # 2. get the point cloud from the uid in raw points cloud
    # 3. determine the frame number based on the graph_uid in closed loop traj fiel

    online_calibs = mps.read_online_calibration(online_calib_path)
    closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)
    pcd, raw_points = load_point_cloud(glo_points_path, save_pc_path)
    observations = mps.read_point_observations(observations_path)
    
    process_large_observations(observations, raw_points, closed_loop_traj, online_calibs)

    stream_mappings = {
        "camera-slam-left": StreamId("1201-1"),
        "camera-slam-right": StreamId("1201-2"),
        "camera-rgb": StreamId("214-1"),
        # "camera-eyetracking": StreamId("211-1"),
    }
    
    # filter the point cloud using thresholds on the inverse depth and distance standard deviation
    inverse_distance_std_threshold = 0.001
    distance_std_threshold = 0.15

    provider: VrsDataProvider = data_provider.create_vrs_data_provider(vrsfile)

    # left_config = provider.get_image_configuration(StreamId("1201-1"))
    # right_config = provider.get_image_configuration(StreamId("1201-2"))
    # rgb_config = provider.get_image_configuration(StreamId("214-1"))

    left_cam_calib  = provider.get_device_calibration().get_camera_calib("camera-slam-left")
    right_cam_calib = provider.get_device_calibration().get_camera_calib("camera-slam-right")
    rgb_cam_calib   = provider.get_device_calibration().get_camera_calib("camera-rgb")


    pinhole_rgb = calibration.get_linear_camera_calibration(
        480, 
        480, 
        240, "camera-rgb",rgb_cam_calib.get_transform_device_camera())

    for calib in online_calibs[240*80: -240*8]:
        query_timestamp_ns = int(calib.tracking_timestamp.total_seconds()*1e9)
        l, r, rgb = get_stereo_image(provider, query_timestamp_ns, TimeDomain.DEVICE_TIME)

        transform_world_device = get_nearest_pose(closed_loop_traj, query_timestamp_ns).transform_world_device # device to world coordinates

        calib_l     = calib.camera_calibs[0] # left camera calibration
        calib_r     = calib.camera_calibs[1]  # right camera calibration
        calib_rgb   = calib.camera_calibs[2] # rgb camera calibration
        
        left_new_calib  = get_updated_calib(calib_l, "left")
        right_new_calib = get_updated_calib(calib_r, "right")
        rgb_new_calib   = get_updated_calib(calib_rgb, "rgb")

        left_img    = cv2.cvtColor(l.to_numpy_array(), cv2.COLOR_GRAY2RGB)
        right_img   = cv2.cvtColor(r.to_numpy_array(), cv2.COLOR_GRAY2RGB)
        rgb_img     = cv2.cvtColor(rgb.to_numpy_array(), cv2.COLOR_BGR2RGB)

        # l_image = Image.fromarray(l.to_numpy_array())
        
        undistorted_left  = undistort_image(left_img, left_new_calib, calib_l, "left")
        undistorted_right = undistort_image(right_img, right_new_calib, calib_r, "right")
        undistorted_rgb   = undistort_image(rgb_img, pinhole_rgb, rgb_cam_calib, "rgb")

        # compute_depth_from_undistorted(undistorted_left, undistorted_right, left_new_calib, right_new_calib)
        compute_depth_from_undistorted(undistorted_left, undistorted_right, left_new_calib, right_new_calib, "stereo")

        undistorted_left = undistorted_left[:, 80:-80]
        left_new_calib = calibration.get_linear_camera_calibration(
                left_new_calib.get_image_size()[1], 
                left_new_calib.get_image_size()[1], 
                left_new_calib.get_focal_lengths()[0],
                "crop_left",
                SE3.from_matrix(pinhole_rgb.get_transform_device_camera().inverse().to_matrix() @ left_new_calib.get_transform_device_camera().to_matrix()))
        
        compute_depth_from_undistorted(undistorted_rgb, undistorted_left, pinhole_rgb, left_new_calib, "left-cam")

        undistorted_right = undistorted_right[:, 80:-80]
        right_new_calib = calibration.get_linear_camera_calibration(
                right_new_calib.get_image_size()[1], 
                right_new_calib.get_image_size()[1], 
                right_new_calib.get_focal_lengths()[0],
                "crop_left",
                SE3.from_matrix(pinhole_rgb.get_transform_device_camera().inverse().to_matrix() @ right_new_calib.get_transform_device_camera().to_matrix()))
        compute_depth_from_undistorted(undistorted_rgb, undistorted_right, pinhole_rgb, right_new_calib, "right-cam")
        
        cv2.waitKey(0)
