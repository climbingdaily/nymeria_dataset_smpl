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

import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.data_provider import VrsDataProvider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import (
    ImageData,
    ImageDataRecord,
    TimeDomain,
    TimeQueryOptions,
)

from utils import depth_estimation

def get_stereo_image(vrs_dp, t_ns: int, time_domain: TimeDomain = TimeDomain.TIME_CODE
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

def compute_depth_from_undistorted(left_image, right_image, left_calib, right_calib, is_show: bool=True):
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

    K_left = get_intrinsic_from_calib(left_calib)
    K_right = get_intrinsic_from_calib(right_calib)
    R = right_calib.get_transform_device_camera().to_matrix()[:3, :3]
    T = right_calib.get_transform_device_camera().to_matrix()[:3, 3]
    
    stereo_rectify = cv2.stereoRectify(K_left, None, K_right, None, np.rot90(left_image, -1).shape[:2], R, T)
    map1, map2 = cv2.initUndistortRectifyMap(K_left, None, np.eye(3), K_left, np.rot90(left_image, -1).shape[:2], cv2.CV_32FC1)
    map3, map4 = cv2.initUndistortRectifyMap(K_right, None, R, K_right, np.rot90(right_image, -1).shape[:2], cv2.CV_32FC1)

    # after rectifying
    rectified_left = cv2.remap(left_image, map1, map2, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_image, map3, map4, cv2.INTER_LINEAR)

    if is_show:
        cv2.imshow("Rectified Left", np.rot90(rectified_left, -1))
        cv2.imshow("Rectified Right", np.rot90(rectified_right, -1))

    # minDisparity = 0
    # numDisparities = 64  # 适应较远的物体
    # blockSize = 7        # 合适的匹配块大小
    # P1 = 8 * blockSize**2
    # P2 = 32 * blockSize**2

    # # 创建StereoSGBM对象
    # stereo = cv2.StereoSGBM_create(
    #     minDisparity=minDisparity,
    #     numDisparities=numDisparities,
    #     blockSize=blockSize,
    #     P1=P1,
    #     P2=P2,
    # )

    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)

    disparity_map = stereo.compute(cv2.cvtColor(rectified_left, cv2.COLOR_RGB2GRAY),
                                    cv2.cvtColor(rectified_right, cv2.COLOR_RGB2GRAY))

    disparity_smoothed = cv2.GaussianBlur(disparity_map, (5, 5), 0)

    disparity_map_normalized = cv2.normalize(disparity_smoothed, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_normalized = np.uint8(disparity_map_normalized)

    if is_show:
        cv2.imshow("Disparity Map", np.rot90(cv2.applyColorMap(disparity_map_normalized, cv2.COLORMAP_PARULA), -1))
    
    # depth estimation
    f = K_left[0, 0]  # focal length
    B = np.linalg.norm(T)  # baseline
    depth_map = (f * B) / (disparity_map + 1e-5)

    depth_map[depth_map > 35] = 35

    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = np.uint8(depth_normalized)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PARULA)

    if is_show:
        cv2.imshow("Depth Map", np.rot90(depth_colormap, -1))

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
    cv2.imshow(label, np.rot90(undistorted_img, -1))
    cv2.imshow(f"Undistored {label}", np.rot90(undistorted_img, -1))
    return undistorted_img

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()

    parser.add_argument("--root_folder", type=str, 
                        default="/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd",
                        help="The data's root directory")

    args, opts    = parser.parse_known_args()
    root_folder   = args.root_folder

    head_f        = glob(root_folder + '/*head')[0]

    online_calib_path = os.path.join(head_f, 'mps', 'slam', 'online_calibration.jsonl')
    traj_file     = os.path.join(head_f, 'mps', 'slam', 'closed_loop_trajectory.csv')
    vrsfile       = os.path.join(head_f, 'data', 'data.vrs')

    stream_mappings = {
        "camera-slam-left": StreamId("1201-1"),
        "camera-slam-right": StreamId("1201-2"),
        "camera-rgb": StreamId("214-1"),
        # "camera-eyetracking": StreamId("211-1"),
    }

    online_calibs = mps.read_online_calibration(online_calib_path)
    provider: VrsDataProvider = data_provider.create_vrs_data_provider(vrsfile)

    # left_config = provider.get_image_configuration(StreamId("1201-1"))
    # right_config = provider.get_image_configuration(StreamId("1201-2"))
    # rgb_config = provider.get_image_configuration(StreamId("214-1"))

    left_cam_calib  = provider.get_device_calibration().get_camera_calib("camera-slam-left")
    right_cam_calib = provider.get_device_calibration().get_camera_calib("camera-slam-right")
    rgb_cam_calib   = provider.get_device_calibration().get_camera_calib("camera-rgb")


    pinhole_rgb = calibration.get_linear_camera_calibration(
        512, 
        512, 
        250, "camera-rgb")

    for calib in online_calibs:
        l, r, rgb = get_stereo_image(provider, int(calib.tracking_timestamp.total_seconds()*1e9), TimeDomain.DEVICE_TIME)
        
        calib_l     = calib.camera_calibs[0] # left camera calibration
        calib_r     = calib.camera_calibs[1]  # right camera calibration
        calib_rgb   = calib.camera_calibs[2] # rgb camera calibration
        
        left_new_calib  = get_updated_calib(calib_l, "left")
        right_new_calib = get_updated_calib(calib_r, "right")
        rgb_new_calib   = get_updated_calib(calib_rgb, "rgb")

        left_img    = cv2.cvtColor(l.to_numpy_array(), cv2.COLOR_GRAY2RGB)
        right_img   = cv2.cvtColor(r.to_numpy_array(), cv2.COLOR_GRAY2RGB)

        # l_image = Image.fromarray(l.to_numpy_array())
        undistorted_left = undistort_image(left_img, left_new_calib, calib_l, "left")
        undistorted_right = undistort_image(right_img, right_new_calib, calib_r, "right")

        compute_depth_from_undistorted(undistorted_left, undistorted_right, left_new_calib, right_new_calib)

        undistorted_image = calibration.distort_by_calibration(rgb.to_numpy_array(), 
                                                               pinhole_rgb,
                                                               rgb_cam_calib)
        # cv2.imshow("rgb", np.rot90(rgb.to_numpy_array(), -1))
        cv2.imshow("un rgb", np.rot90(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB), -1))
        cv2.waitKey(0)
