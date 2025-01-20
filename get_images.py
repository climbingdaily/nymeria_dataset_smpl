import sys
import os
from glob import glob
import pickle
from os.path import dirname, split, abspath
from pathlib import Path
import time 
import gc

import numpy as np
import configargparse
from PIL import Image
import cv2
import open3d as o3d
import matplotlib
from tqdm import tqdm
import gzip
import csv
import torch
import matplotlib.pyplot as plt
import h5py

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
from utils.tsdf_fusion_pipeline import VDBFusionPipeline

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
    t_diff = t_ns_device - right_image_meta.capture_timestamp_ns

    image_data, image_meta = vrs_dp.get_image_data_by_time_ns(
        StreamId("214-1"),
        time_ns=t_ns_device,
        time_domain=TimeDomain.DEVICE_TIME,
        time_query_options=TimeQueryOptions.CLOSEST,
    )

    t_diff = t_ns_device - right_image_meta.capture_timestamp_ns

    return left_image_data, left_image_meta, right_image_data, right_image_meta, image_data, image_meta

def undistort_image(input_img, new_calib, input_calib, label="", is_show=False):
    undistorted_img = calibration.distort_by_calibration(input_img, new_calib, input_calib)
    if is_show:
        cv2.imshow(label, np.rot90(input_img, -1))
        cv2.imshow(f"Undistored {label}", np.rot90(undistorted_img, -1))
    return undistorted_img


def main(args):
    root_folder   = args.root_folder

    head_f        = glob(root_folder + '/*head')[0]
    image_folder      = os.path.join(head_f, "images")
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    online_calib_path = os.path.join(head_f, 'mps', 'slam', 'online_calibration.jsonl')
    closed_loop_path  = os.path.join(head_f, 'mps', 'slam', 'closed_loop_trajectory.csv')
    vrsfile           = os.path.join(head_f, 'data', 'data.vrs')

    # load data
    online_calibs    = mps.read_online_calibration(online_calib_path)
    closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)

    provider: VrsDataProvider = data_provider.create_vrs_data_provider(vrsfile)
    rgb_cam_calib   = provider.get_device_calibration().get_camera_calib("camera-rgb")

    new_calib = calibration.get_linear_camera_calibration(
        1024, 
        1024, 
        512, "camera-rgb",rgb_cam_calib.get_transform_device_camera())
    

    skip_start = 240*5
    skip_end = 240*20
    counts = skip_start - 1

    extrinsics_sequence = []
    timestamps_sequence = []
    depth_sequence = []
    rgb_sequence = []

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    fps = 30  
    w, h  = new_calib.get_image_size()
    video_writer = cv2.VideoWriter(os.path.join(head_f, "video.avi"), fourcc, fps, (w, h))

    for calib in tqdm(online_calibs[skip_start: -skip_end], desc="Processing"):
        counts += 1
        device_time = calib.tracking_timestamp.total_seconds()
        calib_l, calib_r, calib_rgb = calib.camera_calibs # calibration results
        globa_pose  = get_nearest_pose(closed_loop_traj, int(device_time*1e9))

        l, _, r, _, rgb, _ = get_stereo_image(provider, int(device_time*1e9), TimeDomain.DEVICE_TIME)

        rgb_img   = cv2.cvtColor(rgb.to_numpy_array(), cv2.COLOR_BGR2RGB)
        undistorted_img   = undistort_image(rgb_img, new_calib, rgb_cam_calib, "rgb")
        video_writer.write(np.rot90(undistorted_img, -1))
    video_writer.release()

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()

    parser.add_argument("--root_folder", type=str, 
                        default="/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd",
                        # default="D:\\Data\\20231222_s1_kenneth_fischer_act7_56uvqd",
                        help="The data's root directory")

    parser.add_argument("-VS", "--voxel_size", type=float, default=0.06, 
                        help="The voxel filter parameter for TSDF fusion")
    
    parser.add_argument("-T", "--tsdf", type=bool, default=False)

    parser.add_argument("--skip_frame", type=int, default=6, 
                        help='The everay n frame used for mapping')
    
    parser.add_argument('--sdf_trunc', type=float, default=0.10,
                        help="The trunction distance for SDF funtion")
    
    args, opts    = parser.parse_known_args()

    main(args)