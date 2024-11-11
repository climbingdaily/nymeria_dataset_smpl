import sys
import os
from glob import glob
import pickle
from os.path import dirname, split, abspath
from pathlib import Path

from numpy import append
import configargparse
from PIL import Image

import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider, image
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import (
    ImageData,
    ImageDataRecord,
    TimeDomain,
    TimeQueryOptions,
)
def get_rgb_image(vrs_stream, t_ns: int, time_domain: TimeDomain = TimeDomain.TIME_CODE
) -> tuple[ImageData, ImageDataRecord, int]:
    assert vrs_stream.has_rgb, "recording has no rgb video"
    assert time_domain in [
        TimeDomain.DEVICE_TIME,
        TimeDomain.TIME_CODE,
    ], "unsupported time domain"

    if time_domain == TimeDomain.TIME_CODE:
        t_ns_device = vrs_stream.vrs_dp.convert_from_timecode_to_device_time_ns(
            timecode_time_ns=t_ns
        )
    else:
        t_ns_device = t_ns

    image_data, image_meta = vrs_stream.vrs_dp.get_image_data_by_time_ns(
        StreamId("214-1"),
        time_ns=t_ns_device,
        time_domain=TimeDomain.DEVICE_TIME,
        time_query_options=TimeQueryOptions.CLOSEST,
    )
    t_diff = t_ns_device - image_meta.capture_timestamp_ns

    return image_data, image_meta, t_diff

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()

    parser.add_argument("--root_folder", type=str, 
                        default="C:\\Users\\dyd12\\Documents\\Nymeria\\20231222_s1_kenneth_fischer_act7_56uvqd",
                        help="The data's root directory")

    parser.add_argument("--traj_file", type=str, default='lidar_trajectory.txt')

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
        "camera-eyetracking": StreamId("211-1"),
    }

    online_calibs = mps.read_online_calibration(online_calib_path)
    provider = data_provider.create_vrs_data_provider(vrsfile)
    stream_id = provider.get_stream_id_from_label("camera-rgb")

    t_ns_global = 2
    for [stream_name, stream_id] in stream_mappings.items():
        image = provider.get_image_data_by_index(stream_id, index)
        image = provider.get_rgb_image(
            "camera-slam-left",
            t_ns_global, 
            time_domain=TimeDomain.TIME_CODE)
        Image.fromarray(image[0].to_numpy_array()).save(f'{stream_name}.png')

    for calib in online_calibs:
        # for imuCalib in calib.imu_calibs:
        #     if imuCalib.get_label() == "imu-left":
        #         leftImuCalib = imuCalib
        
        # get left SLAM camera's online calibration
        for camCalib in calib.camera_calibs:
            if camCalib.get_label() == "camera-slam-left":
                leftCamCalib = camCalib
            if camCalib.get_label() == "camera-slam-left":
                rightCamCalib = camCalib
            if camCalib.get_label() == "camera-rgb":
                rightCamCalib = camCalib