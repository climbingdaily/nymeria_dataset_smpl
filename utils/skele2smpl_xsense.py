

import numpy as np
import argparse
import math
import os
import pandas as pd
import torch
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R

'''
pose --> rotation of 24 skelentons
beta --> shape of human

pose can be:
    1. (B, 24, 3, 3)
    or
    2. (B, 72)
beta should be:
    (B, 10)
'''

'''
SMPL
'Root', 'Left_Hip', 'Right_Hip', 'Waist', 'Left_Knee', 'Right_Knee',
'Upper_Waist', 'Left_Ankle', 'Right_Ankle', 'Chest',
'Left_Toe', 'Right_Toe', 'Base_Neck', 'Left_Shoulder',
'Right_Shoulder', 'Upper_Neck', 'Left_Arm', 'Right_Arm',
'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist',
'Left_Finger', 'Right_Finger'
'''


def get_x_rot_mat(theta):
    res = np.eye(3)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    res[1, 1] = cos_theta
    res[1, 2] = -sin_theta
    res[2, 1] = sin_theta
    res[2, 2] = cos_theta
    return res


def get_y_rot_mat(theta):
    res = np.eye(3)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    res[0, 0] = cos_theta
    res[0, 2] = sin_theta
    res[2, 0] = -sin_theta
    res[2, 2] = cos_theta
    return res


def get_z_rot_mat(theta):
    res = np.eye(3)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    res[0, 0] = cos_theta
    res[0, 1] = -sin_theta
    res[1, 0] = sin_theta
    res[1, 1] = cos_theta
    return res


def rotmat_to_axis_angle(rotmat, return_angle=False):
    angle = math.acos((rotmat[0, 0] + rotmat[1, 1] + rotmat[2, 2] - 1) / 2)
    vec = [rotmat[2, 1] - rotmat[1, 2], rotmat[0, 2] -
           rotmat[2, 0], rotmat[1, 0] - rotmat[0, 1]]
    norm = np.linalg.norm(vec)
    if abs(norm) < 1e-8:
        norm = 1.0
    for i in range(3):
        vec[i] /= norm
    if return_angle:
        return np.array([vec, angle])
    for i in range(3):
        vec[i] *= angle
    return np.array(vec)

axis2xsens = {
    'Head': 'Head',
    'Neck1': 'Neck',
    'Neck': 'Neck',
    'Spine2': 'Chest4',
    'RightShoulder': 'RightCollar',
    'RightArm': 'RightShoulder',
    'RightForeArm': 'RightElbow',
    'RightHand': 'RightWrist',
    'RightHandThumb2': 'RightWristEnd',
    'Spine1': 'Chest3',
    'Spine': 'Chest',
    'Hips': 'Hips',
    'RightUpLeg': 'RightHip',
    'RightLeg': 'RightKnee',
    'RightFoot': 'RightAnkle',
    'RightFootEnd': 'RightToe',
}

axis2xsens.update({k.replace('Right', 'Left'): v.replace('Right', 'Left') for k, v in axis2xsens.items()})

smpl_to_imu = ['Hips', 'LeftHip', 'RightHip', 'Chest', 'LeftKnee', 
               'RightKnee', 'Chest3', 'LeftAnkle', 'RightAnkle', 'Chest4',
               'LeftToe', 'RightToe', 'Neck', 'LeftCollar', 
               'RightCollar', 'Head', 'LeftShoulder', 'RightShoulder',
               'LeftElbow', 'RightElbow', 'LeftWrist', 'RightWrist', 
               'LeftWristEnd', 'RightWristEnd']

def get_pose_from_bvh(rotation_df, idx=0, converter_version = True ):
    smpl_to_imu = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', 'LeftLeg',
                   'RightLeg', 'Spine1', 'LeftFoot', 'RightFoot', 'Spine2',
                   'LeftFootEnd', 'RightFootEnd', 'Neck', 'LeftShoulder',
                   'RightShoulder', 'Head', 'LeftArm', 'RightArm',
                   'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand',
                   'LeftHandThumb2', 'RightHandThumb2']
    smpl_to_imu = [axis2xsens[e] for e in smpl_to_imu]

    pose = []
    import pandas as pd
    for each in smpl_to_imu:
        xrot = math.radians(rotation_df.at[idx, each + '.X'])
        yrot = math.radians(rotation_df.at[idx, each + '.Y'])
        zrot = math.radians(rotation_df.at[idx, each + '.Z'])

        rotmat = np.eye(3)
        rotmat = np.dot(rotmat, get_y_rot_mat(yrot))
        rotmat = np.dot(rotmat, get_x_rot_mat(xrot))
        rotmat = np.dot(rotmat, get_z_rot_mat(zrot))
        pose.append(rotmat)
    for i in range(len(pose)):
        pose[i] = rotmat_to_axis_angle(pose[i])

    pose = np.stack(pose).flatten()
    return pose  