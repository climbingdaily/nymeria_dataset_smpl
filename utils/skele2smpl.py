# -*- coding: utf-8 -*-

import numpy as np
import math
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

def get_smpl_pose_from_xsense_keypoints(input_quat_pose):
    """
    Convert a quaternion array of shape (n, 23, 4) to axis-angle representation (n, 24, 3).
    Rearranges from Xsens keypoints order to SMPL order.
    
    Parameters:
    input_quat_pose (numpy.ndarray): Quaternion array of shape (n, 23, 4), where quaternions 
                                     are in [qw, qx, qy, qz] format.
    
    Returns:
    numpy.ndarray: SMPL pose array of shape (n, 24, 3), where 24 joints are in axis-angle format.
    """
    xsense_to_smpl_order = [1, 19, 15, 2, 20, 16, 3, 21, 17, 4, 22, 18, 5, 11, 7, 6, 12, 8, 13, 9, 14, 10]

    # Convert from [qw, qx, qy, qz] to [qx, qy, qz, qw] format used by scipy
    quat_array_xyzw = np.roll(input_quat_pose[:, xsense_to_smpl_order], shift=-1, axis=-1)
    
    # Create a Rotation object from the quaternion array
    rotation = R.from_quat(quat_array_xyzw.reshape(-1, 4)).as_rotvec() # (n, 22, 3) axis-angle output
    
    smpl_pose = np.zeros((input_quat_pose.shape[0], 24, 3))

    smpl_pose[:, :22, :] = rotation.reshape(-1,22, 3)

    return smpl_pose.reshape(-1, 72)


def get_pose_from_bvh(rotation_df, idx=0, converter_version=False):
    smpl_to_imu = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', 'LeftLeg',
                   'RightLeg', 'Spine1', 'LeftFoot', 'RightFoot', 'Spine2',
                   'LeftFootEnd', 'RightFootEnd', 'Neck', 'LeftShoulder',
                   'RightShoulder', 'Head', 'LeftArm', 'RightArm',
                   'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand',
                   'LeftHandThumb2', 'RightHandThumb2']
    pose = []
    columns = [c.lower() for c in rotation_df.columns]
    smpl_to_imu = [c.lower() for c in smpl_to_imu]
    
    for each in smpl_to_imu:
        if converter_version:
            xrot = math.radians(rotation_df.at[idx, each + '.X'])
            yrot = math.radians(rotation_df.at[idx, each + '.Y'])
            zrot = math.radians(rotation_df.at[idx, each + '.Z'])
        else:
            xrot = 0
            yrot = 0
            zrot = 0
            if each + '.x' in columns:
                xrot = math.radians(rotation_df.at[idx, each + '.x'])
                yrot = math.radians(rotation_df.at[idx, each + '.y'])
                zrot = math.radians(rotation_df.at[idx, each + '.z'])
        if each == 'LeftShoulder'.lower() : #108
            zrot -= 0.3 
        elif each == 'RightShoulder'.lower(): # 39
            zrot += 0.3
        elif each == 'LeftArm'.lower(): # 111
            zrot += 0.3 
        elif each == 'RightArm'.lower(): # 42
            zrot -= 0.3
        rotmat = np.eye(3)
        rotmat = np.dot(rotmat, get_y_rot_mat(yrot))
        rotmat = np.dot(rotmat, get_x_rot_mat(xrot))
        rotmat = np.dot(rotmat, get_z_rot_mat(zrot))
        pose.append(rotmat)
    for i in range(len(pose)):
        pose[i] = rotmat_to_axis_angle(pose[i])
    pose = np.stack(pose).flatten()
    return pose  # return rotation matrix