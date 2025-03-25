# -*- coding: utf-8 -*-
# @Author  : Zhang.Jingyi

"""
This file contains definitions of useful data structures and the paths 
for the datasets and data files necessary to run the code.
"""

import os
import sys
from os.path import join

import numpy as np

from ThirdParties.human_body_prior.body_model.body_model import BodyModel

SMPL_DIR = os.path.split(os.path.abspath( __file__))[0]

SMPL_NEUTRAL = os.path.join(SMPL_DIR, 'smpl', 'SMPL_NEUTRAL.pkl')
SMPL_FEMALE = os.path.join(SMPL_DIR, 'smpl', 'SMPL_FEMALE.pkl')
SMPL_MALE = os.path.join(SMPL_DIR, 'smpl', 'SMPL_MALE.pkl')

SMPLH_NEUTRAL = os.path.join(SMPL_DIR, 'body_models', 'smplh', 'neutral', 'model.npz')
SMPLH_FEMALE = os.path.join(SMPL_DIR, 'body_models', 'smplh', 'female', 'model.npz')
SMPLH_MALE = os.path.join(SMPL_DIR, 'body_models', 'smplh', 'male', 'model.npz')

SMPL_SAMPLE_PLY = os.path.join(SMPL_DIR, 'smpl_sample.ply')

def load_body_models(gender = 'neutral', support_dir=os.path.dirname(__file__), num_betas=10, num_dmpls=0):
    bm_fname   = os.path.join(support_dir, f'body_models/smplh/{gender}/model.npz')
    dmpl_fname = os.path.join(support_dir, f'body_models/dmpls/{gender}/model.npz')
    bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname)#.to(comp_device)
    return bm
class SmplParams:
    # Shared parameters across all instances
    _pose = np.array([])  
    _trans = np.array([])
    _hand_pose = np.array([])
    _betas = np.array([])
    _gender = 'Neutral'  # Default gender

    @classmethod
    def _ensure_list_or_array(cls, value, name):
        """Ensures the value is a list or NumPy array."""
        if isinstance(value, (list, np.ndarray)):
            return np.array(value) if isinstance(value, list) else value
        raise TypeError(f"{name} must be a list or NumPy array")

    @property
    def pose(self):
        return SmplParams._pose
    
    @property
    def hand_pose(self):
        return SmplParams._hand_pose

    @pose.setter
    def pose(self, value):
        SmplParams._pose = self._ensure_list_or_array(value, "pose")

    @hand_pose.setter
    def _hand_pose(self, value):
        SmplParams._hand_pose = self._ensure_list_or_array(value, "hand_pose")

    @property
    def trans(self):
        return SmplParams._trans

    @trans.setter
    def trans(self, value):
        SmplParams._trans = self._ensure_list_or_array(value, "trans")

    @property
    def betas(self):
        return SmplParams._betas

    @betas.setter
    def betas(self, value):
        SmplParams._betas = self._ensure_list_or_array(value, "betas")

    @property
    def gender(self):
        return SmplParams._gender

    @gender.setter
    def gender(self, value):
        if value not in {'neutral', 'male', 'female'}:
            raise ValueError("gender must be 'neutral', 'male', or 'female'")
        SmplParams._gender = value

    def __repr__(self):
        return (f"SmplParams(pose={self.pose.tolist()}, trans={self.trans.tolist()}, hand_pose={self.pose.tolist()},"
                f"betas={self.betas.tolist()}, gender='{self.gender}')")
"""
Each dataset uses different sets of joints.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are:
0 - Right Ankle
1 - Right Knee
2 - Right Hip
3 - Left Hip
4 - Left Knee
5 - Left Ankle
6 - Right Wrist
7 - Right Elbow
8 - Right Shoulder
9 - Left Shoulder
10 - Left Elbow
11 - Left Wrist
12 - Neck (LSP definition)
13 - Top of Head (LSP definition)
14 - Pelvis (MPII definition)
15 - Thorax (MPII definition)
16 - Spine (Human3.6M definition)
17 - Jaw (Human3.6M definition)
18 - Head (Human3.6M definition)
19 - Nose
20 - Left Eye
21 - Right Eye
22 - Left Ear
23 - Right Ear
"""

JOINTS_IDX = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18,
              20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]

COL_NAME = [
    "time",
    "Hips.y",
    "Hips.x",
    "Hips.z",
    "RightUpLeg.y",
    "RightUpLeg.x",
    "RightUpLeg.z",
    "RightLeg.y",
    "RightLeg.x",
    "RightLeg.z",
    "RightFoot.y",
    "RightFoot.x",
    "RightFoot.z",
    "LeftUpLeg.y",
    "LeftUpLeg.x",
    "LeftUpLeg.z",
    "LeftLeg.y",
    "LeftLeg.x",
    "LeftLeg.z",
    "LeftFoot.y",
    "LeftFoot.x",
    "LeftFoot.z",
    "Spine.y",
    "Spine.x",
    "Spine.z",
    "Spine1.y",
    "Spine1.x",
    "Spine1.z",
    "Spine2.y",
    "Spine2.x",
    "Spine2.z",
    "Neck.y",
    "Neck.x",
    "Neck.z",
    "Neck1.y",
    "Neck1.x",
    "Neck1.z",
    "Head.y",
    "Head.x",
    "Head.z",
    "RightShoulder.y",
    "RightShoulder.x",
    "RightShoulder.z",
    "RightArm.y",
    "RightArm.x",
    "RightArm.z",
    "RightForeArm.y",
    "RightForeArm.x",
    "RightForeArm.z",
    "RightHand.y",
    "RightHand.x",
    "RightHand.z",
    "RightHandThumb1.y",
    "RightHandThumb1.x",
    "RightHandThumb1.z",
    "RightHandThumb2.y",
    "RightHandThumb2.x",
    "RightHandThumb2.z",
    "RightHandThumb3.y",
    "RightHandThumb3.x",
    "RightHandThumb3.z",
    "RightInHandIndex.y",
    "RightInHandIndex.x",
    "RightInHandIndex.z",
    "RightHandIndex1.y",
    "RightHandIndex1.x",
    "RightHandIndex1.z",
    "RightHandIndex2.y",
    "RightHandIndex2.x",
    "RightHandIndex2.z",
    "RightHandIndex3.y",
    "RightHandIndex3.x",
    "RightHandIndex3.z",
    "RightInHandMiddle.y",
    "RightInHandMiddle.x",
    "RightInHandMiddle.z",
    "RightHandMiddle1.y",
    "RightHandMiddle1.x",
    "RightHandMiddle1.z",
    "RightHandMiddle2.y",
    "RightHandMiddle2.x",
    "RightHandMiddle2.z",
    "RightHandMiddle3.y",
    "RightHandMiddle3.x",
    "RightHandMiddle3.z",
    "RightInHandRing.y",
    "RightInHandRing.x",
    "RightInHandRing.z",
    "RightHandRing1.y",
    "RightHandRing1.x",
    "RightHandRing1.z",
    "RightHandRing2.y",
    "RightHandRing2.x",
    "RightHandRing2.z",
    "RightHandRing3.y",
    "RightHandRing3.x",
    "RightHandRing3.z",
    "RightInHandPinky.y",
    "RightInHandPinky.x",
    "RightInHandPinky.z",
    "RightHandPinky1.y",
    "RightHandPinky1.x",
    "RightHandPinky1.z",
    "RightHandPinky2.y",
    "RightHandPinky2.x",
    "RightHandPinky2.z",
    "RightHandPinky3.y",
    "RightHandPinky3.x",
    "RightHandPinky3.z",
    "LeftShoulder.y",
    "LeftShoulder.x",
    "LeftShoulder.z",
    "LeftArm.y",
    "LeftArm.x",
    "LeftArm.z",
    "LeftForeArm.y",
    "LeftForeArm.x",
    "LeftForeArm.z",
    "LeftHand.y",
    "LeftHand.x",
    "LeftHand.z",
    "LeftHandThumb1.y",
    "LeftHandThumb1.x",
    "LeftHandThumb1.z",
    "LeftHandThumb2.y",
    "LeftHandThumb2.x",
    "LeftHandThumb2.z",
    "LeftHandThumb3.y",
    "LeftHandThumb3.x",
    "LeftHandThumb3.z",
    "LeftInHandIndex.y",
    "LeftInHandIndex.x",
    "LeftInHandIndex.z",
    "LeftHandIndex1.y",
    "LeftHandIndex1.x",
    "LeftHandIndex1.z",
    "LeftHandIndex2.y",
    "LeftHandIndex2.x",
    "LeftHandIndex2.z",
    "LeftHandIndex3.y",
    "LeftHandIndex3.x",
    "LeftHandIndex3.z",
    "LeftInHandMiddle.y",
    "LeftInHandMiddle.x",
    "LeftInHandMiddle.z",
    "LeftHandMiddle1.y",
    "LeftHandMiddle1.x",
    "LeftHandMiddle1.z",
    "LeftHandMiddle2.y",
    "LeftHandMiddle2.x",
    "LeftHandMiddle2.z",
    "LeftHandMiddle3.y",
    "LeftHandMiddle3.x",
    "LeftHandMiddle3.z",
    "LeftInHandRing.y",
    "LeftInHandRing.x",
    "LeftInHandRing.z",
    "LeftHandRing1.y",
    "LeftHandRing1.x",
    "LeftHandRing1.z",
    "LeftHandRing2.y",
    "LeftHandRing2.x",
    "LeftHandRing2.z",
    "LeftHandRing3.y",
    "LeftHandRing3.x",
    "LeftHandRing3.z",
    "LeftInHandPinky.y",
    "LeftInHandPinky.x",
    "LeftInHandPinky.z",
    "LeftHandPinky1.y",
    "LeftHandPinky1.x",
    "LeftHandPinky1.z",
    "LeftHandPinky2.y",
    "LeftHandPinky2.x",
    "LeftHandPinky2.z",
    "LeftHandPinky3.y",
    "LeftHandPinky3.x",
    "LeftHandPinky3.z"
]

bones=[(0,1), (1,4), (4,7), (7,10), # R leg
       (0,2), (2,5), (5,8), (8,11), # L leg
       (0,3), (3,6), (6,9), # Spine
       (9,12), (12,15), # Head
       (9,13), (13,16), (16,18), (18,20), (20,22), # R arm
       (9,14), (14,17), (17,19), (19,21), (21,23)] # L arm

body_idx = {
    'neck': [12],
    'head': [15],
    'shold': [13, 14, 16, 17],
    'u_arm': [18, 19],
    'hands': [20, 21],
    'torso': [3, 6, 9],
    'hips' : [1, 2],
    'knees': [4, 5],
    'foot' : [7, 8],
    'ends' : [10, 11, 22, 23]
}

body_weight = np.asarray([1.] * 24).astype('float32')
body_weight[body_idx['neck']]  = 0.8
body_weight[body_idx['head']]  = 0.8
body_weight[body_idx['shold']] = 0.8
body_weight[body_idx['u_arm']] = 0.5
body_weight[body_idx['hands']] = 0.1
body_weight[body_idx['torso']] = 1.0
body_weight[body_idx['hips']]  = 0.8
body_weight[body_idx['knees']] = 0.4
body_weight[body_idx['foot']]  = 0.1
body_weight[body_idx['ends']]  = 0.1

body_prior_weight = np.asarray([1.] * 24).astype('float32')
body_prior_weight[body_idx['head']]  = 5
body_prior_weight[body_idx['neck']]  = 5
body_prior_weight[body_idx['hands']] = 1
body_prior_weight[body_idx['foot']]  = 3
body_prior_weight[body_idx['ends']]  = 5

import json
with open(os.path.join(SMPL_DIR, 'smpl_vert_segmentation.json')) as f:
    verts_seg = json.load(f)

torso     = []
legs      = []
feet      = []
arms      = []
hands     = []
low_limbs = []

for part in ['spine', 
             'spine1', 
             'spine2', 
             'hips']:
    torso += verts_seg[part]
for part in ['leftLeg', 
             'leftUpLeg', 
             'rightLeg', 
             'rightUpLeg']:
    legs += verts_seg[part]
for part in ['leftFoot', 
             'leftToeBase', 
             'rightFoot', 
             'rightToeBase']:
    feet += verts_seg[part]
for part in ['leftArm', 
             'leftForeArm', 
             'leftShoulder', 
             'rightArm', 
             'rightShoulder', 
             'rightForeArm']:
    arms += verts_seg[part]
for part in ['leftHandIndex1',
             'leftHand', 
             'rightHandIndex1', 
             'rightHand',]:
    hands += verts_seg[part]

for part in ['rightForeArm', 
             'leftForeArm', 
             'leftLeg', 
             'rightLeg']:
    low_limbs += verts_seg[part]

body_seg_verts = {
    'torso'     : list(set(torso)),
    'legs'      : list(set(legs)),
    'feet'      : list(set(feet)),
    'arms'      : list(set(arms)),
    'hands'     : list(set(hands)),
    'leftHand'  : list(set(verts_seg['leftHandIndex1'] + verts_seg['leftHand'])),
    'rightHand' : list(set(verts_seg['rightHandIndex1'] + verts_seg['rightHand'])),
    'low_limbs' : list(set(low_limbs)),
    'body'      : list(set(torso + verts_seg['leftUpLeg'] + verts_seg['rightUpLeg'])),
    'head'      : list(set(verts_seg['neck'] + verts_seg['head'])),
    'left_arm'  : list(set(verts_seg['leftForeArm'] + verts_seg['leftHand'] + verts_seg['leftHandIndex1'])),
    'right_arm' : list(set(verts_seg['rightForeArm'] + verts_seg['rightHand'] + verts_seg['rightHandIndex1'])),
    'left_leg'  : list(set(verts_seg['leftLeg'] + verts_seg['leftFoot'] + verts_seg['leftToeBase'])),
    'right_leg' : list(set(verts_seg['rightLeg'] + verts_seg['rightFoot'] + verts_seg['rightToeBase']))
}

