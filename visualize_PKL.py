import cv2
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import pickle
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (    
    RasterizationSettings,    
    MeshRasterizer,    
    MeshRenderer,
    SoftSilhouetteShader,
    PerspectiveCameras,)

sys.path.append('.')
sys.path.append('..')


from smpl import SMPL_Layer, load_body_models, SmplParams, rotation_matrix_to_axis_angle
from utils import smplh_to_vertices_torch, Renderer
from optmization import fill_and_smooth_mano_poses, get_hand_pose

LIGHT_RED=(1.0,  0.35686275,  0.31372549)
LIGHT_BLUE=(0.2, 0.6, 1.0)

def get_black_mask_r(img, threshold=10, kernel_size=(5, 5)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2  # 图像的中心点
 
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 可选：腐蚀和膨胀，去掉小的黑色噪声
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    xx, yy = mask[:h//3, :w//3].nonzero()
    distances=  []
    for (x, y) in zip(xx, yy):
        distance = np.sqrt((x - h/2)**2 + (y - w//2)**2)  # 计算欧几里得距离
        distances.append(distance)

    return min(distances) / (h/2)
def create_distance_mask(h=1024, w=1024, r=1.174):
    cx, cy = w // 2, h // 2  # 图像的中心点
    r = r * w / 2
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)  # 计算到中心的距离
            if distance >= r:
                mask[y, x] = 255  # 距离大于 r 的区域设为 255

    return mask

def print_key_functions():
    print("Key Functions:")
    print("q - Exit the program")
    print("a - Please add a click (Enter wait for mouse click mode)")
    print("t - Tracking started (performs tracking and reloads the image)")
    print("Right Arrow (83) - Move to next image")
    print("Left Arrow (81) - Move to previous image")
    print("p or 112 - Perform inference, update mask, and save mask")
    print("s or 114 - Save the current mask")
    print("v or 118 - Start video recording and save current frame")

def init_predictor(img_dir):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    SAM_ROOT = os.path.join(current_dir, '3rdParties', 'sam2')
    sam2_checkpoint = f"{SAM_ROOT}/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = f"configs/sam2.1/sam2.1_hiera_l.yaml"

    assert os.path.exists(sam2_checkpoint), f"checkpoint not found: {sam2_checkpoint}"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    inference_state = predictor.init_state(video_path=img_dir)
    return predictor, inference_state

def copy_images(img_folder, s, e):
    image_list = [f for f in os.listdir(img_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    image_list.sort(key=lambda p: float(os.path.splitext(p)[0]))
    image_list = image_list[s:e]

    new_folder = f'{img_folder}_{s}_{e}'
    os.makedirs(new_folder, exist_ok=True)

    # copy the image files superlink to a new folder
    for image_name in image_list:
        src_path = os.path.join(img_folder, image_name)
        dst_path = os.path.join(new_folder, image_name)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        os.symlink(src_path, dst_path)
    
    if not image_list:
        raise ValueError("No images found in the folder.")
    
    return image_list, new_folder
def convert_smplh_right_to_left(pose_right_hand):
    """
    使用旋转矩阵方法，将右手的 SMPL-H 轴角 Pose 转换为左手 Pose
    :param pose_right_hand: (15, 3) 右手的 Pose（轴角格式）
    :return: (15, 3) 左手的 Pose
    """
    pose_left_hand = np.zeros_like(pose_right_hand)

    for i in range(15):
        # 将轴角转换为旋转矩阵
        rot_mat = R.from_rotvec(pose_right_hand[i]).as_matrix()

        # 定义 YZ 平面的镜像矩阵
        M = np.array([
            [1,  0,  0],  # X 轴不变
            [0, -1,  0],  # Y 轴取反
            [0,  0, -1]   # Z 轴取反
        ])

        # 变换旋转矩阵
        rot_mat_mirrored = M @ rot_mat @ M

        # 变回轴角
        pose_left_hand[i] = R.from_matrix(rot_mat_mirrored).as_rotvec()

    return pose_left_hand
class ImageAnnotator:
    def __init__(self, img_folder, pkl_file, start, end):

        self.rotate_image = True

        self.image_list, self.img_folder = copy_images(img_folder, start, end)        
        self.index = 0  
        self.max_index = 250

        # load prompt if it exists
        prompt_file = os.path.join(os.path.dirname(self.img_folder), 
                                   f"prompts_{os.path.basename(self.img_folder)}.pkl")
        mask_file = os.path.join(os.path.dirname(self.img_folder), 
                                 f"mask_{os.path.basename(self.img_folder)}.pkl")
        mano_file = os.path.join(os.path.dirname(self.img_folder), 
                                 f"mano_{os.path.basename(self.img_folder)}.pkl")
        
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as f:
                synced_data = pickle.load(f)
        else:
            synced_data = {}



        self.proj_img_list = []
        self.proj_img_list2 = []
        
        # =================================================================
        person = 'first_person'
        human_data             = synced_data[person]
        camera_params          = synced_data[person]['cam_head']
        sensor_traj            = synced_data[person]['lidar_traj'].copy()
        self.data_length       = len(synced_data['frame_num'])
        # self.frame_time        = synced_data['device_ts']    
        SMPL = SmplParams()

        if 'manual_pose' in human_data:
            SMPL.pose = human_data['manual_pose'].copy()    # (n, 24, 3)
        else:
            SMPL.pose = human_data['pose'].copy()           # (n, 24, 3)
        self.synced_imu_trans = human_data['mocap_trans'].copy()        # (n, 3)

        if 'T_sensor_head' in human_data:
            self.head2sensor = torch.tensor(human_data['T_sensor_head']).float()

        # Transformation from the sensor (camera of lidar)
        SMPL.trans = human_data['trans'].copy()  # (n, 3)   

        if 'hand_pose' in human_data:
            SMPL.hand_pose = human_data['hand_pose'].copy().astype(np.float32)    # (n, 30, 3)
        else:
            SMPL.hand_pose = np.zeros((len(human_data['trans']), 30, 3))
            with open(mano_file, 'rb') as f:
                mano_data = pickle.load(f)
            SMPL.hand_pose[start: end] = get_hand_pose(mano_data, len(self.image_list))
        if 'opt_hand_pose' in human_data:
            opt_hand_pose = torch.tensor(human_data['opt_hand_pose'][start: end]).float()
        else:
            opt_hand_pose = torch.tensor(fill_and_smooth_mano_poses(SMPL.hand_pose[start: end])).float()

        if 'beta' in human_data:
            betas = torch.from_numpy(np.array(human_data['beta'])).float()
        else:
            betas = torch.zeros(10).float()

        if 'gender' in human_data:
            SMPL.gender = human_data['gender']
        # =================================================================
        if end > self.data_length:
            end = self.data_length
        # =================================================================
        opt_trans = human_data['opt_trans'][start: end].copy()
        opt_pose = human_data['opt_pose'][start: end].copy()
        #sensor_ori_params = torch.from_numpy(SMPL.trans[start: end, 4:8])
        trans = torch.from_numpy(SMPL.trans[start: end])
        hand_pose = torch.from_numpy(SMPL.hand_pose[start: end])

        sensor_t  = np.array([np.eye(4)] * self.data_length)
        sensor_t[:, :3, :3] = R.from_quat(sensor_traj[:, 4:8]).as_matrix()
        sensor_t[:, :3, 3:] = sensor_traj[:, 1:4].reshape(-1, 3, 1)
        sensor_t = torch.from_numpy(sensor_t[start: end])
            
        mocap_trans_params = torch.from_numpy(self.synced_imu_trans[start: end])

        ori_params  = torch.from_numpy(SMPL.pose[start: end])[:, :3]
        pose_params = torch.from_numpy(SMPL.pose[start: end])[:, 3:]
        
        smpl_layer = SMPL_Layer(gender=SMPL.gender)
        body_model = load_body_models(gender=SMPL.gender)

        if True:
            smpl_layer.cuda()
            body_model.cuda()
            betas              = betas.unsqueeze(0).type(torch.FloatTensor).cuda()
            trans              = trans.type(torch.FloatTensor).cuda()
            sensor_t           = sensor_t.type(torch.FloatTensor).cuda()
            mocap_trans_params = mocap_trans_params.type(torch.FloatTensor).cuda()
            ori_params         = ori_params.type(torch.FloatTensor).cuda()
            pose_params        = pose_params.type(torch.FloatTensor).cuda()

        #sensor_ori_params
        params = {'trans': trans,   # translation from the camera
                    'mocap_trans': mocap_trans_params,  # translation from the imu
                    'ori': ori_params,
                    'pose': pose_params,
                    'hand_pose': hand_pose.cuda(),
                    'sensor_traj':sensor_t,
                    'cameras': {'w': camera_params['w'],
                                'h': camera_params['h'],
                                'intrinsic': camera_params['intrinsic'],
                                'extrinsic': camera_params['extrinsic'][start:end],
                                } 
        }
        # cameras setting
        ex = torch.tensor(params['cameras']['extrinsic']).float()   # (B, 4, 4)
        image_size  = torch.tensor([[params['cameras']['h'], params['cameras']['w']]] * len(ex))
        f = torch.tensor([params['cameras']['intrinsic'][:2]] * len(ex)).float()
        c = torch.tensor([params['cameras']['intrinsic'][2:]] * len(ex)).float()
        K = torch.tensor([[
            [params['cameras']['intrinsic'][0], 0, params['cameras']['intrinsic'][2]],
            [0, params['cameras']['intrinsic'][1], params['cameras']['intrinsic'][3]],
            [0, 0, 1]
            ]] * len(ex))
        cameras = cameras_from_opencv_projection(R=ex[:, :3, :3].cuda(), tvec=ex[:, :3, 3].cuda(), 
                                                 image_size=image_size.cuda(), 
                                                 camera_matrix=K.cuda())
        smpl_verts, _ = smplh_to_vertices_torch(SMPL.pose[start: end], trans, hand_pose, betas=betas, gender=SMPL.gender)
        smpl_verts_2, _ = smplh_to_vertices_torch(opt_pose, opt_trans, opt_hand_pose, betas=betas, gender=SMPL.gender)
        self.empty_mask = create_distance_mask(camera_params['w'], camera_params['h'])
        
        render = Renderer(resolution=(camera_params['w'], camera_params['h']), wireframe=True)

        for idx, verts in tqdm(enumerate(smpl_verts[:self.max_index]), total=self.max_index, desc="Renderring verts"):

            img = render.render(
                np.zeros((camera_params['w'], camera_params['h'], 4)),
                smpl_model    = (verts.cpu(), smpl_layer.th_faces.cpu()),
                cam           = (camera_params['intrinsic'], ex[idx].cpu()),
                color         = LIGHT_BLUE,
                human_pc      = None,
                sence_pc_pose = None,
                mesh_filename = None,
                a=1)

            img2 = render.render(
                np.zeros((camera_params['w'], camera_params['h'], 4)),
                smpl_model    = (smpl_verts_2[idx].cpu(), smpl_layer.th_faces.cpu()),
                cam           = (camera_params['intrinsic'], ex[idx].cpu()),
                color         = LIGHT_RED,
                human_pc      = None,
                sence_pc_pose = None,
                mesh_filename = None,
                a=1)
            
            self.proj_img_list.append(img)
            self.proj_img_list2.append(img2)
        # =================================================================

        if os.path.exists(prompt_file):
            with open(prompt_file, 'rb') as f:
                self.prompt_dict = pickle.load(f)
        else:
            self.prompt_dict = {}

        if os.path.exists(mask_file):
            with open(mask_file, 'rb') as f:
                self.mask_dict = pickle.load(f)

            if self.rotate_image:
                for k, v in self.mask_dict.items():
                    v[1] = cv2.rotate(v[1][0].astype(np.uint8), cv2.ROTATE_90_CLOCKWISE)[None, ...]
                    v[2] = cv2.rotate(v[2][0].astype(np.uint8), cv2.ROTATE_90_CLOCKWISE)[None, ...]
        else:
            self.mask_dict = {}  #

        self.video_segments = {}  # video_segments contains the per-frame segmentation results
        self.adding_object = False

        # 创建窗口
        cv2.namedWindow("Image Annotator")
        cv2.setMouseCallback("Image Annotator", self.mouse_callback)

        self.load_image()
    def load_image(self, view=True, show=True):
        """load the image based on the current index"""
        self.index = self.index % self.max_index
        img_path = os.path.join(self.img_folder, self.image_list[self.index])
        self.img = cv2.imread(img_path)
        if self.img is None:
            print(f"Error loading image: {img_path}")
            return  
        
        # if self.rotate_image:
        #     self.img = cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE)
        if show:
            return self.show_image(view)
    
    def show_image(self, view=True):
        """显示图片，并在顶部绘制文件名，如果有mask，则叠加mask"""
        img_display = self.img.copy()
        fname = self.image_list[self.index]
        proj_img = self.proj_img_list[self.index]
        proj_img2 = self.proj_img_list2[self.index]

        if fname in self.mask_dict:
            colormap = cm.get_cmap("tab10")
            for obj_id, mask in self.mask_dict[fname].items():
                color = (np.array(colormap(obj_id % 10)[:3]) * 255).astype(np.uint8)
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                img_display = cv2.addWeighted(img_display, 1.0, mask_image, 0.4, 0)
        
        if fname in self.prompt_dict:
            colormap = cm.get_cmap("tab10")
            for obj_id, prompt in self.prompt_dict[fname].items():
                for point, label in zip(prompt["point"], prompt["label"]):
                    color = (np.array(colormap(label % 10)[:3]) * 255).astype(np.uint8)
                    cv2.circle(img_display, tuple(point), 5
                               , color.tolist(), -1)
        
        img_display = cv2.addWeighted(img_display, 1.0, cv2.rotate(proj_img[:, :, :3], cv2.ROTATE_90_CLOCKWISE), 1.0, 0)
        img_display = cv2.addWeighted(img_display, 1.0, cv2.rotate(proj_img2[:, :, :3], cv2.ROTATE_90_CLOCKWISE), 1.0, 0)
        img_display[self.empty_mask > 0] = 0
        cv2.putText(img_display, fname, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if view:
            cv2.imshow("Image Annotator", img_display)
        
        return img_display
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击事件处理"""
        if event == cv2.EVENT_LBUTTONDOWN and self.adding_object:
            frame_name = self.image_list[self.index]
            if frame_name not in self.prompt_dict:
                self.prompt_dict[frame_name] = {}
            
            obj_id = input("Enter obj_id for the clicked point: ")
            try:
                obj_id = int(obj_id)
            except ValueError:
                print("Invalid obj_id input, defaulting to 1.")
                obj_id = 1
            label = input("Enter label for the clicked point (1: positive, 0: negtive): ")
            try:
                label = int(label)
            except ValueError:
                print("Invalid obj_id input, defaulting to 1.")
                label = 1
            
            if obj_id not in self.prompt_dict[frame_name]:
                self.prompt_dict[frame_name][obj_id] = {"point": [[x, y]], "label": [label]}
            else:
                self.prompt_dict[frame_name][obj_id]["point"].append([x, y])
                self.prompt_dict[frame_name][obj_id]["label"].append(label)
            print(f"Updated prompt for frame {self.index} ({frame_name}): {{'points': [{x}, {y}], 'obj_id': {obj_id}, 'label': {label}}}")
            self.adding_object = False
            self.load_image()  # 重新加载图片并绘制 mask

    
    def run(self):
        """主循环，处理键盘事件"""
        print_key_functions()
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break  # 退出
            elif key == ord('a'):
                print("Please add a click")
                self.adding_object = True  # 进入等待鼠标点击模式
            elif key == ord('t'):
                print("Tracking started...")
                self.load_image()  # 重新加载图片并绘制 mask
            elif key == 83:  # 右箭头键
                self.index = min(self.index + 1, len(self.image_list) - 1)
                self.load_image()
            elif key == 81:  # 左箭头键
                self.index = max(self.index - 1, 0)
                self.load_image()
            elif key == ord('p') or key == 112: # key p
                self.prop_predict()  # 进行推理并更新 mask_dict
                self.save_mask(self.img_folder)  # 保存 mask_dict
                self.load_image()  # 重新加载图片并绘制 mask
            elif key == ord('s'): # key s
                self.save_mask(self.img_folder)  # 保存 mask_dict
            elif key == ord('v') or key == 118:  # key v
                self.start_video_recording()  # 保存当前帧
                print(f"Frame {self.index} saved!")
            elif key == ord('r'):  # key r 114
                self.rotate_image = not self.rotate_image 
                self.load_image()  # 重新加载图片并绘制 mask
            elif key == ord('h'):  # Option to display the help with key functions
                print_key_functions()  # Display key function list
        cv2.destroyAllWindows()

    def start_video_recording(self):
        """自动按顺序读取 image_list 并录制视频"""
        print("Starting video recording...")
        
        # 创建视频编写器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用XVID编码
        fps = 30  # 设置帧率
        frame_size = (self.img.shape[1], self.img.shape[0])  # 图像的大小
        self.video_filename = os.path.join(os.path.dirname(self.img_folder), 
                     f"seg_{os.path.basename(self.img_folder)}.mp4")
        self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, fps, frame_size)  # 创建视频写入对象

        # 遍历所有图像，按顺序写入视频
        for idx in np.arange(self.max_index):
            self.index = idx  # 更新当前帧
            img = self.load_image(view=True) 
            self.video_writer.write(img) 
            print(f"Writing frame {idx + 1}/{self.max_index}")

        # 录制完成，释放视频编写器
        self.stop_video_recording()

    def stop_video_recording(self):
        """停止视频录制"""
        if self.video_writer:
            self.video_writer.release()  # 释放视频编写器
            self.video_writer = None
        self.is_recording = False
        print(f"Video saved at {self.video_filename}")

# 用法示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, default='/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd/log/2025-03-27T01:04:12_.pkl', help="Path to the pkl file")
    parser.add_argument("--img_folder", type=str, default='/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd/recording_head/imgs', help="Path to the image folder")

    parser.add_argument("-S", "--start", type=int, default=1049,
                        help='Optimization start frame in the original trajectory')
    
    parser.add_argument("-E", "--end", type=int, default=1990,
                        help='Optimization end frame in the original trajectory')
    args = parser.parse_args()
    
    img_folder = args.img_folder
    start = args.start
    end = args.end
    
    annotator = ImageAnnotator(img_folder, args.pkl_path, start, end)
    annotator.run()
