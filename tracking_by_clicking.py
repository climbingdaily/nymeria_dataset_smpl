import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import pickle

from sam2.build_sam import build_sam2_video_predictor

def sam2_tracking_function(prompts:dict, predictor, inference_state, mask_dict:dict, img_list):
    # lefthand_id = 1
    # righthand_id = 2
    print("SAM2 tracking with prompts:", prompts)
    # predictor.reset_state(inference_state)
    for frame_name, prompt in prompts.items():
        if frame_name not in mask_dict:
            mask_dict[frame_name] = {}
        idx = img_list.index(frame_name)
        for obj_id in prompt:

            # for labels, `1` means positive click and `0` means negative click
            points = prompt[obj_id]['point']
            labels = np.array(prompt[obj_id]['label'], np.int32)
            
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )
            
            for i, id in enumerate(out_obj_ids):
                mask_dict[frame_name][id] = (out_mask_logits[i] > 0.0).cpu().numpy()
            
            print(f"Processed tracking for frame {idx} ({frame_name}) with obj_id {obj_id}: {points}.")

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
    SAM_ROOT = os.path.join(current_dir, 'ThirdParties', 'sam2')
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

class ImageAnnotator:
    def __init__(self, img_folder, sam2_function, start, end):
        self.sam2_function = sam2_function  

        self.rotate_image = input("Enter 0 or 1 for rotating the image: ")
        try:
            self.rotate_image = bool(int(self.rotate_image))
        except ValueError:
            print("Invalid rotating input, defaulting to 0.")
            self.rotate_image = False

        self.image_list, self.img_folder = copy_images(img_folder, start, end)        
        self.index = 0  

        # load prompt if it exists
        prompt_file = os.path.join(os.path.dirname(self.img_folder), 
                                   f"prompts_{os.path.basename(self.img_folder)}.pkl")
        mask_file = os.path.join(os.path.dirname(self.img_folder), 
                                 f"mask_{os.path.basename(self.img_folder)}.pkl")
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
        self.predictor, self.inference_state =  init_predictor(self.img_folder)

        self.load_image()
        
    def prop_predict(self, ):
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            fname = self.image_list[out_frame_idx]
            self.mask_dict[fname] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    def load_image(self, view=True):
        """load the image based on the current index"""
        img_path = os.path.join(self.img_folder, self.image_list[self.index])
        self.img = cv2.imread(img_path)
        if self.img is None:
            print(f"Error loading image: {img_path}")
            return
        
        if self.rotate_image:
            self.img = cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE)

        return self.show_image(view)
    
    def show_image(self, view=True):
        """显示图片，并在顶部绘制文件名，如果有mask，则叠加mask"""
        img_display = self.img.copy()
        fname = self.image_list[self.index]
        
        if fname in self.mask_dict:
            colormap = cm.get_cmap("tab10")
            for obj_id, mask in self.mask_dict[fname].items():
                color = (np.array(colormap(obj_id % 10)[:3]) * 255).astype(np.uint8)
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                img_display = cv2.addWeighted(img_display, 1.0, mask_image, 0.3, 0)
        
        if fname in self.prompt_dict:
            colormap = cm.get_cmap("tab10")
            for obj_id, prompt in self.prompt_dict[fname].items():
                for point, label in zip(prompt["point"], prompt["label"]):
                    color = (np.array(colormap(label % 10)[:3]) * 255).astype(np.uint8)
                    cv2.circle(img_display, tuple(point), 5, color.tolist(), -1)
        
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
            self.sam2_function(self.prompt_dict, self.predictor, self.inference_state, self.mask_dict, self.image_list)
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
                self.sam2_function(self.prompt_dict, self.predictor, self.inference_state, self.mask_dict, self.image_list)  
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
        for idx, image_name in enumerate(self.image_list):
            self.index = idx  # 更新当前帧
            img = self.load_image(view=True) 
            self.video_writer.write(img) 
            print(f"Writing frame {idx + 1}/{len(self.image_list)}")

        # 录制完成，释放视频编写器
        self.stop_video_recording()

    def stop_video_recording(self):
        """停止视频录制"""
        if self.video_writer:
            self.video_writer.release()  # 释放视频编写器
            self.video_writer = None
        self.is_recording = False
        print(f"Video saved at {self.video_filename}")

    def save_mask(self, img_dir):
        """将 mask_dict 保存到 pickle 文件"""
        out_mask = os.path.join(os.path.dirname(img_dir), f"mask_{os.path.basename(img_dir)}.pkl")
        out_prompts = os.path.join(os.path.dirname(img_dir), f"prompts_{os.path.basename(img_dir)}.pkl")

        if self.rotate_image:
           for k, v in self.mask_dict.items():
                 v[1] = cv2.rotate(v[1][0].astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)[None, ...]
                 v[2] = cv2.rotate(v[2][0].astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)[None, ...]

        with open(out_mask, "wb") as f:
            pickle.dump(self.mask_dict, f)
        print(f"Mask dictionary saved to {out_mask}")

        with open(out_prompts, "wb") as f:
            pickle.dump(self.prompt_dict, f)
        print(f"Prompt dictionary saved to {out_prompts}")

# 用法示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--pkl_path", type=str, default='/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd/synced_data/humans_param.pkl', help="Path to the pkl file")
    parser.add_argument("--img_folder", type=str, default='/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd/recording_head/imgs', help="Path to the image folder")

    parser.add_argument("-S", "--start", type=int, default=1049,
                        help='Optimization start frame in the original trajectory')
    
    parser.add_argument("-E", "--end", type=int, default=1990,
                        help='Optimization end frame in the original trajectory')
    args = parser.parse_args()
    
    img_folder = args.img_folder
    start = args.start
    end = args.end
    
    annotator = ImageAnnotator(img_folder, sam2_tracking_function, start, end)
    annotator.run()
