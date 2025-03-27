import argparse
import pickle

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm

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
    def __init__(self, img_folder, start, end):

        # self.rotate_image = input("Enter 0 or 1 for rotating the image: ")
        # try:
        #     self.rotate_image = bool(int(self.rotate_image))
        # except ValueError:
        #     print("Invalid rotating input, defaulting to 0.")
        self.rotate_image = True
        self.current_hand = 0

        self.image_list, self.img_folder = copy_images(img_folder, start, end)        
        self.index = 0  

        img = cv2.imread(os.path.join(self.img_folder, self.image_list[0]))
        self.height, self.width = img.shape[:2]

        self.video_filename = os.path.join(os.path.dirname(self.img_folder), 
                     f"det_hand_{os.path.basename(self.img_folder)}.mp4")
        self.video_writer = None
        
        # load prompt if it exists
        det_file = os.path.join(os.path.dirname(self.img_folder), 
                                   f"det_hand_{os.path.basename(self.img_folder)}.pkl")
        if os.path.exists(det_file):
            with open(det_file, 'rb') as f:
                self.det_dict = pickle.load(f)
            for k, v in self.det_dict.items():
                assert len(v) == 2, "more than two hands"
                if self.rotate_image:
                    if len(v[0]) == 2:
                        v[0] = [self.height -v[0][1], v[0][0]]
                    if len(v[1]) == 2:
                        v[1] = [self.height -v[1][1], v[1][0]]

        else:
            self.det_dict = {iname: [[],[]] for iname in self.image_list}

        self.adding_object = False

        # 创建窗口
        cv2.namedWindow("Image Annotator")
        cv2.setMouseCallback("Image Annotator", self.mouse_callback)

        self.load_image()
        
    def load_image(self, view=True):
        """load the image based on the current index"""
        img_path = os.path.join(self.img_folder, self.image_list[self.index])
        self.img = cv2.imread(img_path)
        self.height, self.width = self.img.shape[:2]
        if self.img is None:
            print(f"Error loading image: {img_path}")
            return
        
        # if self.rotate_image:
            # self.img = cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE)

        return self.show_image(view)
    
    def show_image(self, view=True):
        """显示图片，并在顶部绘制文件名，如果有mask，则叠加mask"""
        img_display = self.img.copy()
        fname = self.image_list[self.index]
        
        if fname in self.det_dict:
            colormap = cm.get_cmap("tab10")
            left, right = self.det_dict[fname]
            c1 = (np.array(colormap(0 % 10)[:3]) * 255).astype(np.uint8)
            c2 = (np.array(colormap(1 % 10)[:3]) * 255).astype(np.uint8)
            if len(left) == 2:
                cv2.circle(img_display, tuple(left), 5, c1.tolist(), -1)
            if len(right) == 2:
                cv2.circle(img_display, tuple(right), 5, c2.tolist(), -1)
        
        cv2.putText(img_display, f"{self.index}: {fname}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if view:
            cv2.imshow("Image Annotator", img_display)
        
        return img_display
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击事件处理"""
        if event == cv2.EVENT_LBUTTONDOWN:
            frame_name = self.image_list[self.index]
            if frame_name not in self.det_dict:
                self.det_dict[frame_name] = [[], []]
            
            self.det_dict[frame_name][self.current_hand] = [x, y]
            print(f"{self.index} Added click {self.current_hand} at ({x}, {y})")
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
            elif key == ord('c'):
                self.current_hand = 1 - self.current_hand
                print(f"Current hand: {self.current_hand}")
            elif key == 83:  # 右箭头键
                self.index = min(self.index + 1, len(self.image_list) - 1)
                self.load_image()
            elif key == 81:  # 左箭头键
                self.index = max(self.index - 1, 0)
                self.load_image()
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
        self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, fps, frame_size)  # 创建视频写入对象

        # 遍历所有图像，按顺序写入视频
        for idx, _ in enumerate(self.image_list):
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
        out_dets = os.path.join(os.path.dirname(img_dir), f"det_hand_{os.path.basename(img_dir)}.pkl")

        if self.rotate_image:
           for k, v in self.det_dict.items():
               assert len(v) == 2, "more than two hands"
               if len(v[0]) == 2:
                   # rotate the images  
                   a, b = v[0]
                   v[0] = [b, self.width - a]
               if len(v[1]) == 2:
                   a, b = v[1]
                   v[1] = [b, self.width - a]

        with open(out_dets, "wb") as f:
            pickle.dump(self.det_dict, f)
        print(f"Hand det dictionary saved to {out_dets}")

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
    
    annotator = ImageAnnotator(img_folder, start, end)
    annotator.run()
