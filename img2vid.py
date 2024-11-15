import cv2
import os
import re
import argparse

# 使用 argparse 处理命令行输入参数
def parse_args():
    parser = argparse.ArgumentParser(description="Convert images to video")
    parser.add_argument('-f', '--folder', type=str, required=True, help="Path to the folder containing images")
    return parser.parse_args()

def main():
    # 获取命令行参数
    args = parse_args()
    image_folder = args.folder  # 获取输入的文件夹路径

    # 获取文件夹中的所有图像文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    # 按时间戳排序
    image_files.sort(key=lambda x: float(re.search(r'(\d+\.\d+)_rgb_overlay.png', x).group(1)))

    # 确保至少有一张图像
    if len(image_files) == 0:
        raise ValueError("No images found!")

    # 读取第一张图像以获得视频的尺寸
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape

    # 设置视频输出参数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video_filename = 'output_video.mp4'  # 输出视频文件名
    fps = 30  # 设置每秒帧数

    # 创建视频写入对象
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    # 将每张图像写入视频
    for image_file in image_files:
        img = cv2.imread(os.path.join(image_folder, image_file))
        video_writer.write(img)

    # 释放视频写入对象
    video_writer.release()

    print(f"Video saved as {video_filename}")

if __name__ == "__main__":
    main()
