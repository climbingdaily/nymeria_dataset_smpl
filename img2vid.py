import cv2
import os
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Convert images to video")
    parser.add_argument('-f', '--folder', type=str, required=True, help="Path to the folder containing images")
    return parser.parse_args()

def main():
    args = parse_args()
    image_folder = args.folder  

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    # sort images by timestamp in filename
    image_files.sort(key=lambda x: float(re.search(r'(\d+\.\d+).png', x).group(1)))

    if len(image_files) == 0:
        raise ValueError("No images found!")

    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video_filename = 'output_video.mp4'  
    fps = 30  

    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    for image_file in image_files:
        img = cv2.imread(os.path.join(image_folder, image_file))
        video_writer.write(img)

    video_writer.release()

    print(f"Video saved as {video_filename}")

if __name__ == "__main__":
    main()
