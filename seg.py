import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import torch

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_sam import build_sam2_video_predictor

# get current file's directory 
current_dir = os.path.dirname(os.path.realpath(__file__))
SAM_ROOT = os.path.join(current_dir, 'ThirdParties', 'sam2')

sam2_checkpoint = f"{SAM_ROOT}/checkpoints/sam2.1_hiera_large.pt"
model_cfg = f"configs/sam2.1/sam2.1_hiera_l.yaml"

assert os.path.exists(sam2_checkpoint), f"checkpoint not found: {sam2_checkpoint}"
# assert os.path.exists(model_cfg), f"model config not found: {model_cfg}"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax, obj_id=None):
    cmap = plt.get_cmap("tab10")
    cmap_idx = 0 if obj_id is None else obj_id
    color = np.array([*cmap(cmap_idx)[:3], 0.6])
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2))


video_dir = '/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd/recording_head/imgs_1'

frame_names = os.listdir(video_dir)
frame_names.sort(key=lambda p: float(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path=video_dir)

lefthand_id = 1
righthand_id = 2

param_file = os.path.join('/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd', 'synced_data', 'humans_param.pkl')

with open(param_file, "rb") as f:
    save_data = pickle.load(f)
    fp_data = save_data['first_person']
    first_pose, first_tran, bboxes = fp_data['pose'], fp_data['trans'], fp_data['bboxes']

width = 1024
height = 1024

IDX_START = 1050
IDX_END = 1050 + 943

obj_idx = {'leftHand': 1, 'rightHand': 2, 'left_arm': 3, 'right_arm': 4}

for idx in range(0, len(frame_names), 3):
    # points = np.array([[768, 664]], dtype=np.float32)
    prompts = {}  # hold all the clicks we add for visualization

    for parts, item in bboxes.items():
        if parts not in obj_idx:
            continue
        x1, y1, x2, y2 = item[idx + IDX_START]
        box = np.array([height - y2, x1, height- y1, x2], dtype=np.float32) # rotate the image for 90 degrees
        prompts[obj_idx[parts]] = box

        # for labels, `1` means positive click and `0` means negative click
        # labels = np.array([1], np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=idx,
            obj_id=obj_idx[parts],
            box=box,
            # points=points,
            # labels=labels,
        )

    # show the results on the current (interacted) frame
    plt.figure(figsize=(6, 6))
    plt.title(f"frame {idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[idx])))
    # show_points(points, labels, plt.gca())
    # show_box(box, plt.gca())
    for i, out_obj_id in enumerate(out_obj_ids):
        # show_points(*prompts[out_obj_id], plt.gca())
        show_box(prompts[out_obj_id], plt.gca(), out_obj_id)
        show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)