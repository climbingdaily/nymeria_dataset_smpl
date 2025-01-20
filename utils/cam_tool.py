import os
import pickle

import numpy as np
import cv2
import open3d as o3d
import matplotlib
from tqdm import tqdm
import gzip
import csv
import torch
import matplotlib.pyplot as plt
import h5py

import projectaria_tools.core.mps as mps
from projectaria_tools.core import calibration
from projectaria_tools.core.data_provider import VrsDataProvider
from projectaria_tools.core.mps.utils import filter_points_from_confidence, bisection_timestamp_search
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import (
    ImageData,
    ImageDataRecord,
    TimeDomain,
    TimeQueryOptions,
)

__all__ = ['project_point_cloud_to_image', 
           'remove_further_points_and_color', 
           'overlay_depth_on_image',
           'color_depth',
           'undistort_image']


def color_depth(depth, normalize=True, is_rot=False, max_depth=10):
    if normalize:
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    else:
        depth = np.minimum(depth, max_depth) / max_depth * 255.0
    depth = depth.astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    if is_rot:
        depth = np.rot90(depth, -1)
    return depth

def get_depth_model(encoder='vitl', is_metric=True, is_indoor=True):

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    dataset = 'hypersim' if is_indoor else 'vkitti'
    max_depth = 20 # 20 for indoor model, 80 for outdoor model
    if is_metric:
        module = __import__("Depth-Anything-V2.metric_depth.depth_anything_v2.dpt", fromlist=["DepthAnythingV2"])
        DepthAnythingV2 = module.DepthAnythingV2
        model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    else:
        module = __import__("Depth-Anything-V2.depth_anything_v2.dpt", fromlist=["DepthAnythingV2"])
        DepthAnythingV2 = module.DepthAnythingV2
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    return model

def get_stereo_image(vrs_dp: VrsDataProvider, t_ns: int, time_domain: TimeDomain = TimeDomain.TIME_CODE
) -> tuple[ImageData, ImageData, ImageData]:
    assert time_domain in [
        TimeDomain.DEVICE_TIME,
        TimeDomain.TIME_CODE,
    ], "unsupported time domain"

    if time_domain == TimeDomain.TIME_CODE:
        t_ns_device = vrs_dp.convert_from_timecode_to_device_time_ns(
            timecode_time_ns=t_ns
        )
    else:
        t_ns_device = t_ns

    # left image
    left_image_data, left_image_meta = vrs_dp.get_image_data_by_time_ns(
        StreamId("1201-1"),
        time_ns=t_ns_device,
        time_domain=TimeDomain.DEVICE_TIME,
        time_query_options=TimeQueryOptions.CLOSEST,
    )
    t_diff = t_ns_device - left_image_meta.capture_timestamp_ns

    right_image_data, right_image_meta = vrs_dp.get_image_data_by_time_ns(
        StreamId("1201-2"),
        time_ns=t_ns_device,
        time_domain=TimeDomain.DEVICE_TIME,
        time_query_options=TimeQueryOptions.CLOSEST,
    )
    t_diff = t_ns_device - right_image_meta.capture_timestamp_ns

    image_data, image_meta = vrs_dp.get_image_data_by_time_ns(
        StreamId("214-1"),
        time_ns=t_ns_device,
        time_domain=TimeDomain.DEVICE_TIME,
        time_query_options=TimeQueryOptions.CLOSEST,
    )

    t_diff = t_ns_device - image_meta.capture_timestamp_ns

    return left_image_data, left_image_meta, right_image_data, right_image_meta, image_data, image_meta

def get_intrinsic_from_calib(calib):
    cam = np.eye(3).astype(np.float32)
    cam[0, 0] = calib.get_focal_lengths()[0]
    cam[1, 1] = calib.get_focal_lengths()[1]
    cam[0, 2] = calib.get_principal_point()[0]
    cam[1, 2] = calib.get_principal_point()[1]
    return cam

def compute_depth_from_undistorted(left_image, right_image, left_calib, right_calib, label: str ="",is_show: bool=True):
    """
    Computes a depth map from already-undistorted stereo images using the extrinsic parameters from the right to the left camera.

    Parameters:
    - left_image (np.array): Undistorted left grayscale image.
    - right_image (np.array): Undistorted right grayscale image.
    - left_calib (calibration.Calibration): Calibration object for the left camera.
    - right_calib (calibration.Calibration): Calibration object for the right camera.
    - is_show (bool): Whether to show the rectified images. Defaults to True.

    Returns:
    - depth_map (np.array): The computed depth map.
    """

    K_1 = get_intrinsic_from_calib(left_calib)
    K_2 = get_intrinsic_from_calib(right_calib)
    R = right_calib.get_transform_device_camera().to_matrix()[:3, :3]
    T = right_calib.get_transform_device_camera().to_matrix()[:3, 3]
    f = K_1[0, 0]  # focal length
    B = np.linalg.norm(T)
    units = 0.512     # depth units, adjusted for the output to fit in one byte

    R1 = np.eye(3)
    P1 = K_1
    P2 = np.array([
            [K_2[0,0], 0, K_2[0,2], 0],
            [0, K_2[1,1], K_2[1,2], -f * B],
            [0, 0, 1, 0]
        ])
    # P2 = K_2
    R2 = R
    # R1, R2, P1, P2, Q, roi1, roi2  = cv2.stereoRectify(K_1, None, K_2, None, np.rot90(left_image, -1).shape[:2], R, T)
    map1, map2 = cv2.initUndistortRectifyMap(K_1, None, R1, P1, np.rot90(left_image, -1).shape[:2], cv2.CV_32FC1)
    map3, map4 = cv2.initUndistortRectifyMap(K_2, None, R2, P2, np.rot90(right_image, -1).shape[:2], cv2.CV_32FC1)

    # after rectifying
    rectified_left = cv2.remap(left_image, map1, map2, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_image, map3, map4, cv2.INTER_LINEAR)

    if is_show:
        cv2.imshow("Rectified main", np.rot90(rectified_left, -1))
        cv2.imshow(f"Rectified {label}", np.rot90(rectified_right, -1))

    # minDisparity = 0
    numDisparities = 64  # 适应较远的物体
    blockSize = 5        # 合适的匹配块大小
    # P1 = 8 * blockSize**2
    # P2 = 32 * blockSize**2

    # stereo = cv2.StereoSGBM_create(
    #     minDisparity=minDisparity,
    #     numDisparities=numDisparities,
    #     blockSize=blockSize,
    #     P1=P1,
    #     P2=P2,
    # )

    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity_map = stereo.compute(cv2.cvtColor(rectified_left, cv2.COLOR_RGB2GRAY),
                                    cv2.cvtColor(rectified_right, cv2.COLOR_RGB2GRAY))
    
    disparity_map = np.maximum(disparity_map, 0)
    # disparity_map = cv2.medianBlur(disparity_map, 3)
    disparity_map = cv2.GaussianBlur(disparity_map, (3, 3), 0)

    valid_pixels = disparity_map > 0
    depth_map = np.zeros(shape=rectified_left.shape[:2]).astype("uint8")
    depth_map[valid_pixels] = (f * B * 1e3) / (units * disparity_map[valid_pixels] + 1e-5)
    # disparity_map = cv2.GaussianBlur(disparity_map, (5, 5), 0)
    depth_map = cv2.equalizeHist(depth_map)
    colorized_depth = np.zeros((rectified_left.shape[0], rectified_left.shape[1], 3), dtype="uint8")
    temp = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    colorized_depth[valid_pixels] = temp[valid_pixels]

    # depth_map[depth_map > 10] = 10
    # depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    # depth_normalized = np.uint8(depth_normalized)
    # depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)


    disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_normalized = np.uint8(disparity_map_normalized)

    if is_show:
        cv2.imshow(f"Disparity Map {label}", np.rot90(cv2.applyColorMap(disparity_map_normalized, cv2.COLORMAP_INFERNO), -1))
        cv2.imshow(f"Depth Map {label}", np.rot90(colorized_depth, -1))

    return depth_map

def get_updated_calib(input_calib, label=""):
    return calibration.get_linear_camera_calibration(
        input_calib.get_image_size()[0], 
        input_calib.get_image_size()[1], 
        input_calib.get_focal_lengths()[0],
        label,
        input_calib.get_transform_device_camera())

def undistort_image(input_img, new_calib, input_calib, label="", is_show=False):
    undistorted_img = calibration.distort_by_calibration(input_img, new_calib, input_calib)
    if is_show:
        cv2.imshow(label, np.rot90(input_img, -1))
        cv2.imshow(f"Undistored {label}", np.rot90(undistorted_img, -1))
    return undistorted_img

def load_point_cloud(points_path:str, save_pc_path:str, inv_dist_std: float=0.0006, dist_std: float=0.01, voxel_size=0.01, max_point_count=100000):
    raw_pts = mps.read_global_point_cloud(points_path)
    filtered_points = filter_points_from_confidence(raw_pts, inv_dist_std, dist_std)
    pts = []
    for p in filtered_points:
        pts.append(p.position_world.astype(np.float32))
    pts = np.stack(pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
    pcd, _, _ = pcd.voxel_down_sample_and_trace(voxel_size, pcd.get_min_bound(), pcd.get_max_bound())  # remove the duplicate points
    o3d.io.write_point_cloud(save_pc_path, pcd)

    print(f"{len(pcd.points)} points after filtering.")

    points = mps.utils.filter_points_from_count(
        raw_points=[filtered_points[i] for i in ind], max_point_count=max_point_count
    )
    return pcd, points

def project_point_cloud_to_image(points, calib, extrnsics):
    w, h  = calib.get_image_size()
    intrinsic_matrix = get_intrinsic_from_calib(calib)
    R = extrnsics[:3, :3].astype(np.float32) # (3, 3)
    T = extrnsics[:3, 3:].astype(np.float32) # (3,3)

    camera_coordinates = R.dot(points.T) + T    # (3, n)

    z = camera_coordinates[2, :]
    valid_points = z > 0 

    uv = intrinsic_matrix @ camera_coordinates[:, valid_points]
    u, v = uv[0, :] / uv[2, :], uv[1, :] / uv[2, :]

    valid_uv = (u > 0) & (u < w) & (v > 0) & (v < h)

    return u[valid_uv], v[valid_uv], z[valid_points][valid_uv]

def remove_further_points_and_color(us, vs, zs, image_width, image_height):

    assert len(us) == len(vs) == len(zs), "Unexpected number of points for uv pairs."

    depth = np.zeros((image_height, image_width)).astype(np.float32)

    # 遍历所有投影点
    for u, v, z in zip(us, vs, zs):
        u_int, v_int = int(u), int(v)

        if 0 <= u_int < image_width and 0 <= v_int < image_height:

            if depth[v_int, u_int] == 0 or z < depth[v_int, u_int]:
                depth[v_int, u_int] = z
    
    return depth

def process_large_observations(observations_path: str, points, online_calibs):
    observations_by_frame = {}
    file_path = os.path.join(os.path.dirname(observations_path), "observations_by_frame.pkl.gz")
    if os.path.exists(file_path):
        try:
            with gzip.open(file_path, 'rb') as gz_file:
                observations_by_frame = pickle.load(gz_file)
            print(f"Observations loaded from {file_path}")
            return observations_by_frame
        except Exception as e:
                print(f"Error loading observations from {file_path}: {str(e)}")
                
    p_map = {}

    for pp in points: 
        p_map[pp.uid] = pp.position_world.astype(np.float32)

    # observations = mps.read_point_observations(observations_path)

    with gzip.open(observations_path, 'rt') as gz_file:  # Open .gz file in text mode
        reader = csv.reader(gz_file)
        header = next(reader)  # Read the header
        print(f"Header: {header}")

        with tqdm(desc="Processing observations", unit="row") as pbar:
            for i, obs in enumerate(reader):
                point_uid = int(obs[0])  # Convert the first column to an integer
                frame_capture_timestamp = int(obs[1])  # Convert to us timestamp
                camera_serial = obs[2]  # Third column is the camera serial
                uv = np.array([np.float32(obs[3]), np.float32(obs[4])])  # Last two columns are the UV coordinates as floats

                if point_uid in p_map: 
                    idx = bisection_timestamp_search(online_calibs, frame_capture_timestamp * 1e3)
                    if idx is not None:
                        ts = online_calibs[idx].tracking_timestamp.total_seconds() * 1e9
                        if idx not in observations_by_frame:
                            observations_by_frame[idx] = [[p_map[point_uid], uv, ts]]
                        else:
                            observations_by_frame[idx].append([p_map[point_uid], uv, ts])
               
                if i % 10000 == 0:  # Update tqdm every 1000 rows
                    pbar.update(10000)
    
    if len(observations_by_frame) > 0:
        with gzip.open(file_path, 'wb') as gz_file:
            pickle.dump(observations_by_frame, gz_file)
        print(f"Observations saved in {file_path}")

    return observations_by_frame


def overlay_depth_on_image(original_image, depth, min_d = 0.8, max_d = 30, size=3):
    if original_image.shape[:2] != depth.shape[:2]:
        raise ValueError("Original image and color image must have the same dimensions")
    
    
    mask = (depth > min_d) & (depth < max_d)
    depth_image = color_depth(depth)

    overlay_image = original_image.copy()
    # overlay_image[mask] = depth_image[mask]
    # Iterate through each point where the depth is valid (i.e., within the mask)
    for y in range(depth.shape[0]):
        for x in range(depth.shape[1]):
            if mask[y, x]:
                cv2.circle(overlay_image, (x, y), size, depth_image[y, x].tolist(), thickness=-1)

    depth_image[~mask] = 0

    return overlay_image, depth_image

def depth_point_cloud(depth, color_image, calib, T_w_c, outdir, filename, save_pcd=False):

    # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 1.0
    # Generate mesh grid and calculate point cloud coordinates
    h, w = color_image.shape[:2]
    fx, fy = calib.get_focal_lengths()
    cx, cy = calib.get_principal_point()
    x, y = np.meshgrid(np.arange(h), np.arange(w))
    x = (x - cx) / fx
    y = (y - cy) / fy
    z = np.array(depth)
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1)
    # colors = np.array(color_image).reshape(-1, 3) / 255.0

    w_cut = int(0.15*w)
    h_cut = int(0.15*h)
    points = points[h_cut:-h_cut, w_cut:-w_cut].reshape(-1, 3)
    colors = color_image[h_cut:-h_cut, w_cut:-w_cut].reshape(-1, 3) / 255.0

    # Filter points by z value
    mask = (points[..., 2] >= 0.5) & (points[..., 2] <= 8)
    points = points[mask].reshape(-1, 3)
    colors = colors[mask].reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.transform(T_w_c)
    if save_pcd:
        o3d.io.write_point_cloud(os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + ".pcd"), pcd)
    return pcd

    """
    # Extract x, y, z coordinates
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]

    # Create the plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud
    ax.scatter(x_coords, y_coords, z_coords, c=colors, s=0.1, alpha=0.6)

    # Apply camera transformation (move right by 4 meters and rotate 40°)
    # Translate camera 4 meters to the right (along x-axis)
    ax.view_init(elev=20, azim=40)  # Set the camera elevation and azimuth angles
    ax.set_xlim([x_coords.min(), x_coords.max()])
    ax.set_ylim([y_coords.min(), y_coords.max()])
    ax.set_zlim([z_coords.min(), z_coords.max()])

    # Add grid
    ax.grid(True)

    # Set labels (optional)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Save the figure as an image
    # output_path = os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + "_point_cloud.png")
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.show()
    """


def visualize_error(error):
    """
    Visualize a 1D array of errors sorted from small to large.

    Parameters:
        error (np.ndarray): 1D array of errors.
    """
    # Sort the errors
    sorted_error = np.sort(error)
    
    mean_value = np.mean(error)
    std_value = np.std(error)

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_error, label='Error (Sorted)', color='blue', marker='o', markersize=3, linestyle='-')
    plt.axhline(mean_value, color='red', linestyle='--', label=f'Mean = {mean_value:.2f}')
    plt.axhline(mean_value + std_value, color='green', linestyle='--', label=f'Mean + Std = {mean_value + std_value:.2f}')
    plt.axhline(mean_value - std_value, color='green', linestyle='--', label=f'Mean - Std = {mean_value - std_value:.2f}')
    
    # Add labels and grid
    plt.title('Errors per meter with Mean and Std')
    plt.xlabel('Index (Sorted)')
    plt.ylabel('Error Magnitude')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def get_depth_scale(depth_by_projection, depth_estimated, is_show=False):
    """
    Estimate scale (a) and offset (b) to transform predicted depth to approximate real depth.

    Args:
        depth_by_projection (numpy.ndarray): Sparse ground truth depth (non-zero where valid).
        depth_estimated (numpy.ndarray): Predicted depth.

    Returns:
        tuple: (scale, offset) where scale is `a` and offset is `b` in the equation `a * depth + b`.
    """
    from sklearn.linear_model import RANSACRegressor

    # Extract valid points where depth_by_projection is non-zero
    h, w = depth_by_projection.shape
    w_cut = int(0.15*w)
    h_cut = int(0.15*h)
    mask = (depth_by_projection > 1) & (depth_by_projection < 15)
    mask[:, :w_cut] = False
    mask[:, -w_cut:] = False
    mask[:h_cut] = False
    mask[-h_cut:] = False
    d_gt = depth_by_projection[mask]
    d_pred = depth_estimated[mask]
    for i in range(1):
        # Ensure there are valid points to compute
        if len(d_gt) < 20:
            # raise ValueError("No valid points found in depth_by_projection for scaling.")
            return None

        try:
            # Fit RANSAC regressor
            ransac = RANSACRegressor()
            ransac.fit(d_pred.reshape(-1, 1), d_gt.reshape(-1, 1))
            scale = ransac.estimator_.coef_[0][0]
            offset = ransac.estimator_.intercept_[0]
        except:
            # Real depth: d_gt = a * d_pred + b
            A = np.vstack([d_pred, np.ones_like(d_pred)]).T
            x, _, _, _ = np.linalg.lstsq(A, d_gt, rcond=None)  # Solve Ax = d_gt
            scale, offset = x

        depth_corrected = scale * depth_estimated + offset
        err = abs(scale * d_pred + offset - d_gt) / d_gt
        mask = (err > err.mean() - err.std()) & (err < err.mean() + err.std())
        d_gt = d_gt[mask]
        d_pred = d_pred[mask]
        if is_show:
            visualize_error(err)
        
        if scale < 0:
            print(f"scale is negative: {scale}")

    return depth_corrected
