import time
import h5py
import open3d as o3d
import open3d.core as o3c
from tqdm import tqdm
import configargparse

def integrate_from_hdf5(hdf5_file, config):
    # 打开HDF5文件
    with h5py.File(hdf5_file, 'r') as f:
        depth_sequence = f['depth_sequence'][:]  # 深度图存储在'depth_sequence'数据集中
        rgb_sequence = f['rgb_sequence'][:] if config.integrate_color else None
        extrinsics = f['extrinsics'][:]  # 外参存储在'extrinsics'数据集中
        intrinsic = f['intrinsic'][:]  # 外参存储在'extrinsics'数据集中

    n_files = len(depth_sequence)

    device = o3d.core.Device(config.device)

    if config.integrate_color:
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=3.0 / 512,
            block_resolution=16,
            block_count=50000,
            device=device)
    else:
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight'),
            attr_dtypes=(o3c.float32, o3c.float32),
            attr_channels=((1), (1)),
            voxel_size=3.0 / 512,
            block_resolution=16,
            block_count=50000,
            device=device)

    start = time.time()
    for i in tqdm(range(n_files)):
        depth = o3d.t.geometry.Image(depth_sequence[i]).to(device)
        extrinsic = extrinsics[i]

        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, intrinsic, extrinsic, config.depth_scale,
            config.depth_max)

        if config.integrate_color:
            color = o3d.t.geometry.Image(rgb_sequence[i]).to(device)
            vbg.integrate(frustum_block_coords, depth, color, intrinsic,
                          intrinsic, extrinsic, config.depth_scale,
                          config.depth_max)
        else:
            vbg.integrate(frustum_block_coords, depth, intrinsic,
                          extrinsic, config.depth_scale, config.depth_max)
        dt = time.time() - start

    print('Finished integrating {} frames in {} seconds'.format(n_files, dt))

    return vbg


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add('--path_hdf5',
               help='Path to the HDF5 file containing depth, color, and extrinsics.',
               default='data.h5')
    parser.add('--path_npz',
               help='Path to the npz file that stores voxel block grid.',
               default='vbg.npz')
    config = parser.get_config()

    vbg = integrate_from_hdf5(config.path_hdf5, config)

    pcd = vbg.extract_point_cloud()
    o3d.visualization.draw([pcd])

    mesh = vbg.extract_triangle_mesh()
    o3d.visualization.draw([mesh.to_legacy()])
