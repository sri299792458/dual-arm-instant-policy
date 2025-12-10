import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch

def visualize_sample_from_rlbench(data):
    num_samples = data['pcds'].shape[0]

    ## Add PC to the scene
    cmap = plt.get_cmap('viridis', num_samples)
    colors = [cmap(i)[:3] for i in range(num_samples)]
    pcds = []
    for i in range(num_samples):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data['pcds'][i])
        pcd.paint_uniform_color(colors[i])
        pcds.append(pcd)

    ## Add Gripper Poses
    gripper_left_frames = []
    gripper_right_frames = []
    for i in range(num_samples):
        T_left = data['T_w_es_left'][i]
        T_right = data['T_w_es_right'][i]

        frame_left = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        frame_right = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        frame_left.transform(T_left)
        frame_right.transform(T_right)
        gripper_left_frames.append(frame_left)
        gripper_right_frames.append(frame_right)
    
    ## Add world frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        pcds + gripper_left_frames + gripper_right_frames + [world_frame],
        window_name='Bimanual Lift Tray',
        mesh_show_back_face=True
    )

def visualize_sample_ip(data):
    num_demos = data['demo_scene_node_pos'].shape[1]
    traj_length = data['demo_scene_node_pos'].shape[2]

    cmap = plt.get_cmap('viridis', num_demos+traj_length+1)
    colors = [cmap(i)[:3] for i in range(num_demos+traj_length+1)]
    
    demo_pcds = []
    demo_gripper_left_frames = []
    demo_gripper_right_frames = []
    for i in range(1):
        for j in range(traj_length):
            ## Grippers
            T_left = data['demo_T_w_es_left'][0, i, j]
            T_right = data['demo_T_w_es_right'][0, i, j]
            # T_right = T_left @ T_right
            frame_left = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
            frame_right = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
            frame_left.transform(T_left)
            frame_right.transform(T_right)

            ## PCD
            pts = data['demo_scene_node_pos'][0, i, j]
            ## Conver pts to numpy array
            pts = pts.cpu().numpy()
            print(f"pts shape: {pts.shape}")
            ## Transform to world frame
            # pts = T_left[:3, :3] @ pts.T + T_left[:3, 3:4]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.paint_uniform_color(colors[j])

            demo_pcds.append(pcd)
            demo_gripper_left_frames.append(frame_left)
            demo_gripper_right_frames.append(frame_right)

    # ## Full demo
    # data_file_path = "/home/mohit/dual-arm-instant-policy/src/data/rl_bench_data/traj_0.npz"
    # data_rlbench = np.load(data_file_path, allow_pickle=True)

    # left_gripper_frames = []
    # for i in range(data_rlbench['T_w_es_left'].shape[0]):
    #     T_left = data_rlbench['T_w_es_left'][i]
    #     frame_left = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    #     frame_left.transform(T_left)
    #     left_gripper_frames.append(frame_left)

    
    ## Add world frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        [
            *demo_pcds,
            *demo_gripper_left_frames,
            *demo_gripper_right_frames,
            world_frame
        ],
    )


def load_rlbench_data(file_path):
    data = np.load(file_path, allow_pickle=True)

    print(f"Data loaded from {file_path}")
    print("Keys in the data file:")
    for k, v in data.items():
        print(f"Key: {k}, Shape: {v.shape}")
    return data

def load_ip_sample(file_path):
    data = torch.load(file_path)

    print(f"Data loaded from {file_path}")
    print("Keys in the data file:")
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(f"Key: {k}, Shape: {v.shape}")
        else:
            print(f"Key: {k}, Type: {type(v)}")
    return data

def main():
    # data_file_path = "/home/mohit/dual-arm-instant-policy/src/data/rl_bench_data/traj_1.npz"
    # data = load_rlbench_data(data_file_path)
    # visualize_sample_from_rlbench(data)

    data_file_path = "/home/mohit/dual-arm-instant-policy/src/data/bimanual_train/data_0.pt"
    data = load_ip_sample(data_file_path)
    visualize_sample_ip(data)

if __name__ == "__main__":
    main()