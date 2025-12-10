import numpy as np
import open3d as o3d
from scipy.spatial.transform import Slerp, Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm

## half dimensions of the object
## Tray Length 0.36m, width 0.21m, height 0.006m
## Tray Orientation: 22.5 degrees
OBJ_SIZE_X_MIN = 0.05 ## 5 cm
OBJ_SIZE_X_MAX = 0.15 ## 15 cm
OBJ_SIZE_Y_MIN = 0.15 ## 15 cm
OBJ_SIZE_Y_MAX = 0.25 ## 25 cm

## Not used
OBJ_SIZE_Z_MIN = 0.005 ## 5 mm
OBJ_SIZE_Z_MAX = 0.05 ## 5 cm

OBJ_ROT_RZ = np.pi/6  ## 30 degrees

WORKSPACE_MIN_X = 0.05 ## 5 cm
WORKSPACE_MAX_X = 0.45 ## 45 cm
WORKSPACE_MIN_Y = -0.2 ## -20
WORKSPACE_MAX_Y = 0.2 ## 20 cm
WORKSPACE_MIN_Z = 0.82 ## 82 cm
WORKSPACE_MAX_Z = 0.88 ## 88 cm

GRIPPER_DISPLACEMENT_MIN = 0.03 ## 3 cm inside the object from the edge
GRIPPER_DISPLACEMENT_MAX = 0.05 ## 5 cm inside the object from the edge
END_POSE_Z = 1.22
PRE_GRASP_OFFSET_MIN = 0.08
PRE_GRASP_OFFSET_MAX = 0.09

GLOBAL_FRAME = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])


T_LEFT_START = [
    [-0.96513554, -0.07748225, 0.25001978, 0.30894375],
    [-0.07267577, 0.99695079, 0.02841384, 0.06049896],
    [-0.25145898, 0.00925283, -0.96782373, 1.47196281],
    [0.0, 0.0, 0.0, 1.0],
]
T_RIGHT_START = [
    [9.70791886e-01, 2.14309644e-05, -2.39923141e-01, 2.45567814e-01],
    [2.67429970e-05, -9.99999999e-01, 1.88848819e-05, -2.41870582e-01],
    [-2.39923140e-01, -2.47495539e-05, -9.70791886e-01, 1.47204947e+00],
    [0.0, 0.0, 0.0, 1.0],
]

## Helper funtions to get Transfromation matrices for rotations
def rot_x(theta): return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

def rot_y(theta): return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

def rot_z(theta): return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

def get_transformation_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def generate_cuboid_pcd(size=[0.2, 0.2, 0.2], num_points=10000):
    """Input is the half dimensions of the cuboid"""
    # Create a cuboid mesh
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=size[0] * 2, height=size[1] * 2, depth=size[2] * 2
    )
    mesh.translate(np.array([-size[0], -size[1], -size[2]]))  # Center the cuboid at the origin
    # Sample points on the surface of the cuboid
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(pcd.points)

def generate_ellipsoid_pcd(size=[0.1, 0.2, 0.3], num_points=10000):
    """Input is the radii of the ellipsoid"""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh.scale(1.0, center=mesh.get_center())
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh.vertices) * np.array(size)
    )
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(pcd.points)

def viz_objs(objs):
    objs_o3d = []
    for obj in objs:
        ## if the object is a frame add it directly
        if isinstance(obj, o3d.geometry.TriangleMesh):
            objs_o3d.append(obj)
            continue
        ## else if it is a point cloud add it as a point cloud
        if isinstance(obj, o3d.geometry.PointCloud):
            objs_o3d.append(obj)
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj)
        pcd.paint_uniform_color(np.random.rand(3))
        objs_o3d.append(pcd)
    o3d.visualization.draw_geometries(objs_o3d + [GLOBAL_FRAME])

def make_grasp_poses(y_size, gripper_displacement):
    rot_right = rot_x(-np.pi/2) @ rot_z(np.pi)
    t_right = np.array([0.0, -y_size + gripper_displacement, 0.0])
    T_right = get_transformation_matrix(rot_right, t_right)

    rot_left = rot_x(np.pi/2)
    t_left = np.array([0.0, y_size - gripper_displacement, 0.0])
    T_left = get_transformation_matrix(rot_left, t_left)

    return T_right, T_left

def bezier_interpolate_poses(T_start, T_end, num_steps=100, bow=0.25, up=np.array([0,0,1.0])):
    """
    Cubic Bézier between T_start and T_end with a lateral 'bow'.
    - bow ~ 0.0 -> straight line
    - bow ~ 0.2–0.4 -> nice arc
    """
    R0, p0 = T_start[:3,:3], T_start[:3,3]
    R1, p1 = T_end[:3,:3],   T_end[:3,3]

    # Rotation: still do SLERP (smooth & robust)
    slerp = Slerp([0.0, 1.0], R.from_matrix([R0, R1]))
    ts = np.linspace(0.0, 1.0, num_steps)
    Rts = slerp(ts).as_matrix()

    # Translation: cubic Bézier with sideways offset
    d = p1 - p0
    if np.linalg.norm(d) < 1e-9:
        # Degenerate: just sit still
        pts = np.repeat(p0[None,:], num_steps, axis=0)
    else:
        # Build a sideways normal using 'up'. If nearly colinear, fall back to a fixed axis.
        n = np.cross(d, up)
        if np.linalg.norm(n) < 1e-6:
            n = np.cross(d, np.array([0,1,0.0]))
            if np.linalg.norm(n) < 1e-6:
                n = np.array([0,0,1.0])
        n = n / np.linalg.norm(n)

        # Control points: push both C1 and C2 by the same lateral offset
        offset = bow * np.linalg.norm(d) * n
        C0 = p0
        C1 = p0 + (1.0/3.0)*d + offset
        C2 = p0 + (2.0/3.0)*d + offset
        C3 = p1

        # Sample cubic Bézier
        t  = ts[:,None]
        omt = (1.0 - t)
        pts = (omt**3)*C0 + 3*(omt**2)*t*C1 + 3*omt*(t**2)*C2 + (t**3)*C3

    # Pack back into SE(3)
    traj = []
    for i in range(num_steps):
        T = np.eye(4)
        T[:3,:3] = Rts[i]
        T[:3, 3] = pts[i]
        traj.append(T)
    return traj

def interpolate_poses(T_start, T_end, num_steps=10):
    """Interpolate between two transformation matrices"""
    traj = []
    R_start = T_start[:3, :3]
    t_start = T_start[:3, 3]
    R_end = T_end[:3, :3]
    t_end = T_end[:3, 3]

    # Build key rotations and Slerp
    key_times = [0.0, 1.0]
    key_rots  = R.from_matrix([R_start, R_end])  # shape (2,)
    slerp     = Slerp(key_times, key_rots)

    times = np.linspace(0.0, 1.0, num_steps)
    rots  = slerp(times).as_matrix()             # shape (num_steps, 3, 3)

    traj = []
    for i, a in enumerate(times):
        T = np.eye(4)
        T[:3, :3] = rots[i]
        T[:3, 3]  = (1 - a) * t_start + a * t_end
        traj.append(T)
    return traj

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
        ## print size of plate
        points_np = np.asarray(pcd.points)
        x_min = np.min(points_np[:, 0])
        x_max = np.max(points_np[:, 0])
        y_min = np.min(points_np[:, 1])
        y_max = np.max(points_np[:, 1])
        z_min = np.min(points_np[:, 2])
        z_max = np.max(points_np[:, 2])
        print(f"Point Cloud {i}: x_size: {x_max - x_min:.3f}, y_size: {y_max - y_min:.3f}, z_size: {z_max - z_min:.3f}")

        break

    ## Add Gripper Poses
    gripper_left_frames = []
    gripper_right_frames = []
    for i in range(num_samples):
        T_left = data['T_w_es_left'][i]
        T_right = data['T_w_es_right'][i]
        frame_left = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        frame_right = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        if data['grips_left'][i] == 0.0: ## Closed
            frame_left.paint_uniform_color([1,1,0]) ## Yellow for closed
        if data['grips_right'][i] == 0.0: ## Closed
            frame_right.paint_uniform_color([1,0,1]) ## Yellow for closed
        # frame_right = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        frame_left.transform(T_left)
        frame_right.transform(T_right)
        gripper_left_frames.append(frame_left)
        gripper_right_frames.append(frame_right)
    
    ## Add world frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    return pcds + gripper_left_frames + gripper_right_frames

def load_rlbench_data(file_path):
    data = np.load(file_path, allow_pickle=True)

    print(f"Data loaded from {file_path}")
    print("Keys in the data file:")
    for k, v in data.items():
        print(f"Key: {k}, Shape: {v.shape}")
    return data

def generate_traj():
    ## Pick a random shape
    # shape = np.random.choice(['cuboid', 'ellipsoid'])
    shape = 'cuboid'

    ## Pick a random size of the object between the limits
    x_size = np.random.uniform(OBJ_SIZE_X_MIN, OBJ_SIZE_X_MAX)
    y_size = np.random.uniform(OBJ_SIZE_Y_MIN, OBJ_SIZE_Y_MAX)
    # z_size = np.random.uniform(OBJ_SIZE_Z_MIN, OBJ_SIZE_Z_MAX)
    z_size = 0.001 ## Make it a thin plate
    ## Pick a random position of the object within the workspace
    x_pos = np.random.uniform(WORKSPACE_MIN_X, WORKSPACE_MAX_X)
    y_pos = np.random.uniform(WORKSPACE_MIN_Y, WORKSPACE_MAX_Y)
    z_pos = np.random.uniform(WORKSPACE_MIN_Z, WORKSPACE_MAX_Z)

    pre_grasp_offset = np.random.uniform(PRE_GRASP_OFFSET_MIN, PRE_GRASP_OFFSET_MAX)
    gripper_displacement = np.random.uniform(GRIPPER_DISPLACEMENT_MIN, GRIPPER_DISPLACEMENT_MAX)

    ## Pick a random orientation of the object
    r_z = np.random.uniform(-OBJ_ROT_RZ, OBJ_ROT_RZ)

    ## Random Data for saving
    rand_values = {
        "x": x_pos,
        "y": y_pos,
        "z": z_pos,
        "theta_z": r_z,
        "half_width": x_size,
        "half_length": y_size,
        "half_height": z_size,
        "pre_grasp_offset": pre_grasp_offset,
        "gripper_displacement": gripper_displacement,
    }

    ## pick a random direction for the traj to bow towards
    up_dir_left = np.random.uniform(-1.0, 1.0, size=(3,))
    up_dir_left = up_dir_left / np.linalg.norm(up_dir_left)
    up_dir_right = np.random.uniform(-1.0, 1.0, size=(3,))
    up_dir_right = up_dir_right / np.linalg.norm(up_dir_right)

    bow_left = np.random.uniform(0.1, 0.25)
    bow_right = np.random.uniform(0.1, 0.25)

    ## Obj Transformation matrix
    R_obj = rot_z(r_z)
    t_obj = np.array([x_pos, y_pos, z_pos])
    T_obj = get_transformation_matrix(R_obj, t_obj)

    if shape == 'cuboid':
        pcd = generate_cuboid_pcd(size=[x_size, y_size, z_size], num_points=10000)
    elif shape == 'ellipsoid':
        pcd = generate_ellipsoid_pcd(size=[x_size, y_size, z_size], num_points=10000)

    ## Start of grippers
    T_right_start = np.array(T_RIGHT_START)
    T_left_start = np.array(T_LEFT_START)

    ## Generate grasp poses for both arms
    T_right_grasp, T_left_grasp = make_grasp_poses(y_size, gripper_displacement)

    ## Generate end poses for both arms
    T_right_end = T_right_grasp.copy()
    T_right_end[2,3] = END_POSE_Z
    T_left_end = T_left_grasp.copy()
    T_left_end[2,3] = END_POSE_Z

    ## Make Pregrasp Poses
    T_right_pregrasp = T_right_grasp.copy()
    T_right_pregrasp[1,3] -= pre_grasp_offset
    T_left_pregrasp = T_left_grasp.copy()
    T_left_pregrasp[1,3] += pre_grasp_offset

    ## Transform all poses to world frame
    pcd = (T_obj @ np.hstack((pcd, np.ones((pcd.shape[0], 1)))).T).T[:, :3]
    T_left_grasp = T_obj @ T_left_grasp
    T_right_grasp = T_obj @ T_right_grasp
    T_left_end = T_obj @ T_left_end
    T_left_end[2,3] = END_POSE_Z
    T_right_end = T_obj @ T_right_end
    T_right_end[2,3] = END_POSE_Z
    T_left_pregrasp = T_obj @ T_left_pregrasp
    T_right_pregrasp = T_obj @ T_right_pregrasp


    ## Interp between start to pregrasp
    right_traj_start_to_pre = bezier_interpolate_poses(T_right_start, T_right_pregrasp, num_steps=100, bow=bow_right, up=up_dir_right)
    left_traj_start_to_pre = bezier_interpolate_poses(T_left_start, T_left_pregrasp, num_steps=100, bow=bow_left, up=up_dir_left)
    ## Interp between pregrasp to grasp
    right_traj_pre_to_grasp = interpolate_poses(T_right_pregrasp, T_right_grasp, num_steps=20)
    left_traj_pre_to_grasp = interpolate_poses(T_left_pregrasp, T_left_grasp, num_steps=20)
    ## Interp between grasp to end
    right_traj_grasp_to_end = interpolate_poses(T_right_grasp, T_right_end, num_steps=100)
    left_traj_grasp_to_end = interpolate_poses(T_left_grasp, T_left_end, num_steps=100)

    full_right_traj = right_traj_start_to_pre + right_traj_pre_to_grasp + right_traj_grasp_to_end
    full_left_traj = left_traj_start_to_pre + left_traj_pre_to_grasp + left_traj_grasp_to_end

    full_right_traj = np.array(full_right_traj)
    full_left_traj = np.array(full_left_traj)

    ## Gripper Open Closes at grasp pose
    left_grips = np.ones((len(full_left_traj))) ## 1 means open
    right_grips = np.ones((len(full_right_traj))) ## 1 means open

    ## Copy PCD for each step in the traj
    pcds = np.tile(pcd[None, :, :], (len(full_right_traj), 1, 1))

    grasped_idxs = np.arange(
        len(right_traj_start_to_pre) + len(right_traj_pre_to_grasp),
        len(full_right_traj)
    )

    ## Move the pcd with the grippers after grasp
    ## Set z of point cloud to be the mid z of two grippers after grasp
    for idx in grasped_idxs:
        mid_z = (full_right_traj[idx][2,3] + full_left_traj[idx][2,3]) / 2.0
        pcds[idx,:,2] = mid_z
        left_grips[idx] = 0.0
        right_grips[idx] = 0.0

    traj = {
        'pcds': pcds,
        'T_w_es_right': full_right_traj,
        'T_w_es_left': full_left_traj,
        'grips_right': right_grips,
        'grips_left': left_grips
    }

    # ## Create trajectory visualization
    # frames = []
    # for i in range(len(full_right_traj)):
    #     T_right = full_right_traj[i]
    #     T_left = full_left_traj[i]
    #     right_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    #     if right_grips[i] == 0.0: ## Closed
    #         right_frame.paint_uniform_color([1,1,0]) ## Yellow for closed
    #     right_frame.transform(T_right)
    #     left_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    #     if left_grips[i] == 0.0: ## Closed
    #         left_frame.paint_uniform_color([1,0,1]) ## Magenta for closed
    #     left_frame.transform(T_left)
    #     frames.append(right_frame)
    #     frames.append(left_frame)

    # ##### Load one traj from rlbench also and viz
    # data_file_path = "/home/mohit/dual-arm-instant-policy/src/data/rl_bench_data/traj_1.npz"
    # data = load_rlbench_data(data_file_path)
    # data_frames = visualize_sample_from_rlbench(data)

    # frames.extend(data_frames)
    # frames.extend([pcds[0]])
    # viz_objs(frames)
    # return

    return traj, rand_values

def generate_traj_dataset(save_dir, num_trajs=200):
    all_rand_values = []
    for i in tqdm(range(num_trajs)):
        traj, rand_values = generate_traj()
        file_path = f"{save_dir}/traj_{i}.npz"
        np.savez_compressed(file_path, **traj)
        ## Save the rand values too
        all_rand_values.append(rand_values)
        print(f"Trajectory {i} saved to {file_path}")
    np.savez_compressed(f"{save_dir}/psuedo_data_200_random_values.npz", all_rand_values=all_rand_values)

if __name__ == "__main__":
    generate_traj_dataset(save_dir="src/data/pseudo_data_200", num_trajs=200)