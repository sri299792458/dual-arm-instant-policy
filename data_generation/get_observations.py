import robosuite as suite


import robosuite.utils.camera_utils as cam_utils
import numpy as np
import open3d as o3d
import cv2

import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"  # Set the image convention to OpenCV for depth images
from dual_arm_env import DualArmEnv
from robosuite.environments.base import register_env
register_env(DualArmEnv)



demo_observations = {
    "pcds": [],
    "T_w_es": [],
    "thunder_T_w_eff": [],
    "lightning_T_w_eff": [],
    "thunder_grips": [],
    "lightning_grips": [],
}

robot_name_to_num = {
    "thunder" : 0,
    "lightning" : 1,
}


def main():
    env = suite.make(
        env_name="DualArmEnv",
        robots=["UR5e", "UR5e"],
        gripper_types=["Robotiq85Gripper", "Robotiq85Gripper"],
        has_renderer=True,
        render_camera="frontview", ##"frontview", "birdview", "agentview", "sideview", "robot0_robotview", "robot0_eye_in_hand", "robot1_robotview", "robot1_eye_in_hand"
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["frontview","birdview", "agentview", "robotview"],
        camera_heights=480,
        camera_widths=480,
        camera_depths=[True, True, True, True],  # Enable depth for both cameras
        camera_segmentations=["instance", "instance", "instance", "instance"],  # Options: None, 'instance', 'class', 'element'
    )
    env.reset()

    # # print("Num of objcts: ", env.sim.model.ngeom)
    # # print("Other stuff: ", dir(env.sim.model))
    # # print("EVN ids: ", env.sim.model.geom_id2name(0))
    # # print("Body names: ", env.sim.model.body_names)
    # # print("Num of bodies: ", env.sim.model.nbody)
    # # print("Body Parents: ", env.sim.model.body_parentid)
    # # print("Joint names: ", env.sim.model.joint_names)
    # # print("Geom names: ", env.sim.model.geom_names)
    # # for i in range(env.sim.model.ngeom):
    # #     print(f"Geom {i}: {env.sim.model.geom_names[i]}")
    # # print(": ", dir(env.sim.model.geom_names))
    # print("Cam Id: ", env.sim.model.camera_name2id("frontview"))
    # print("Camera intrinsics: ", env.sim.model.cam_intrinsic[0])
    # print("Camera body id: ", env.sim.model.cam_bodyid[0])
    # print("Camera cam_fovy: ", env.sim.model.cam_fovy[0])
    # print("camera ipd: ", env.sim.model.cam_ipd[0])
    # print("Camera pos: ", env.sim.model.cam_pos[0])
    # print("Camera quat: ", env.sim.model.cam_quat[0])
    # print("camera resolution: ", env.sim.model.cam_resolution[0])

    # print("Cam Pos: ", env.sim.data.cam_xpos[0])

    # print(dir(env.sim.model))

    # print("instance to id: ", env.model.instances_to_ids)
    global demo_observations, robot_name_to_num

    for _ in range(100):
        print("Step:", _)
        action = np.random.randn(*env.action_spec[0].shape) * 0.1
        action = np.zeros_like(action)  # Zero action for testing
        obs, reward, done, info = env.step(action)
        # save_obs(obs, env)
    
    # np.savez("demo_observations.npz", demo_observations)



def save_obs(obs, env):
    ## Maybe get meta instaed of evn
    global demo_observations, robot_name_to_num

    print("Observations keys:", obs.keys())

    ## End effector processing
    thunder_pose = obs["robot0_eef_pos"]
    thunder_quat = obs["robot0_eef_quat"]
    lightning_pose = obs["robot1_eef_pos"]
    lightning_quat = obs["robot1_eef_quat"]
    demo_observations["thunder_T_w_eff"].append(np.concatenate([thunder_pose, thunder_quat]))
    demo_observations["lightning_T_w_eff"].append(np.concatenate([lightning_pose, lightning_quat]))

    ## Gripper Pose
    thunder_gripper = obs["robot0_gripper_qpos"]
    lightning_gripper = obs["robot1_gripper_qpos"]
    demo_observations["thunder_grips"].append(thunder_gripper)
    demo_observations["lightning_grips"].append(lightning_gripper)


    ## Process Depth observations
    frontview_depth = obs["frontview_depth"].squeeze(-1)
    birdview_depth = obs["birdview_depth"].squeeze(-1)
    frontview_segmentation_instance = obs["frontview_segmentation_instance"].squeeze(-1)
    birdview_segmentation_instance = obs["birdview_segmentation_instance"].squeeze(-1)

    box_id = list(env.model.instances_to_ids).index("box")
    # box_mask = (frontview_segmentation_instance == box_id + 1)  # +1 as 0 is for background in segmentation
    box_mask = np.ones_like(frontview_depth, dtype=bool)  # Ensure the mask is boolean

    depth_proper_scale = cam_utils.get_real_depth_map(env.sim, frontview_depth)
    print("Depth proper scale shape:", depth_proper_scale.shape)

    frontview_depth = depth_proper_scale  # Remove the last dimension if it's a single channel

    box_pcd = depth_to_pcd(frontview_depth, box_mask, env.sim.model.cam_fovy[0])

    print("Max depth birdview:", np.max(birdview_depth))
    print("Min depth birdview:", np.min(birdview_depth))
    print("Max depth value:", np.max(frontview_depth))
    print("Min depth value:", np.min(frontview_depth))
    print("Max z in pcd:", np.max(box_pcd[:, 2]))
    print("Min z in pcd:", np.min(box_pcd[:, 2]))

    np.save("new_box_pcd.npy", box_pcd)  # Save the point cloud of the box
    exit(0)

    frontview_camera_pos = env.sim.model.cam_pos[env.sim.model.camera_name2id("frontview")]
    frontview_camera_quat = env.sim.model.cam_quat[env.sim.model.camera_name2id("frontview")]
    print("Quaternion of frontview camera:", frontview_camera_quat)

    T_w_cam_frontview = np.eye(4)
    ## Set the quaternion in right order
    frontview_camera_quat = np.array([frontview_camera_quat[1], frontview_camera_quat[2], frontview_camera_quat[3], frontview_camera_quat[0]])  # Convert from (x, y, z, w) to (y,
    T_w_cam_frontview[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(frontview_camera_quat)
    T_w_cam_frontview[:3, 3] = frontview_camera_pos

    np.save("T_w_cam_frontview.npy", T_w_cam_frontview)

    box_pcd_T = box_pcd.T
    box_pcd_T_homogeneous = np.vstack((box_pcd_T, np.ones(box_pcd_T.shape[1])))  # Convert to homogeneous coordinates
    box_pcd_transformed = np.linalg.inv(T_w_cam_frontview) @ box_pcd_T_homogeneous

    ## Drop the homogeneous coordinate
    box_pcd_transformed = box_pcd_transformed[:3, :].T  # Shape (N, 3)
    print("Frontview camera position:", frontview_camera_pos)
    ## Save the point cloud as a npy array
    print("Box PCD shape:", box_pcd.shape)
    np.save("box_pcd_transformed.npy", box_pcd_transformed)

    demo_observations["pcds"].append(...)

def depth_to_pcd(depth_image, mask, cam_fovy):
    height, width = depth_image.shape[0], depth_image.shape[1]
    fx = fy = width / (2 * np.tan(np.radians(cam_fovy) / 2))
    cx, cy = width / 2, height / 2

    print("Depth image shape:", depth_image.shape)
    print("Height:", height, "Width:", width)
    print("Cam fovy:", cam_fovy)
    print("Focal lengths (fx, fy):", fx, fy)

    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    Z = depth_image[mask]
    X = (X[mask] - cx) * Z / fx
    Y = (Y[mask] - cy) * Z / fy
    pcd = np.stack((X, Y, Z), axis=-1)  # Shape (N, 3)
    return pcd


def normalize_depth(depth_image):
    """
    Normalize the depth image to the range [0, 1].
    """
    depth_image = np.clip(depth_image, 0, 5)  # Clip to a reasonable range
    ## Normalize to [0, 1]
    depth_image = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))
    return depth_image


if __name__ == "__main__":
    main()