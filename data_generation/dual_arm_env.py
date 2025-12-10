import mujoco.renderer
import mujoco.viewer
import numpy as np
import time
import robosuite as suite
import pprint
from robosuite.environments.base import register_env
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"  # Set the image convention to Open

import mujoco

## Load env base class
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.parts.arm.ik import InverseKinematicsController

import robosuite.utils.camera_utils as cam_utils
import robosuite.utils.transform_utils as transform_utils
from robosuite.utils.traj_utils import LinearInterpolator

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, BreadObject
from robosuite.models.tasks import Task

class DualArmEnv(TwoArmEnv):
    def __init__(self, **kwargs):
        ## Robot initial joint angles

        # thunder_init_pose = np.deg2rad(np.array([-90,-90,0,0,0,0]))
        # self.init_robot0_qpos = thunder_init_pose
        self.init_robot0_qpos = np.array([-1.932, 3.206, 2.166, -2.230, 3.631e-01, 8.599e-04])
        self.init_robot1_qpos = np.array([1.934, -6.427e-02, -2.166, -9.114e-01, -3.631e-01, -8.593e-04])

        super().__init__(**kwargs)

    def _load_model(self):
        super()._load_model()

        ## Load the arena
        self.arena = TableArena()

        ##################
        ## SETUP ROBOTS ##
        ##################
        ## Move the robots away from the center and set the orientations to look like lab setup
        for robot, offset in zip(self.robots, (-0.5, 0.5)):
            xpos = robot.robot_model.base_xpos_offset["empty"]
            xpos = np.array(xpos) + np.array((-0.1, offset, 0.2))
            robot.robot_model.set_base_xpos(xpos)
            robot.robot_model.set_base_ori(np.array([0, np.pi/2, np.pi/2]))

        ## Set base joint positions
        self.robots[0].init_qpos = self.init_robot0_qpos
        self.robots[1].init_qpos = self.init_robot1_qpos
        ############################################################################################


        ###################
        ## LOAD OBJECTS ##
        ###################
        
        box = BoxObject(
            name="box",
            size=(0.05, 0.05, 0.05),
            rgba=[0.8, 0.1, 0.1, 1.0],
        )
        box.get_obj().set("pos", "0 0 1.5")  # Set the position of the box slightly above the table
        
        bread = BreadObject(name="bread")
        bread.get_obj().set("pos", "0.2 0.2 1.5")  # Set the position of the bread slightly above the table

        ## Model is the task I think, it atleast needs arena and robots.
        self.model = Task(
            mujoco_arena=self.arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=[box, bread],
        )

    def transform_action(self, action):
        pass

    def step(self, action):

        T_robot0base_w = self.make_tranformation_matrix(
            pos=self.sim.data.get_body_xpos("robot0_base"),
            quat=self.sim.data.get_body_xquat("robot0_base")
        )

        T_robot1base_w = self.make_tranformation_matrix(
            pos=self.sim.data.get_body_xpos("robot1_base"),
            quat=self.sim.data.get_body_xquat("robot1_base")
        )

        ## Transform the actions to the world frame
        action[0:3] = T_robot0base_w[:3, :3].T @ action[0:3]  # Transform the position delta
        action[3:6] = T_robot0base_w[:3, :3] @ action[3:6]  # Transform the orientation delta


        ## Get the observations, reward, done, and info from the super class
        obs, reward, done, info = super().step(action)
        # print("Keys in Super Obs:", obs.keys())

        # ## Robot0 base Transformation Matrix
        # obs["robot0_T_base_w"] = self.make_tranformation_matrix(
        #     pos=self.sim.data.get_body_xpos("robot0_base"),
        #     quat=self.sim.data.get_body_xquat("robot0_base")
        # )
        # ## Robot1 base Transformation Matrix
        # obs["robot1_T_base_w"] = self.make_tranformation_matrix(
        #     pos=self.sim.data.get_body_xpos("robot1_base"),
        #     quat=self.sim.data.get_body_xquat("robot1_base")
        # )
        ## Set the base transformation matrices
        obs["robot0_T_base_w"] = T_robot0base_w
        obs["robot1_T_base_w"] = T_robot1base_w
        ## Robot0 end effector pose
        obs["robot0_T_w_eef"] = self.make_tranformation_matrix(
            pos=obs["robot0_eef_pos"],
            quat=obs["robot0_eef_quat"]
        )
        ## Robot1 end effector pose
        obs["robot1_T_w_eef"] = self.make_tranformation_matrix(
            pos=obs["robot1_eef_pos"],
            quat=obs["robot1_eef_quat"]
        )



        ## Save the transformation
        np.save("cache/robot0_T_base_w.npy", obs["robot0_T_base_w"])
        np.save("cache/robot1_T_base_w.npy", obs["robot1_T_base_w"])
        np.save("cache/robot0_T_w_eef.npy", obs["robot0_T_w_eef"])
        np.save("cache/robot1_T_w_eef.npy", obs["robot1_T_w_eef"])
        
        ## Solve for gripper state Open Or Close
        gripper0_qpos = obs["robot0_gripper_qpos"]
        gripper0_base = np.array([gripper0_qpos[0], gripper0_qpos[3]])
        obs["gripper0_state"] = int(gripper0_base.mean() > 0.3)  # 1 for open, 0 for closed Maybe

        gripper1_qpos = obs["robot1_gripper_qpos"]
        gripper1_base = np.array([gripper1_qpos[0], gripper1_qpos[3]])
        obs["gripper1_state"] = int(gripper1_base.mean() > 0.3)  # 1 for open, 0 for closed Maybe

        ## Process Depth observations
        ## Frontview Depth
        frontview_depth = obs["frontview_depth"] ## Shape: (480, 480, 1)
        frontview_depth = cam_utils.get_real_depth_map(self.sim, frontview_depth).squeeze(-1)  # Remove the last dimension if it's 1
        frontview_intrinsics = cam_utils.get_camera_intrinsic_matrix(self.sim, "frontview", 480, 480)
        mask = np.ones_like(frontview_depth, dtype=bool)  # Create a mask for the depth map
        frontview_pcd = self.get_pcd_from_depth(frontview_depth, mask, frontview_intrinsics)

        ## Conver PCD to world coordinates
        frontview_extrinsics = cam_utils.get_camera_extrinsic_matrix(self.sim, "frontview")
        T_w_cam_frontview = frontview_extrinsics
        frontview_pcd_world = T_w_cam_frontview @ np.hstack((frontview_pcd, np.ones((frontview_pcd.shape[0], 1)))).T
        frontview_pcd_world = frontview_pcd_world[:3, :].T

        ## Birdview Depth
        birdview_depth = obs["birdview_depth"]  ## Shape: (480, 480, 1)
        birdview_depth = cam_utils.get_real_depth_map(self.sim, birdview_depth).squeeze(-1)  # Remove the last dimension if it's 1
        birdview_intrinsics = cam_utils.get_camera_intrinsic_matrix(self.sim, "birdview", 480, 480)
        mask = np.ones_like(birdview_depth, dtype=bool)  # Create a mask for the depth map
        birdview_pcd = self.get_pcd_from_depth(birdview_depth, mask, birdview_intrinsics)
        
        birdview_extrinsics = cam_utils.get_camera_extrinsic_matrix(self.sim, "birdview")
        T_w_cam_birdview = birdview_extrinsics
        birdview_pcd_world = T_w_cam_birdview @ np.hstack((birdview_pcd, np.ones((birdview_pcd.shape[0], 1)))).T
        birdview_pcd_world = birdview_pcd_world[:3, :].T

        ## Agentview Depth
        agentview_depth = obs["agentview_depth"]  ## Shape: (480, 480, 1)
        agentview_depth = cam_utils.get_real_depth_map(self.sim, agentview_depth).squeeze(-1)  # Remove the last dimension if it's 1
        agentview_intrinsics = cam_utils.get_camera_intrinsic_matrix(self.sim, "agentview", 480, 480)
        mask = np.ones_like(agentview_depth, dtype=bool)  # Create a mask for the depth map
        agentview_pcd = self.get_pcd_from_depth(agentview_depth, mask, agentview_intrinsics)        

        agentview_extrinsics = cam_utils.get_camera_extrinsic_matrix(self.sim, "agentview")
        T_w_cam_agentview = agentview_extrinsics
        agentview_pcd_world = T_w_cam_agentview @ np.hstack((agentview_pcd, np.ones((agentview_pcd.shape[0], 1)))).T
        agentview_pcd_world = agentview_pcd_world[:3, :].T

        ## Robot0_robotview Depth
        robot0_robotview_depth = obs["robot0_robotview_depth"]  ## Shape: (480, 480, 1)
        robot0_robotview_depth = cam_utils.get_real_depth_map(self.sim, robot0_robotview_depth).squeeze(-1)  # Remove the last dimension if it's 1
        robot0_robotview_intrinsics = cam_utils.get_camera_intrinsic_matrix(self.sim, "robot0_robotview", 480, 480)
        mask = np.ones_like(robot0_robotview_depth, dtype=bool)  # Create a mask for the depth map
        robot0_robotview_pcd = self.get_pcd_from_depth(robot0_robotview_depth, mask, robot0_robotview_intrinsics)
        robot0_robotview_extrinsics = cam_utils.get_camera_extrinsic_matrix(self.sim, "robot0_robotview")
        T_w_cam_robot0_robotview = robot0_robotview_extrinsics
        robot0_robotview_pcd_world = T_w_cam_robot0_robotview @ np.hstack((robot0_robotview_pcd, np.ones((robot0_robotview_pcd.shape[0], 1)))).T
        robot0_robotview_pcd_world = robot0_robotview_pcd_world[:3, :].T

        ## Save the pcd and transform matrix for visualization
        # np.save("cache/frontview_pcd_world.npy", frontview_pcd_world)
        # np.save("cache/frontview_extrinsics.npy", frontview_extrinsics)
        # np.save("cache/birdview_pcd_world.npy", birdview_pcd_world)
        # np.save("cache/birdview_extrinsics.npy", birdview_extrinsics)
        # np.save("cache/agentview_pcd_world.npy", agentview_pcd_world)
        # np.save("cache/agentview_extrinsics.npy", agentview_extrinsics)
        # # np.save("cache/robotview_pcd_world.npy", robotview_pcd_world)
        # # np.save("cache/robotview_extrinsics.npy", robotview_extrinsics)
        # np.save("cache/robot0_robotview_pcd_world.npy", robot0_robotview_pcd_world)
        # np.save("cache/robot0_robotview_extrinsics.npy", robot0_robotview_extrinsics)

        # exit(0)
        return obs, reward, done, info
    
    def get_pcd_from_depth(self, depth_map, mask, intrinsics):
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        height, width = depth_map.shape
        ## Create a grid of pixel coordinates
        x_coords = np.arange(width)
        y_coords = np.arange(height)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        ## Only keep the points where mask is True
        z = depth_map[mask]
        x = (x_grid[mask] - cx) * z / fx
        y = (y_grid[mask] - cy) * z / fy
        pcd = np.stack((x, y, z), axis=-1)
        return pcd

    def make_tranformation_matrix(self, pos, quat):
        transform = np.eye(4)
        transform[:3, :3] = transform_utils.quat2mat(quat)
        transform[:3, 3] = pos
        return transform

    def reward(self, action=None):
        return 0.0 
    
    def _reset_internal(self):
        return super()._reset_internal()

def get_action(evn, obs):
    gripper_pose = obs["robot0_T_w_eef"][:3, 3]  # Get the end effector position of robot0
    ## Get the current box position.
    box_position = evn.sim.data.get_body_xpos("box_main")
    print(f"Box Position: {box_position}")
    print(f"Gripper Pose: {gripper_pose}")
    action = np.zeros((14,))
    action[:3] = (box_position - gripper_pose)
    print(f"Action Delta: {action[:3]}")
    return action

def main():
    ## Register the environment
    register_env(DualArmEnv)
    ## make a base environment
    env = suite.make(
        env_name="DualArmEnv",
        robots=["UR5e", "UR5e"],
        gripper_types=["Robotiq85Gripper", "Robotiq85Gripper"],
        has_renderer=True,
        controller_configs=None,
        render_camera=None, ## Options: "frontview", "birdview", "agentview"
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["frontview","birdview", "agentview", "robot0_robotview"],
        camera_heights=480,
        camera_widths=480,
        camera_depths=[True, True, True, True],  # Enable depth for both cameras
        camera_segmentations=["instance", "instance", "instance", "instance"],  # Options: None, 'instance', 'class', 'element'
    )
    xml_str = env.sim.model.get_xml()

    mj_model = mujoco.MjModel.from_xml_string(xml_str)
    mj_data = mujoco.MjData(mj_model)


    for cam_id in range(mj_model.ncam):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
        print(f"Camera {cam_id}: {name}")
    exit(0)

    i = 0
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        print("Viewer is running...")

        # Keep the window open until closed by user
        while viewer.is_running():
            mj_data.qpos[:] = i*0.01
            mujoco.mj_forward(mj_model, mj_data)
            time.sleep(0.01)  # Sleep to prevent busy-waiting
            viewer.sync()
            i+= 1

    
    # env.reset()
    # env.step(np.zeros(env.action_spec[0].shape))  # Initial step to set up the environment

    # print(f"ENV MODEL: {env.sim.model}")
    # print(f"ENV DATA: {env.sim.data}")

    # for _ in range(1000):
    #     # env.sim.data.qpos[:] = 0
    #     # env.robots[0].set_robot_joint_positions([0,0,0,0,0,0])
    #     # print(f"Robot0 Joint Positions: {env.robots[0].get_robot_joint_positions()}")
    #     # env.sim.forward()

    #     # action = np.random.randn(*env.action_spec[0].shape) * 0.1
    #     # action = np.zeros_like(action)  # Zero action for testing
    #     # obs, reward, done, info = env.step(action)
    #     time.sleep(0.1)
    #     env.render()

if __name__ == "__main__":
    main()