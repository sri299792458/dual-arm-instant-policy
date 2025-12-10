import open3d as o3d 
import numpy as np

def main():
    ## Load the points clouds to visualize
    ## Frontview Transfromation Matrix
    T_w_cam_frontview = np.load("cache/frontview_extrinsics.npy")
    T_w_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    T_w_cam_frame.transform(T_w_cam_frontview)

    ## frontview pcd
    frontview_pcd_world = np.load("cache/frontview_pcd_world.npy")
    frontview_pcd_world_o3d = o3d.geometry.PointCloud()
    frontview_pcd_world_o3d.points = o3d.utility.Vector3dVector(frontview_pcd_world)
    frontview_pcd_world_o3d.paint_uniform_color([0.1, 0.1, 0.8])  # Blue color  

    ## Birdview Transfromation Matrix
    T_w_cam_birdview = np.load("cache/birdview_extrinsics.npy")
    T_w_cam_birdview_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    T_w_cam_birdview_frame.transform(T_w_cam_birdview)

    ## Birdview pcd
    birdview_pcd_world = np.load("cache/birdview_pcd_world.npy")
    birdview_pcd_world_o3d = o3d.geometry.PointCloud()
    birdview_pcd_world_o3d.points = o3d.utility.Vector3dVector(birdview_pcd_world)
    birdview_pcd_world_o3d.paint_uniform_color([0.8, 0.1, 0.1])  # Red color

    ## Agentview Transfromation Matrix
    T_w_cam_agentview = np.load("cache/agentview_extrinsics.npy")
    T_w_cam_agentview_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    T_w_cam_agentview_frame.transform(T_w_cam_agentview)

    ## Agentview pcd
    agentview_pcd_world = np.load("cache/agentview_pcd_world.npy")
    agentview_pcd_world_o3d = o3d.geometry.PointCloud()
    agentview_pcd_world_o3d.points = o3d.utility.Vector3dVector(agentview_pcd_world)
    agentview_pcd_world_o3d.paint_uniform_color([0.1, 0.8, 0.1])  # Green color

    # ## Robotview Transfromation Matrix
    # T_w_cam_robotview = np.load("cache/robotview_extrinsics.npy")
    # T_w_cam_robotview_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # T_w_cam_robotview_frame.transform(T_w_cam_robotview)    

    # ## Robotview pcd
    # robotview_pcd_world = np.load("cache/robotview_pcd_world.npy")
    # robotview_pcd_world_o3d = o3d.geometry.PointCloud()
    # robotview_pcd_world_o3d.points = o3d.utility.Vector3dVector(robotview_pcd_world)
    # robotview_pcd_world_o3d.paint_uniform_color([0.8, 0.8, 0.1])  # Yellow color

    ## Robot0 Robotview Transfromation Matrix
    T_w_cam_robot0_robotview = np.load("cache/robot0_robotview_extrinsics.npy")
    T_w_cam_robot0_robotview_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    T_w_cam_robot0_robotview_frame.transform(T_w_cam_robot0_robotview)

    ## Robot0 Robotview pcd
    robot0_robotview_pcd_world = np.load("cache/robot0_robotview_pcd_world.npy")
    robot0_robotview_pcd_world_o3d = o3d.geometry.PointCloud()
    robot0_robotview_pcd_world_o3d.points = o3d.utility.Vector3dVector(robot0_robotview_pcd_world)
    robot0_robotview_pcd_world_o3d.paint_uniform_color([0.8, 0.1, 0.8])  # Purple color

    ## Add additional frames.
    robot0_T_w_eef = np.load("cache/robot0_T_w_eef.npy")
    robot0_T_w_eef_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    robot0_T_w_eef_frame.transform(robot0_T_w_eef)

    robot0_T_base_w = np.load("cache/robot0_T_base_w.npy")
    robot0_T_base_w_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    robot0_T_base_w_frame.transform(robot0_T_base_w)

    ## Coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    ## Visualize the stuff
    o3d.visualization.draw_geometries(
        [
            frontview_pcd_world_o3d,
            T_w_cam_frame,
            #birdview_pcd_world_o3d,
            T_w_cam_birdview_frame,
            # agentview_pcd_world_o3d,
            T_w_cam_agentview_frame,
            # robotview_pcd_world_o3d,
            # T_w_cam_robotview_frame,
            # robot0_robotview_pcd_world_o3d,
            # T_w_cam_robot0_robotview_frame,
            robot0_T_base_w_frame,
            #robot0_T_w_eef_frame,
            coordinate_frame
        ]
    )

if __name__ == "__main__":
    main()