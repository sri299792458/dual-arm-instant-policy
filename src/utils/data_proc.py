import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
from ip.utils.common_utils import transform_pcd
from ip.utils.data_proc import pose_error, subsample_pcd
import torch

from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

def save_sample(sample, save_dir=None, offset=0, scene_encoder=None):
    # First, subsample demo pcds and concatenate them.
    joint_demo_pcd = []
    # joint_demo_grasp = []
    joint_demo_grasp_left = []
    joint_demo_grasp_right = []
    batch_indices = []

    num_demos = len(sample['demos']) ## Will Mostly be 2.
    num_traj_waypoints = len(sample['demos'][0]['obs'])
    num_points = len(sample['live']['obs'][0])
    for n in range(len(sample['demos'])):
        for i in range(len(sample['demos'][n]['obs'])):
            joint_demo_pcd.append(sample['demos'][n]['obs'][i])
            joint_demo_grasp_left.append(sample['demos'][n]['grips_left'][i])
            joint_demo_grasp_right.append(sample['demos'][n]['grips_right'][i])
            batch_indices.append(np.zeros(len(sample['demos'][n]['obs'][i])) + i + n * len(sample['demos'][n]['obs']))

    joint_demo_pcd = np.concatenate(joint_demo_pcd)
    joint_demo_grasp_left = (np.array(joint_demo_grasp_left) - 0.5) * 2
    joint_demo_grasp_right = (np.array(joint_demo_grasp_right) - 0.5) * 2
    batch_indices = np.concatenate(batch_indices)

    data = Data(
        pos_demos = torch.tensor(joint_demo_pcd, dtype=torch.float32),
        grasp_left_demo = torch.tensor(joint_demo_grasp_left, dtype=torch.float32).view(
            num_demos,
            num_traj_waypoints,
            1,
        ).unsqueeze(0),
        grasp_right_demo = torch.tensor(joint_demo_grasp_right, dtype=torch.float32).view(
            num_demos,
            num_traj_waypoints,
            1,
        ).unsqueeze(0),
        batch_demos = torch.tensor(batch_indices, dtype=torch.int64),
        batch_pos_obs = torch.tensor(np.zeros(num_points), dtype=torch.int64),
        demo_T_w_es_left = torch.tensor(np.stack([sample['demos'][n]['T_w_es_left'] for n in range(num_demos)]),
                                        dtype=torch.float32).unsqueeze(0),
        demo_T_w_es_right = torch.tensor(np.stack([sample['demos'][n]['T_w_es_right'] for n in range(num_demos)]),
                                            dtype=torch.float32).unsqueeze(0),
    )

    if scene_encoder is not None:
        bs = 1
        demo_scene_node_embds, demo_scene_node_pos, demo_scene_node_batch = \
            scene_encoder(None,
                          data.pos_demos.to(next(scene_encoder.parameters()).device),
                            data.batch_demos.to(next(scene_encoder.parameters()).device))
        
        demo_scene_node_embds = to_dense_batch(demo_scene_node_embds, demo_scene_node_batch, fill_value=0)[0]
        demo_scene_node_embds = demo_scene_node_embds.view(
            bs,
            num_demos,
            num_traj_waypoints,
            -1,
            scene_encoder.embd_dim,
        )
        demo_scene_node_pos = to_dense_batch(demo_scene_node_pos, demo_scene_node_batch, fill_value=0)[0]
        demo_scene_node_pos = demo_scene_node_pos.view(
            bs,
            num_demos,
            num_traj_waypoints,
            -1,
            3,
        )
        data.demo_scene_node_embds = demo_scene_node_embds.detach().cpu()
        data.demo_scene_node_pos = demo_scene_node_pos.detach().cpu()

        data.pos_demos = None
        data.batch_demos = None
        # data.batch_pos_obs = None
    
    k = 0
    for i in range(len(sample['live']['obs'])):
        data.pos_obs = torch.tensor(sample['live']['obs'][i], dtype=torch.float32)
        if scene_encoder is not None:
            live_scene_node_embds, live_scene_node_pos, live_scene_node_batch = \
                scene_encoder(None,
                              data.pos_obs.to(next(scene_encoder.parameters()).device),
                              torch.tensor(np.zeros(num_points),
                                           dtype=torch.int64).to(next(scene_encoder.parameters()).device))
            data.live_scene_node_embds = live_scene_node_embds.detach().cpu().unsqueeze(0)
            data.live_scene_node_pos = live_scene_node_pos.detach().cpu().unsqueeze(0)
            # data.pos_obs = None

        data.current_grip_left = torch.tensor(sample['live']['grips_left'][i], dtype=torch.float32).unsqueeze(0)
        data.current_grip_right = torch.tensor(sample['live']['grips_right'][i], dtype=torch.float32).unsqueeze(0)
        data.current_grip_left = (data.current_grip_left - 0.5) * 2
        data.current_grip_right = (data.current_grip_right - 0.5) * 2
        data.actions_left = torch.tensor(sample['live']['actions_left'][i], dtype=torch.float32).unsqueeze(0)
        data.actions_right = torch.tensor(sample['live']['actions_right'][i], dtype=torch.float32).unsqueeze(0)
        data.actions_grip_left = torch.tensor(sample['live']['actions_grip_left'][i], dtype=torch.float32).unsqueeze(0)
        data.actions_grip_right = torch.tensor(sample['live']['actions_grip_right'][i], dtype=torch.float32).unsqueeze(0)
        data.actions_grip_left = (data.actions_grip_left - 0.5) * 2
        data.actions_grip_right = (data.actions_grip_right - 0.5) * 2
        data.T_w_es_left = torch.tensor(sample['live']['T_w_es_left'][i], dtype=torch.float32).unsqueeze(0)
        data.T_w_es_right = torch.tensor(sample['live']['T_w_es_right'][i], dtype=torch.float32).unsqueeze(0)
        if save_dir is not None:
            torch.save(data, f'{save_dir}/data_{k + offset}.pt')
            k += 1
        else:
            return data


def sample_to_live(sample, pred_horizon, num_points=2048, trans_space=0.01, rot_space=3, subsample=True):
    if subsample:
        sample['T_w_es_left'], sample['T_w_es_right'], sample['grips_left'], sample['grips_right'], sample['pcds'] = subsample_traj(
            sample['T_w_es_left'],
            sample['T_w_es_right'],
            sample['grips_left'],
            sample['grips_right'],
            pcds=sample['pcds'],
        )
    
    live_data = {
        'obs': [],
        'actions_left': [],
        'actions_right': [],
        'actions_grip_left': [],
        'actions_grip_right': [],
    }
    live_data['T_w_es_left'] = sample['T_w_es_left']
    ## Below transforms T_w_es_right to T_w_es_left frame.
    # live_data['T_left_es_right'] = [np.linalg.inv(sample['T_w_es_left'][i]) @ sample['T_w_es_right'][i] for i in range(len(sample['T_w_es_left']))]
    live_data['T_w_es_right'] = sample['T_w_es_right']
    # live_data['obs'] = [transform_pcd(subsample_pcd(sample['pcds'][idx], num_points),
    #                                   np.linalg.inv(sample['T_w_es_left'][idx])) for idx in range(len(sample['pcds']))]
    live_data['obs'] = [subsample_pcd(sample['pcds'][idx], num_points) for idx in range(len(sample['pcds']))]
    live_data['grips_left'] = sample['grips_left']
    live_data['grips_right'] = sample['grips_right']

    for i in range(len(sample['pcds'])):
        actions_left = []
        actions_right = []
        actions_grip_left = []
        actions_grip_right = []
        for j in range(1, pred_horizon + 1):
            if i + j < len(sample['pcds']):
                ## Below transforms the next pose to the current gripper's frame.
                actions_left.append(
                    np.linalg.inv(sample['T_w_es_left'][i]) @ sample['T_w_es_left'][i + j])
                actions_right.append(
                    np.linalg.inv(sample['T_w_es_right'][i]) @ sample['T_w_es_right'][i + j])
                actions_grip_left.append(sample['grips_left'][i + j])
                actions_grip_right.append(sample['grips_right'][i + j])
            else:
                ## Identity means no relative motion between current gripper and the next gripper.
                actions_left.append(np.eye(4))
                actions_right.append(np.eye(4))
                actions_grip_left.append(sample['grips_left'][-1])
                actions_grip_right.append(sample['grips_right'][-1])
        live_data['actions_left'].append(np.array(actions_left))
        live_data['actions_right'].append(np.array(actions_right))
        live_data['actions_grip_left'].append(actions_grip_left)
        live_data['actions_grip_right'].append(actions_grip_right)
    return live_data


def subsample_traj(traj_left, traj_right, grips_left, grips_right, pcds=None, trans_space=0.01, rot_space=3):
    ## Put the first point in the trajectory.
    subsampled_traj_left = [traj_left[0].copy()]
    subsampled_traj_right = [traj_right[0].copy()]
    subsampled_grips_left = [grips_left[0]]
    subsampled_grips_right = [grips_right[0]]

    if pcds is not None:
        subsampled_pcds = [pcds[0]]

    i = 1
    while i < len(traj_left):
        trans_dist_left = np.linalg.norm(traj_left[i][:3, 3] - subsampled_traj_left[-1][:3, 3])
        trans_dist_right = np.linalg.norm(traj_right[i][:3, 3] - subsampled_traj_right[-1][:3, 3])
        rot_dist_left = np.linalg.norm(
            Rot.from_matrix(traj_left[i][:3, :3] @ subsampled_traj_left[-1][:3, :3].T).as_rotvec(degrees=True))
        rot_dist_right = np.linalg.norm(
            Rot.from_matrix(traj_right[i][:3, :3] @ subsampled_traj_right[-1][:3, :3].T).as_rotvec(degrees=True))
        
        if trans_dist_left < trans_space and trans_dist_right < trans_space and \
           rot_dist_left < rot_space and rot_dist_right < rot_space and \
           grips_left[i] == subsampled_grips_left[-1] and grips_right[i] == subsampled_grips_right[-1]:
            ## All the distances are inside the threshold and gripper states are also not 
            ## changing, so we skip this point. This is what will remove very close points in the trajectory
            ## and will make the distance consistent.
            i += 1
        elif (trans_dist_left > trans_space or trans_dist_right > trans_space or \
              rot_dist_left > rot_space or rot_dist_right > rot_space):
            ## If any of the distance is greater than the threshold, we add this point to the subsampled trajectory.
            ## Placeholder for the new waypoint.
            new_wp_left = subsampled_traj_left[-1].copy()
            new_wp_right = subsampled_traj_right[-1].copy()

            ## Calculate the error/deltas between the current and the previous point.
            err_left = traj_left[i][:3, 3] - subsampled_traj_left[-1][:3, 3]
            err_right = traj_right[i][:3, 3] - subsampled_traj_right[-1][:3, 3]
            rot_vec_delta_left = Rot.from_matrix(subsampled_traj_left[-1][:3, :3].T @ traj_left[i][:3, :3]).as_rotvec()
            rot_vec_delta_right = Rot.from_matrix(subsampled_traj_right[-1][:3, :3].T @ traj_right[i][:3, :3]).as_rotvec()

            ## Get the scale by which they need to be scaled.
            factor_left = trans_dist_left / trans_space
            factor_right = trans_dist_right / trans_space
            factor_rot_left = rot_dist_left / rot_space
            factor_rot_right = rot_dist_right / rot_space
            factors = np.array([factor_left, factor_rot_left, factor_right, factor_rot_right])

            ## switch condition based on which of the factor is greater
            if np.isclose(factor_left, factors.max()):
                ## If left gripper is the max item.
                new_wp_left[:3, 3] += trans_space * err_left / np.linalg.norm(err_left)

                new_wp_right[:3, 3] += err_right / factor_left
                rot_vec_delta_left = rot_vec_delta_left / factor_left
                rot_vec_delta_right = rot_vec_delta_right / factor_left
            elif np.isclose(factor_rot_left, factors.max()):
                rot_vec_delta_left = np.deg2rad(rot_space) * rot_vec_delta_left / np.linalg.norm(rot_vec_delta_left)

                new_wp_left[:3, 3] += err_left / factor_rot_left
                new_wp_right[:3, 3] += err_right / factor_rot_left
                rot_vec_delta_right = rot_vec_delta_right / factor_rot_left
            elif np.isclose(factor_right, factors.max()):
                ## If right gripper is the max item.
                new_wp_right[:3, 3] += trans_space * err_right / np.linalg.norm(err_right)

                new_wp_left[:3, 3] += err_left / factor_right
                rot_vec_delta_left = rot_vec_delta_left / factor_right
                rot_vec_delta_right = rot_vec_delta_right / factor_right
            elif np.isclose(factor_rot_right, factors.max()):
                rot_vec_delta_right = np.deg2rad(rot_space) * rot_vec_delta_right / np.linalg.norm(rot_vec_delta_right)

                new_wp_left[:3, 3] += err_left / factor_rot_right
                new_wp_right[:3, 3] += err_right / factor_rot_right
                rot_vec_delta_left = rot_vec_delta_left / factor_rot_right
                
            new_wp_left[:3, :3] = new_wp_left[:3, :3] @ Rot.from_rotvec(rot_vec_delta_left).as_matrix()
            new_wp_right[:3, :3] = new_wp_right[:3, :3] @ Rot.from_rotvec(rot_vec_delta_right).as_matrix()

            subsampled_traj_left.append(new_wp_left)
            subsampled_traj_right.append(new_wp_right)
            subsampled_grips_left.append(subsampled_grips_left[-1])
            subsampled_grips_right.append(subsampled_grips_right[-1])
            if pcds is not None:
                subsampled_pcds.append(pcds[i])
        else:
            ## If the distances are small but the gripper state has changed.
            subsampled_traj_left.append(traj_left[i])
            subsampled_traj_right.append(traj_right[i])
            subsampled_grips_left.append(grips_left[i])
            subsampled_grips_right.append(grips_right[i])
            if pcds is not None:
                subsampled_pcds.append(pcds[i])
            i += 1
        
    ## Save the last point in the trajectory.
    subsampled_traj_left.append(traj_left[-1])
    subsampled_traj_right.append(traj_right[-1])
    subsampled_grips_left.append(grips_left[-1])
    subsampled_grips_right.append(grips_right[-1])
    if pcds is not None:
        subsampled_pcds.append(pcds[-1])
        return subsampled_traj_left, subsampled_traj_right, subsampled_grips_left, subsampled_grips_right, subsampled_pcds
    return subsampled_traj_left, subsampled_traj_right, subsampled_grips_left, subsampled_grips_right
        


def sample_to_cond_demo(sample, num_waypoints, num_points=2048):
    traj_indices = extract_waypoints(
        np.array(sample['T_w_es_left']),
        np.array(sample['T_w_es_right']),
        np.array(sample['grips_left']),
        np.array(sample['grips_right']),
        num_waypoints=num_waypoints,
    )
    ## Keep everything in the left gripper's frame
    # pcds = [transform_pcd(subsample_pcd(sample['pcds'][idx], num_points),
    #                       np.linalg.inv(sample['T_w_es_left'][idx]))
    #                       for idx in traj_indices]
    ## Subsample pcds but keep them in global frame.
    pcds = [subsample_pcd(sample['pcds'][idx], num_points) for idx in traj_indices]
    
    # ## Convert T_w_es_right to T_w_es_left frame.
    # T_left_es_right = [np.linalg.inv(sample['T_w_es_left'][idx]) @ sample['T_w_es_right'][idx] for idx in traj_indices]

    demo_sample = {
        'obs' : pcds,
        'T_w_es_left' : [sample['T_w_es_left'][idx] for idx in traj_indices],
        # 'T_left_es_right' : T_left_es_right,
        'T_w_es_right' : [sample['T_w_es_right'][idx] for idx in traj_indices],
        'grips_left' : [sample['grips_left'][idx] for idx in traj_indices],
        'grips_right' : [sample['grips_right'][idx] for idx in traj_indices],
    }

    return demo_sample


def extract_waypoints(traj_left, traj_right, traj_left_states, traj_right_states, num_waypoints):
    ## Add the first and last waypoints
    waypoints = [0, len(traj_left) - 1]

    ## Add all waypoints where gripper state changed
    for i in range(1, len(traj_left_states) - 2):
        if (traj_left_states[i] != traj_left_states[i - 1] or traj_right_states[i] != traj_right_states[i - 1]) and i not in waypoints:
            waypoints.append(i)
    waypoints.sort()

    ## Add more waypoints where trajectory is close to the previous waypoint to smooth out the trajectory
    for i in range(1, len(traj_left_states)):
        dist_left = pose_error(traj_left[i], traj_left[i - 1], rot_scale=1)
        dist_right = pose_error(traj_right[i], traj_right[i - 1], rot_scale=1)
        closest_waypoint = np.argmin([abs(i - w) for w in waypoints])
        ## Here we are kind of checking if the gripper slowed down at this point, so I think this is 
        ## imp step for both grippers. If any on the gripper has slowed down that means it's a important waypoint.
        if (dist_left < 3.5e-3 or dist_right < 3.5e-3) and abs(i - waypoints[closest_waypoint]) > 5 and (len(waypoints) < num_waypoints):
            waypoints.append(i)
    
    waypoints.sort()

    ## If there are still too few waypoints, add more. Otherwise, exit.
    wp_to_add = max(0, num_waypoints - len(waypoints))
    if wp_to_add == 0:
        return waypoints

    wp_segment_dist = [(pose_error(traj_left[waypoints[i]], traj_left[waypoints[i + 1]], rot_scale=1) + 
                        pose_error(traj_right[waypoints[i]], traj_right[waypoints[i + 1]], rot_scale=1)) / 2
                        for i in range(len(waypoints) - 1)]
    wp_segment_dist = np.array(wp_segment_dist) / np.sum(wp_segment_dist)
    extra_wp_segments = np.ceil(wp_segment_dist * wp_to_add).astype(int)

    seg_order = np.argsort(extra_wp_segments)[::-1]
    for ii in range(len(extra_wp_segments)):
        i = seg_order[ii]
        wp_a = waypoints[i]
        wp_b = waypoints[i + 1]
        for j in range(extra_wp_segments[i]):
            waypoints.append(wp_a + (wp_b - wp_a) * (j + 1) // (extra_wp_segments[i] + 1))
            if len(waypoints) == num_waypoints:
                break
        else: 
            continue
        break
    waypoints.sort()
    return waypoints


