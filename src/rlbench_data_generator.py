from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.bimanual_tasks.bimanual_lift_tray import BimanualLiftTray
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from ip.utils.common_utils import downsample_pcd, pose_to_transform, transform_to_pose

from utils import observation_utils
from utils.data_proc import sample_to_cond_demo, subsample_pcd, save_sample

from tqdm import trange
import torch
import time

from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape

import numpy as np
from tqdm import tqdm

def rollout_model(model, num_demos, task_name='bimanual_lift_tray', max_execution_steps=30,
                  execution_horizon=8, num_rollouts=2, headless=False, num_traj_wp=10, restrict_rot=True):
    ############################################################################
    camera_names = ['front', 'overhead', 'over_shoulder_left', 'over_shoulder_right']
    obs_config = observation_utils.create_obs_config(
        camera_names=camera_names,
        camera_resolution=[256, 256],
        method_name='PERACT_BC',
        robot_name='bimanual'
    )

    action_mode = BimanualMoveArmThenGripper(
        arm_action_mode=BimanualEndEffectorPoseViaPlanning(),
        gripper_action_mode=BimanualDiscrete(),
    )

    env = Environment(
        action_mode,
        './',
        obs_config=obs_config,
        headless=headless,
        robot_setup='dual_panda',
    )
    env.launch()

    task = env.get_task(BimanualLiftTray)
    task.reset()
    ############################################################################
    ## Collect demos for the demos(InContext Learning)
    full_sample = {
        'demos': [dict()] * num_demos,
        'live': dict(),
    }

    for i in tqdm(range(num_demos), desc=f'Collecting demos', total=num_demos, leave=False):
        done = False
        while not done:
            try:
                # task.set_variation(i % 2 * 2)
                demos = task.get_demos(1, live_demos=True, max_attempts=1000)  # -> List[List[Observation]]
                sample = rl_bench_demo_to_sample(demos[0], camera_names=camera_names)
                full_sample['demos'][i] = sample_to_cond_demo(sample, num_traj_wp)
                assert len(full_sample['demos'][i]['obs']) == num_traj_wp
                done = True
                print(f"Done with one demo {done}")
            except Exception as e:
                print(f"Error in collecting demo {i}: {e}")
                continue
    #############################################################################
    successes = []
    pbar = trange(num_rollouts, desc=f'Evaluating model, SR: 0/{num_rollouts}', leave=False)
    for i in pbar:
        done = False
        while not done:
            try:
                task.reset()
                done = True
            except:
                continue

        ## Action will be 16 for bimanual tasks. (7 pos, quat + 1 gripper action) * 2 arms = 16
        env_action = np.zeros(18)
        success = 0
        for k in range(max_execution_steps):
            curr_obs = task.get_observation()
            T_w_e_left = pose_to_transform(curr_obs.left.gripper_pose)
            T_w_e_right = pose_to_transform(curr_obs.right.gripper_pose)

            full_sample['live']['obs'] = [subsample_pcd(get_point_cloud(curr_obs, camera_names=camera_names))]
            full_sample['live']['grips_left'] = [curr_obs.left.gripper_open]
            full_sample['live']['grips_right'] = [curr_obs.right.gripper_open]
            full_sample['live']['actions_grip_left'] = [np.zeros(8)]
            full_sample['live']['actions_grip_right'] = [np.zeros(8)]
            full_sample['live']['T_w_es_left'] = [T_w_e_left]
            full_sample['live']['T_w_es_right'] = [T_w_e_right]
            full_sample['live']['actions_left'] = [full_sample['live']['T_w_es_left'][0].reshape(1, 4, 4).repeat(8, axis=0)]
            full_sample['live']['actions_right'] = [full_sample['live']['T_w_es_right'][0].reshape(1, 4, 4).repeat(8, axis=0)]
            data = save_sample(full_sample, None)

            if k == 0:
                demo_scene_node_embds, demo_scene_node_pos = model.model.get_demo_scene_emb(
                    data.to(model.config['device']))
            live_scene_node_embds, live_scene_node_pos = model.model.get_live_scene_emb(data.to(model.config['device']))
            data.live_scene_node_embds = live_scene_node_embds.clone()
            data.live_scene_node_pos = live_scene_node_pos.clone()
            data.demo_scene_node_embds = demo_scene_node_embds.clone()
            data.demo_scene_node_pos = demo_scene_node_pos.clone()

            with torch.no_grad():
                with torch.autocast(dtype=torch.float32, device_type=model.config['device']):
                    actions_left, actions_right, grips_left, grips_right = model.test_step(data.to(model.config['device']), 0)
                actions_left = actions_left.squeeze().cpu().numpy()
                actions_right = actions_right.squeeze().cpu().numpy()
                grips_left = grips_left.squeeze().cpu().numpy()
                grips_right = grips_right.squeeze().cpu().numpy()

            for j in range(execution_horizon):
                ## Right is first arm in the actions of rlbench
                env_action[:7] = transform_to_pose(T_w_e_right @ actions_right[j]) ## 7 for pos
                env_action[7] = int((grips_right[j] + 1) / 2 > 0.5) ## 1 for gripper action
                env_action[8] = 1   ## 1 for ignore collisions
                env_action[9:16] = transform_to_pose(T_w_e_left @ actions_left[j])
                env_action[16] = int((grips_left[j] + 1) / 2 > 0.5) ## 1 for gripper action
                env_action[17] = 1   ## 1 for ignore collisions

                try:
                    time.sleep(0.1)  # this is helping making the output smoother
                    curr_obs, reward, terminate = task.step(env_action)
                    success = int(terminate and reward > 0.)
                except Exception as e:
                    print(f"Error in executing action {j+1}: {e}")
                    terminate = True
                if terminate:
                    break
            else: ## Attached with the while loop above
                continue
            break
        successes.append(success)
        pbar.set_description(f'Evaluating model, SR: {sum(successes)}/{len(successes)}')
        pbar.refresh()
    pbar.close()
    env.shutdown()
    return sum(successes) / len(successes)


def generate_pseudo_data(
        num_demos=10,
        task_name='bimanual_lift_tray',
        max_execution_steps=50,
        execution_horizon=8,
        num_rollouts=2,
        headless=False,
        camera_names=['front', 'overhead', 'over_shoulder_left', 'over_shoulder_right'],
        num_traj_wp=10,
        restrict_rot=True
):

    obs_config = observation_utils.create_obs_config(
        camera_names=camera_names,
        camera_resolution=[256, 256],
        method_name='PERACT_BC',
        robot_name='bimanual'
    )

    action_mode = BimanualMoveArmThenGripper(
        arm_action_mode=BimanualEndEffectorPoseViaPlanning(),
        gripper_action_mode=BimanualDiscrete(),
    )

    env = Environment(
        action_mode,
        './',
        obs_config=obs_config,
        headless=headless,
        robot_setup='dual_panda',
    )
    env.launch()
    task = env.get_task(BimanualLiftTray)
    task.reset()

    for i in tqdm(range(num_demos), desc=f'Collecting demos', total=num_demos, leave=False):
        demos = task.get_demos(1, live_demos=True, max_attempts=10)  # -> List[List[Observation]]
        sample = rl_bench_demo_to_sample(demos[0], camera_names=camera_names)
        np.savez(f"src/data/rl_bench_data/traj_{i}.npz", **sample)
    env.shutdown()
            

def get_required_handles(names: list[str]) -> list[int]:
    keys = [n.lower() for n in names]
    all_shapes = Shape._get_objects_in_tree()

    filtered_shapes = [
        s for s in all_shapes
        if any(key in s.get_name().lower() for key in keys)
    ]
    return [shape.get_handle() for shape in filtered_shapes]


def rl_bench_demo_to_sample(demo, camera_names):
    sample = {
        'pcds': [],
        'T_w_es_left': [],
        'T_w_es_right': [],
        'grips_left': [],
        'grips_right': [],
    }

    for k, obs in enumerate(demo):
        sample['pcds'].append(get_point_cloud(obs, camera_names))
        sample['T_w_es_left'].append(pose_to_transform(obs.left.gripper_pose))
        sample['T_w_es_right'].append(pose_to_transform(obs.right.gripper_pose))
        sample['grips_left'].append(obs.left.gripper_open)
        sample['grips_right'].append(obs.right.gripper_open)
    sample['pcds'] = np.array(sample['pcds'], dtype=object)
    return sample


def get_point_cloud(obs, camera_names=('front', 'left_shoulder', 'right_shoulder')):
    req_handle = get_required_handles(
        names=[
            #'workspace', ## This was addes as it was hiding the table object from the point cloud.
            'tray',
            #'diningTable'
        ]
    )
    pcds = []
    for cam in camera_names:
        ordered_pcd = obs.perception_data[f'{cam}_point_cloud']
        seg_mask = obs.perception_data[f'{cam}_mask']
        req_mask  = np.isin(seg_mask, req_handle)
        masked_pcd = ordered_pcd[req_mask]
        pcds.append(masked_pcd)
    return downsample_pcd(np.concatenate(pcds, axis=0))
        

def main():
    generate_pseudo_data(
        num_demos=50000,
        task_name='bimanual_lift_tray',
        max_execution_steps=100,
        execution_horizon=8,
        num_rollouts=2,
        headless=True,
        num_traj_wp=10,
        restrict_rot=True
    )
    # rollout_model(
    #     model=None,
    #     num_demos=2,
    #     task_name='bimanual_lift_tray',
    #     max_execution_steps=30,
    #     execution_horizon=8,
    #     num_rollouts=5,
    #     headless=False,
    #     num_traj_wp=10,
    #     restrict_rot=True
    # )

if __name__ == "__main__":
    main()