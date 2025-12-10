import numpy as np
import time
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.arm_action_modes import BimanualJointPosition, BimanualEndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete

from rlbench.observation_config import ObservationConfig
from rlbench.environment import Environment
from rlbench.bimanual_tasks.bimanual_lift_tray import BimanualLiftTray


action_mode = BimanualMoveArmThenGripper(
    arm_action_mode=BimanualEndEffectorPoseViaPlanning(),
    gripper_action_mode=BimanualDiscrete(),
)
obs = ObservationConfig()

env = Environment(
    action_mode=action_mode,
    obs_config=obs,
    robot_setup='dual_panda',
)

env.launch()

task = env.get_task(BimanualLiftTray)
description, obs = task.reset()

## Print the action Shape
print(f"Action shape: {env.action_shape}")

for _ in range(1000):
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # Example action
    print(f"Action: {action}")
    obs, reward, done = task.step(action)
    print(f"Reward: {reward}, Done: {done}")
    time.sleep(0.1)

input("Press Enter to close the environment...")
env.shutdown()