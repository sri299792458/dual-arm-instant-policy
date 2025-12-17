#!/usr/bin/env python3
"""Test that task-frame integration works in graph_rep.py"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from config.train_config import config

print("="*60)
print("Testing Task-Frame Integration")
print("="*60)

# Test 1: Import SE3 utilities
print("\n[1/4] Testing SE3 utilities...")
try:
    from utils.se3_ops import task_pose_from_grippers, apply_se3

    # Create simple test poses
    T_left = torch.eye(4)[None, ...]
    T_left[0, :3, 3] = torch.tensor([0.5, 0.1, 1.0])
    T_right = torch.eye(4)[None, ...]
    T_right[0, :3, 3] = torch.tensor([0.5, -0.1, 1.0])

    T_task = task_pose_from_grippers(T_left, T_right)
    print(f"✓ task_pose_from_grippers works!")
    print(f"  Task frame origin: {T_task[0, :3, 3].tolist()}")
    print(f"  Expected: [0.5, 0.0, 1.0]")

    # Test apply_se3
    points = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    transformed = apply_se3(T_task, points[None, :, :])
    print(f"✓ apply_se3 works!")
    print(f"  Transformed shape: {transformed.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Import graph builder with task-frame
print("\n[2/4] Testing graph builder...")
try:
    from models.graph_rep import BimanualGraphRep

    graph_rep = BimanualGraphRep(config)

    print(f"✓ BimanualGraphRep instantiated!")
    print(f"  Node types: {graph_rep.node_types}")
    print(f"  Has 'task' node type: {'task' in graph_rep.node_types}")
    print(f"  Task keypoints: {graph_rep.num_task_kp}")
    print(f"  Edge types count: {len(graph_rep.edge_types)}")

    # Check for task edges
    task_edges = [e for e in graph_rep.edge_types if 'task' in str(e)]
    print(f"  Task-related edges ({len(task_edges)}):")
    for edge in task_edges:
        print(f"    - {edge}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Initialize graph
print("\n[3/4] Testing graph initialization...")
try:
    graph_rep.initialise_graph()
    print(f"✓ Graph initialized!")
    print(f"  Graph type: {type(graph_rep.graph)}")

    # Check task node info
    if hasattr(graph_rep.graph, 'task_batch'):
        print(f"  Task batch info shape: {graph_rep.graph.task_batch.shape}")
        print(f"  Task time info shape: {graph_rep.graph.task_time.shape}")
        print(f"  Task embd info shape: {graph_rep.graph.task_embd.shape}")
    else:
        print("  ⚠ Task node info not found in graph")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Create mock data and update graph
print("\n[4/4] Testing graph update with mock data...")
try:
    from types import SimpleNamespace

    B = config['batch_size']
    D = config['num_demos']
    T = config['traj_horizon']
    P = config['pred_horizon']
    S = config['num_scenes_nodes']

    # Create mock data
    data = SimpleNamespace()

    # Demo poses
    data.demo_T_w_es_left = torch.eye(4)[None, None, None, :, :].repeat(B, D, T, 1, 1)
    data.demo_T_w_es_left[..., 0, 3] = 0.5  # x
    data.demo_T_w_es_left[..., 1, 3] = 0.1  # y
    data.demo_T_w_es_left[..., 2, 3] = 1.0  # z

    data.demo_T_w_es_right = torch.eye(4)[None, None, None, :, :].repeat(B, D, T, 1, 1)
    data.demo_T_w_es_right[..., 0, 3] = 0.5
    data.demo_T_w_es_right[..., 1, 3] = -0.1
    data.demo_T_w_es_right[..., 2, 3] = 1.0

    # Current poses
    data.T_w_es_left = torch.eye(4)[None, :, :].repeat(B, 1, 1)
    data.T_w_es_left[:, 0, 3] = 0.5
    data.T_w_es_left[:, 1, 3] = 0.1
    data.T_w_es_left[:, 2, 3] = 1.0

    data.T_w_es_right = torch.eye(4)[None, :, :].repeat(B, 1, 1)
    data.T_w_es_right[:, 0, 3] = 0.5
    data.T_w_es_right[:, 1, 3] = -0.1
    data.T_w_es_right[:, 2, 3] = 1.0

    # Action poses
    data.actions_left = torch.eye(4)[None, None, :, :].repeat(B, P, 1, 1)
    data.actions_right = torch.eye(4)[None, None, :, :].repeat(B, P, 1, 1)

    # Gripper states
    data.grasp_left_demo = torch.ones(B, D, T, 1)
    data.grasp_right_demo = torch.ones(B, D, T, 1)
    data.current_grip_left = torch.ones(B, 1)
    data.current_grip_right = torch.ones(B, 1)
    data.actions_grip_left = torch.ones(B, P, 1)
    data.actions_grip_right = torch.ones(B, P, 1)

    # Scene nodes
    data.demo_scene_node_pos = torch.randn(B, D, T, S, 3)
    data.live_scene_node_pos = torch.randn(B, S, 3)
    data.action_scene_node_pos = torch.randn(B, P, S, 3)
    data.demo_scene_node_embds = torch.randn(B, D, T, S, config['local_nn_dim'])
    data.live_scene_node_embds = torch.randn(B, S, config['local_nn_dim'])
    data.action_scene_node_embds = torch.randn(B, P, S, config['local_nn_dim'])

    # Diffusion time
    data.diff_time = torch.zeros(B)

    # Update graph
    graph_rep.update_graph(data)

    print(f"✓ Graph updated successfully!")
    if 'task' in graph_rep.graph.node_types:
        print(f"  Task node positions shape: {graph_rep.graph['task'].pos.shape}")
        print(f"  Task node features shape: {graph_rep.graph['task'].x.shape}")

        # Check a few task node positions
        task_pos = graph_rep.graph['task'].pos[:10]
        print(f"  First few task positions:")
        print(f"    {task_pos}")
    else:
        print("  ⚠ Task nodes not found in updated graph")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Task-Frame Integration Test Complete!")
print("="*60)
