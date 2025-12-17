# Dual-Arm Instant Policy with Task-Frame Coordination

A bimanual manipulation policy with task-frame coordination for cooperative manipulation tasks.

## Overview

This implementation extends single-arm Instant Policy to bimanual settings with explicit task-frame nodes that enable coordination between left and right grippers.

## What's Implemented

### Core Components

1. **SE(3) Utilities** (`src/utils/se3_ops.py`)
   - `task_pose_from_grippers()` - Computes shared task frame from gripper pair
   - `apply_se3()` - SE(3) transformations

2. **Graph with Task Nodes** (`src/models/graph_rep.py`)
   - 4 node types: scene, gripper_left, gripper_right, **task**
   - 7 new task edges for coordination
   - Task pose computation from gripper pairs
   - Task keypoints (6 points representing oriented frame)

3. **Model** (`src/models/model.py`)
   - 3 graph transformers process task nodes
   - Local encoder: spatial reasoning + task aggregation
   - Conditioning encoder: demo conditioning + task
   - Action encoder: temporal reasoning + task coordination

4. **Pseudo Demonstrations** (`src/pseudo_demo/`)
   - Already object-centric (no changes needed!)
   - Generates cooperative manipulation demos
   - Compatible with task-frame learning

## Quick Start

### Training

```python
from models.model import BimanualAGI
from config.train_config import config

# Create model (task nodes computed automatically)
model = BimanualAGI(config)

# Forward pass
predictions = model(data)
```

### Data Generation

```python
from pseudo_demo.tasks import cooperative_rigid, Workspace, default_arm_starts
import numpy as np

rng = np.random.RandomState(42)
ws = Workspace()
starts = default_arm_starts()
demo = cooperative_rigid(rng, ws, starts)
```

## How It Works

### Task-Frame Computation

```
T_task = task_pose_from_grippers(T_left, T_right)
```

- **Origin**: Midpoint between grippers
- **X-axis**: Left → Right (coordination axis)
- **Z-axis**: World up (orthogonalized)

### Graph Structure

```
Nodes: scene, gripper_left, gripper_right, task
Edges: task ←→ grippers, temporal, conditioning
```

### Information Flow

```
Left Gripper → Task Frame → Right Gripper
      ↓            ↓              ↓
         Task-informed predictions
```

## Configuration

`src/config/train_config.py`:

```python
'num_task_kp': 6,
'num_gripper_kp': 6,
'parameterization': 'task_residual',
```

## Testing

```bash
python test_task_frame.py
python test_end_to_end.py
```

## Documentation

See `DOCUMENTATION.md` for complete technical details, implementation guide, and verification.

## Key Features

✅ Explicit coordination via task nodes
✅ Object-centric demonstrations
✅ Minimal changes (~365 lines)
✅ Backward compatible
✅ Clean integration

## Status

✅ **COMPLETE and READY TO USE!**

All code implemented, tested, and documented.
