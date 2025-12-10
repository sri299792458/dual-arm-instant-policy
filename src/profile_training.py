
import torch
import time
import os
import sys
# Add src to path just in case
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torch.profiler import profile, record_function, ProfilerActivity
from models.diffusion import GraphDiffusionBimanual
from utils.running_dataset import RunningDatasetBimanual
from torch_geometric.loader import DataLoader
from config.train_config import config

# Set device to cpu if cuda not available, or respect config
device = config['device'] if torch.cuda.is_available() else 'cpu'
config['device'] = device
print(f"Using device: {device}")

def profile_training_step(data_path, num_samples):
    # Setup
    print("Initializing model...")
    model = GraphDiffusionBimanual(config).to(device)
    model.train()
    
    print(f"Loading dataset from {data_path}...")
    dset = RunningDatasetBimanual(data_path, num_samples)
    loader = DataLoader(dset, batch_size=config['batch_size'], shuffle=True)
    
    print("Getting batch...")
    try:
        data = next(iter(loader)).to(device)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Warmup
    print("Warming up...")
    for _ in range(3):
        loss = model.training_step(data, 0)
        loss.backward()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    print("Profiling...")
    # Profile with PyTorch profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("training_step"):
            loss = model.training_step(data, 0)
        
        with record_function("backward"):
            loss.backward()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Print results
    print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=30))
    
    # Export for visualization
    prof.export_chrome_trace("training_trace.json")
    print("\nExported trace to training_trace.json")
    print("Open in chrome://tracing to visualize")

def manual_timing(data_path, num_samples):
    """Manual timing of each component"""
    print("Starting manual timing...")
    
    model = GraphDiffusionBimanual(config).to(device)
    model.train()
    
    dset = RunningDatasetBimanual(data_path, num_samples)
    loader = DataLoader(dset, batch_size=config['batch_size'], shuffle=True)
    
    try:
        data = next(iter(loader)).to(device)
    except StopIteration:
        print("Dataset empty!")
        return

    # Warmup
    for _ in range(3):
        _ = model.training_step(data, 0)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    timings = {}
    
    # Time each component
    def time_fn(name, fn, *args, **kwargs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.perf_counter() - start) * 1000
        timings[name] = elapsed
        return result
    
    # Data prep
    batch_size = data.actions_left.shape[0]
    timesteps = torch.randint(0, 100, (batch_size,), device=device)
    
    # Time add_noise
    time_fn("add_noise_left", model.add_noise, 
            data.actions_left, data.actions_grip_left, timesteps)
    
    # Time get_labels
    noisy_actions_left, noisy_grip_left = model.add_noise(
        data.actions_left, data.actions_grip_left, timesteps)
    noisy_actions_right, noisy_grip_right = model.add_noise(
        data.actions_right, data.actions_grip_right, timesteps)
    
    time_fn("get_labels", model.model.get_labels,
            data.actions_left, data.actions_right,
            noisy_actions_left, noisy_actions_right,
            data.actions_grip_left.unsqueeze(-1), data.actions_grip_right.unsqueeze(-1),
            noisy_grip_left.unsqueeze(-1), noisy_grip_right.unsqueeze(-1))
    
    # Time graph update
    data.actions_left = noisy_actions_left
    data.actions_right = noisy_actions_right
    data.actions_grip_left = noisy_grip_left
    data.actions_grip_right = noisy_grip_right
    data.diff_time = timesteps.view(-1, 1)
    
    # Get embeddings first
    if not hasattr(data, 'demo_scene_node_embds'):
        time_fn("get_demo_scene_emb", model.model.get_demo_scene_emb, data)
    if not hasattr(data, 'live_scene_node_embds'):
        time_fn("get_live_scene_emb", model.model.get_live_scene_emb, data)
    
    data.action_scene_node_embds = data.live_scene_node_embds[:, None, :, :].repeat(
        1, config['pred_horizon'], 1, 1)
    data.action_scene_node_pos = data.live_scene_node_pos[:, None, :, :].repeat(
        1, config['pred_horizon'], 1, 1)
    
    # Time graph construction
    time_fn("update_graph", model.model.graph.update_graph, data)
    
    # Time encoders
    graph = model.model.graph.graph
    time_fn("local_encoder", model.model.local_encoder,
            graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict)
    
    x_dict = model.model.local_encoder(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict)
    time_fn("cond_encoder", model.model.cond_encoder,
            x_dict, graph.edge_index_dict, graph.edge_attr_dict)
    
    x_dict = model.model.cond_encoder(x_dict, graph.edge_index_dict, graph.edge_attr_dict)
    time_fn("action_encoder", model.model.action_encoder,
            x_dict, graph.edge_index_dict, graph.edge_attr_dict)
    
    # Print results
    print("\n" + "="*60)
    print("TIMING BREAKDOWN (milliseconds)")
    print("="*60)
    total = sum(timings.values())
    for name, ms in sorted(timings.items(), key=lambda x: -x[1]):
        pct = ms / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{name:25s}: {ms:8.2f}ms ({pct:5.1f}%) {bar}")
    print("-"*60)
    print(f"{'TOTAL':25s}: {total:8.2f}ms")

if __name__ == "__main__":
    # Ensure data exists. If not, generate it.
    DATA_DIR = os.path.join("src", "data", "profiling_data")
    NUM_SAMPLES = 40
    
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) < NUM_SAMPLES:
        print("Data not found. Please generate data first using relevant scripts.")
        # For now we assume data is generated by external steps or we could call generation here.
    
    print("Running manual timing...")
    manual_timing(DATA_DIR, NUM_SAMPLES)
    
    print("\n\nRunning PyTorch profiler...")
    profile_training_step(DATA_DIR, NUM_SAMPLES)
