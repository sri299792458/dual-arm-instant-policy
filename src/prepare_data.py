'''
This script shows how to convert demonstrations performing the same task (or a pseudo-task) to a format suitable to
train our In-Context Imitation Learning model.
'''
import os
import torch
from tqdm import tqdm
import numpy as np
from ip.models.scene_encoder import SceneEncoder
from utils.data_proc import sample_to_cond_demo, sample_to_live, save_sample


if __name__ == '__main__':
    ####################################################################################################################
    num_demos = 2  # Demos in the context.
    num_waypoints_demo = 10  # Number of waypoints in one demo.
    pred_horizon = 8  # How many future actions we are predicting.
    # Maximum spacing between subsequent actions (1cm, 3degrees).
    live_spacing_trans = 0.01
    live_spacing_rot = 3
    # We can pre-compute geometry embeddings while processing data, reducing computational requirements during training.
    compute_embeddings = True
    save_dir = 'src/data/bimanual_pseudo_data_200'  # Path to where the data should be saved.
    os.makedirs(save_dir, exist_ok=False)
    ####################################################################################################################
    # Load scene encoder if we are pre-computing geometry embeddings.
    if compute_embeddings:
        scene_encoder = SceneEncoder(num_freqs=10, embd_dim=512)
        scene_encoder.load_state_dict(
            torch.load(f'src/checkpoints/scene_encoder.pt'))
        scene_encoder = scene_encoder.to('cuda')
        scene_encoder.eval()
    else:
        scene_encoder = None
    ####################################################################################################################
    # Get everything into the right format.
    full_sample = {
        'demos': [dict()] * num_demos,
        'live': dict(),
    }

    # TODO: Collect or load demonstrations in a form of {'pcds': [], 'T_w_es': [], 'grips': []}
    # TODO: You can also shuffle them here or create permutations to create more data samples.
    num_files = len(os.listdir("src/data/pseudo_data_200"))

    ## Create 2 time more data by shuffling the existing data.
    for j in tqdm(range(2*num_files), desc="Shuffles"):
        demos = []
        idxs = np.random.choice(range(num_files), num_demos + 1, replace=False)
        
        for i in idxs:
            curr_demo = np.load(f"src/data/pseudo_data_nonlinear/traj_{i}.npz", allow_pickle=True)
            ## Convert the npz obj to a dict.
            curr_demo = {k: curr_demo[k] for k in curr_demo.files}
            demos.append(curr_demo)

        
        for i, sample in enumerate(demos):
            if i < num_demos:
                full_sample['demos'][i] = sample_to_cond_demo(sample, num_waypoints_demo)
            else:
                full_sample['live'] = sample_to_live(sample, pred_horizon, 2048,
                                                    live_spacing_trans, live_spacing_rot, subsample=True)
                break
        ####################################################################################################################
        # Save the data. Data will be saved as data_{i + offset}.pt in save_dir for i [0, len(full_sample['live'])].
        # In our experiments, we continuously generate data and save it by adjusting the offset, overwriting the old samples.
        offset = len(os.listdir(save_dir))
        save_sample(full_sample, save_dir=save_dir, offset=offset, scene_encoder=scene_encoder)
