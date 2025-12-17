from torch.utils.data import Dataset
import torch
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm

class RunningDatasetBimanual(Dataset):
    def __init__(self, data_path, num_samples, rec=False, rand_g_prob=0.0, random_rotation=False, preload=True):
        self.data_path = data_path
        self.num_samples = num_samples
        self.rand_g_prob = rand_g_prob
        self.random_rotation = random_rotation
        self.rec = rec
        ## TODO: Think about the below
        if rec:
            self.data_attr = [
                'pos',
                'queries',
                'batch_queries',
                'batch_pos',
                'occupancy',
            ]
        else:
            self.data_attr = [
                # 'pos_demos',
                # 'graps_demos',
                # 'batch_demos',
                # 'pos_obs',
                # 'current_grip',
                # 'batch_pos_obs',
                # 'past_actions',
                # 'past_actions_grip',
                'actions_left',
                'actions_right',
                'actions_grip_left',
                'actions_grip_right',
            ]
        
        self.preload = preload
        if self.preload:
            print(f"Preloading {num_samples} samples into memory...")
            self.data_cache = {}
            for i in tqdm(range(num_samples), desc="Preloading data"):
                try:
                    self.data_cache[i] = torch.load(os.path.join(data_path, f"data_{i}.pt"), map_location='cpu')
                except FileNotFoundError:
                    # In case data generation is incomplete or sparse
                    pass
            print(f"Preloaded {len(self.data_cache)} samples.")

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        ## While loop seemed weird here, but it is used to handle the case where the data 
        ## file might hot have correct attributes or might be corrupted.
        ## In that case, it will randomly select another index until it finds a valid data file
        ## with the required attributes.
        ## I don't see how this is useful in practice, but it is there in the original code.
        ## I don't see how this is useful in practice, but it is there in the original code.
        while True:
            try:
                if self.preload and idx in self.data_cache:
                    # Need to clone or detach? Torch load usually returns independent objects but
                    # modifying them in place might affect cache if we are not careful.
                    # The original code modifies data.current_grip_right etc.
                    # So we should probably deep copy or rely on the fact that modifications are minor?
                    # Safer to return as is if we trust training step doesn't mutate in place destructively over epochs.
                    # Training step modifies: data.actions_left (with noise).
                    # YES, training step modifies data object in place!
                    # See diffusion.py: data.actions_left = noisy_actions_left
                    # So we MUST restart from a fresh copy or ensure we don't mutate the cached object.
                    # Fast deep copy:
                    import copy
                    data = copy.deepcopy(self.data_cache[idx])
                else:
                    data = torch.load(os.path.join(self.data_path, f"data_{idx}.pt"), map_location='cpu')
                ## Make sure the data has all the required attributes
                ## I don't know what does this mean or how is this used.
                for attr in self.data_attr:
                    assert hasattr(data, attr)
                
                ## Randomly open/close the grippers. But not together probably.
                if np.random.uniform() < self.rand_g_prob:
                    data.current_grip_right *= -1
                if np.random.uniform() < self.rand_g_prob:
                    data.current_grip_left *= -1

                if self.random_rotation and self.rec:
                    raise NotImplementedError("Random rotation is not implemented for the running dataset.")
                    R = torch.tensor(Rot.random().as_matrix(), dtype=data.pose.dtype, device=data.pos.device)
                    data.pos = data.pos @ R.T
                    data.queries = data.queries @ R.T
                return data
            except Exception as e:
                idx = np.random.randint(0, self.num_samples)