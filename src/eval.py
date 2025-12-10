from models.diffusion import GraphDiffusionBimanual
from rlbench_data_generator import rollout_model

from config.train_config import config

if __name__ == "__main__":
    
    model_path = '/home/mohit/dual-arm-instant-policy/src/checkpoints/my_run/best.pt'

    model = GraphDiffusionBimanual.load_from_checkpoint(
        model_path, config=config, strict=True, 
        map_location=config['device']).to(config['device'])
    model.model.reinit_graphs(1, num_demos=config['num_demos'])
    model.eval()

    sr = rollout_model(
        model, num_demos=config['num_demos']
    )
    print(f"Success rate: {sr}")