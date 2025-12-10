from ip.models.diffusion import *
from ip.configs.base_config import config

model_path = "/home/mohit/dual-arm-instant-policy/src/data/model.pt"
# config_path = "models/ldm/text2im-large/config.yaml"

config['pre_trained_encoder'] = False  # Set to False to not use pre-trained encoder

model = GraphDiffusion.load_from_checkpoint(
    model_path,
    config=config,
    strict=True,
    map_location="cpu",
).to('cuda:0')

scene_encoder = model.model.scene_encoder

torch.save(scene_encoder.state_dict(), "scene_encoder.pt")

for name, param in scene_encoder.named_parameters():
    print(name, param.shape)