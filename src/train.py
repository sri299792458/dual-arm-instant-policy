import os
import pickle
from torch_geometric.loader import DataLoader
from utils.running_dataset import RunningDatasetBimanual
from lightning.pytorch.callbacks import LearningRateMonitor
from config.train_config import config
from lightning.pytorch.loggers import WandbLogger
from models.diffusion import GraphDiffusionBimanual
import lightning as L

import torch
torch.set_float32_matmul_precision('high')

def main():
    
    record = config['record'] ## Whether to log the training and save models
    use_wandb = True ## Log training on weights and biases
    save_path = 'src/checkpoints'
    fine_tune = False ## Whether to train from scratch or fine-tune existing model
    model_name = 'dual_arm_ip.pt'
    compile_models = False ## Whether to compile models
    data_path_train = 'src/data/bimanual_pseudo_data_100'
    data_path_val = 'src/data/bimanual_val'
    batch_size = config['batch_size']

    save_dir = f'{save_path}/psuedo_data_100_steps_150' if record else None
    os.makedirs(save_dir, exist_ok=True)

    config['save_dir'] = save_dir
    config['record'] = record
    model = GraphDiffusionBimanual(config).to(config['device'])

    dset_val = RunningDatasetBimanual(data_path_val, len(os.listdir(data_path_val)), rand_g_prob=0)
    dataloader_val = DataLoader(dset_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    dset = RunningDatasetBimanual(data_path_train, len(os.listdir(data_path_train)), rand_g_prob=config['randomize_g_prob'])
    dataloader = DataLoader(dset, batch_size=batch_size, drop_last=True, shuffle=True,
                            num_workers=8, pin_memory=True)
    
    ## Wandb palceholder
    if record:
        if use_wandb:
            logger = WandbLogger(
                project='dual_arm_instant_policy',
                name=f"psuedo_data_100_steps_150",
                save_dir=save_dir,
                log_model=False
            )
    else:
        logger = None
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = L.Trainer(
        enable_checkpointing=False,
        accelerator=config['device'].split(':')[0],
        devices=[int(config['device'].split(':')[1])],
        max_steps=config['num_iters'],
        # max_epochs=config['num_epochs'],
        enable_progress_bar=True,
        precision='16-mixed',
        val_check_interval=10000,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=None,
        logger=logger,
        log_every_n_steps=500, ## Increase when you have more data
        gradient_clip_val=10,
        gradient_clip_algorithm='norm',
        callbacks=[lr_monitor],
    )

    trainer.fit(
        model=model,
        train_dataloaders=dataloader,
        val_dataloaders=dataloader_val,
    )

    ## Save last model
    if record:
        model.save_model(f"{save_dir}/last_model.pt")



if __name__=='__main__':
    main()