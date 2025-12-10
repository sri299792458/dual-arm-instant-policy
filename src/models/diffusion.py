import lightning as L
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.optimization import get_scheduler
import torch
from ip.utils.normalizer import Normalizer
import numpy as np

from models.model import BimanualAGI

from ip.utils.common_utils import transforms_to_actions , actions_to_transforms, rotation_matrix_to_angle_axis, get_rigid_transforms
from ip.utils.repairs import repair_checkpoint
import warnings

class GraphDiffusionBimanual(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = BimanualAGI(config)

        ########################################################################
        self.graph_rep = self.model.graph
        self.scene_encoder = self.model.scene_encoder
        self.local_encoder = self.model.local_encoder
        self.cond_encoder = self.model.cond_encoder
        self.action_encoder = self.model.action_encoder

        self.action_head_trans_left = self.model.prediction_head_left
        self.action_head_trans_right = self.model.prediction_head_right
        self.action_head_rot_left = self.model.prediction_head_rot_left
        self.action_head_rot_right = self.model.prediction_head_rot_right
        self.action_head_grip_left = self.model.prediction_head_g_left
        self.action_head_grip_right = self.model.prediction_head_g_right
        ########################################################################

        self.config = config
        self.record = config['record']
        self.save_dir = config['save_dir']
        self.save_every = config['save_every']
        self.randomise_num_demos = config['randomise_num_demos']
        self.use_lr_scheduler = config['use_lr_scheduler']
        self.best_trans_loss = 1e6
        self.best_sr = 0
        self.val_losses = []

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=config['num_diffusion_iters_train'],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=False,
            prediction_type='sample',
        )
        self.loss_f = torch.nn.L1Loss()
        self.normalizer = Normalizer(
            pred_horizon=config['pred_horizon'],
            min_action=config['min_actions'].to(config['device']),
            max_action=config['max_actions'].to(config['device']),
            device=config['device'],
        )

    def add_noise(self, actions, grip_actions, timesteps):
        '''
        actions: (B, T, 4, 4)
        grip_actions: (B, T, 1)
        timesteps: (B,)
        '''
        # First convert 4x4 to 6 (translation + angle axis)
        b, t = actions.shape[:2]

        actions_6d = transforms_to_actions(actions.view(-1, 4, 4)).view(b, t, 6)
        # Normalize the actions
        actions_6d = self.normalizer.normalize_actions(actions_6d)
        # Add noise
        noise = torch.randn(actions_6d.shape, device=self.device, dtype=actions_6d.dtype)
        noisy_actions = self.noise_scheduler.add_noise(actions_6d, noise, timesteps)
        noisy_actions = torch.clamp(noisy_actions, -1, 1)
        # Denormalize the actions
        noisy_actions = self.normalizer.denormalize_actions(noisy_actions)
        # Convert back to 4x4
        noisy_actions = actions_to_transforms(noisy_actions.view(-1, 6)).view(b, t, 4, 4)

        # Add noise to the gripper actions
        noise_g = torch.randn(grip_actions.shape, device=self.device, dtype=grip_actions.dtype)
        noisy_grip_actions = self.noise_scheduler.add_noise(grip_actions, noise_g, timesteps)
        noisy_grip_actions = torch.clamp(noisy_grip_actions, -1, 1)

        return noisy_actions, noisy_grip_actions
    

    def se3_loss(self, pred, gt):
        '''
        pred: (B, T, 4, 4)
        gt: (B, T, 4, 4)
        '''
        # Get the translation and rotation components
        trans_err = torch.norm(pred[..., :3, 3] - gt[..., :3, 3], dim=-1).mean()
        rot_error = torch.eye(4, device=pred.device, dtype=pred.dtype).repeat(pred.shape[0], pred.shape[1], 1, 1)
        rot_error[..., :3, :3] = pred[..., :3, :3].transpose(-1, -2) @ gt[..., :3, :3]
        rot_error = rot_error.view(-1, 4, 4)
        angle_axis = rotation_matrix_to_angle_axis(rot_error[:, :3, :])
        rot_error = angle_axis.norm(dim=-1).mean() * 180 / np.pi
        return trans_err, rot_error
    
    def training_step(self, data, batch_idx):
        batch_size = data.actions_left.shape[0]
        if self.randomise_num_demos: ## This will be False
            num_demos = np.random.randint(1, self.config['num_demos'] + 1)
            self.model.reinit_graphs(batch_size, num_demos=num_demos)

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()

        noisy_actions_left, noisy_grip_actions_left = self.add_noise(
            data.actions_left, data.actions_grip_left, timesteps
        )
        noisy_actions_right, noisy_grip_actions_right = self.add_noise(
            data.actions_right, data.actions_grip_right, timesteps
        )

        labels = self.model.get_labels(
            data.actions_left,
            data.actions_right,
            noisy_actions_left,
            noisy_actions_right,
            data.actions_grip_left.unsqueeze(-1),
            data.actions_grip_right.unsqueeze(-1),
            noisy_grip_actions_left.unsqueeze(-1),
            noisy_grip_actions_right.unsqueeze(-1),
            delta_grip=False,
        )
        ## labels are B, pred_horizon, 3*4 + 2
        ## We need to normalize the first 12 values as there are action position and rotations
        ## Sending them seperately so that the same function can be used for both left and right grippers.
        labels[..., :6] = self.normalizer.normalize_labels(labels[..., :6])
        labels[..., 6:12] = self.normalizer.normalize_labels(labels[..., 6:12])

        ## Store noisy actions and grips in the data object as gt actions
        data.actions_left = noisy_actions_left
        data.actions_right = noisy_actions_right
        data.actions_grip_left = noisy_grip_actions_left
        data.actions_grip_right = noisy_grip_actions_right
        data.diff_time = timesteps.view(-1, 1)

        preds = self.model(data)
        loss = self.loss_f(preds, labels)

        self.log("Train Loss", loss.mean(), on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, data, batch_idx, vis=False, ret_actions=False):
        batch_size = data.actions_left.shape[0]
        self.model.reinit_graphs(batch_size, num_demos=self.config['num_demos_test'])
        gt_actions_left = data.actions_left.clone()
        gt_actions_right = data.actions_right.clone()
        gt_grip_left = data.actions_grip_left.clone()
        gt_grip_right = data.actions_grip_right.clone()

        with torch.autocast(dtype=torch.float32, device_type='cuda'): # SDV need f32
            actions_left, actions_right, grips_left, grips_right = self.test_step(data, batch_idx, vis=vis)

        grip_loss_left = (grips_left.squeeze() - gt_grip_left.squeeze()).abs().mean()
        grip_loss_right = (grips_right.squeeze() - gt_grip_right.squeeze()).abs().mean()
        trans_err_left, rot_err_left = self.se3_loss(actions_left, gt_actions_left)
        trans_err_right, rot_err_right = self.se3_loss(actions_right, gt_actions_right)

        self.log("Val_Grip_Loss_Left", grip_loss_left, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("Val_Grip_Loss_Right", grip_loss_right, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("Val_Trans_Err_Left", trans_err_left, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("Val_Rot_err_Left", rot_err_left, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("Val_Trans_Err_Right", trans_err_right, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("Val_Rot_Err_Right", rot_err_right, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        self.model.reinit_graphs(self.config['batch_size'], num_demos=self.config['num_demos'])
        self.val_losses.append(trans_err_left)
        self.val_losses.append(trans_err_right)
        if ret_actions:
            return actions_left, actions_right, grips_left, grips_right
        loss = 0
        return loss


    def test_step(self, data, batch_idx, vis=False):
        batch_size = data.actions_left.shape[0]
        noisy_actions_left = torch.randn(
            (batch_size, self.config['pred_horizon'], 6), device=self.device
        )
        noisy_actions_right = torch.randn(
            (batch_size, self.config['pred_horizon'], 6), device=self.device
        )
        noisy_actions_left = torch.clamp(noisy_actions_left, -1, 1)
        noisy_actions_left = self.normalizer.denormalize_actions(noisy_actions_left)
        noisy_actions_left = actions_to_transforms(noisy_actions_left.view(-1, 6)).view(batch_size, -1, 4, 4)
        noisy_actions_right = torch.clamp(noisy_actions_right, -1, 1)
        noisy_actions_right = self.normalizer.denormalize_actions(noisy_actions_right)
        noisy_actions_right = actions_to_transforms(noisy_actions_right.view(-1, 6)).view(batch_size, -1, 4, 4)

        noisy_grip_left = torch.randn((batch_size, self.config['pred_horizon'], 1), device=self.device)
        noisy_grip_left = torch.clamp(noisy_grip_left, -1, 1)
        noisy_grip_right = torch.randn((batch_size, self.config['pred_horizon'], 1), device=self.device)
        noisy_grip_right = torch.clamp(noisy_grip_right, -1, 1)

        ## init scheduler
        self.noise_scheduler.set_timesteps(self.config['num_diffusion_iters_test'])

        for k in range(self.config['num_diffusion_iters_test'] - 1, -1, -1):
            data.actions_left = noisy_actions_left
            data.actions_right = noisy_actions_right
            data.diff_time = torch.tensor([[
                k if  k != self.config['num_diffusion_iters_test'] - 1 else self.config['num_diffusion_iters_train']
            ]] * batch_size, device=self.device)

            preds = self.model(data)
            preds[..., :6] = self.normalizer.denormalize_labels(preds[..., :6])
            preds[..., 6:12] = self.normalizer.denormalize_labels(preds[..., 6:12])

            current_gripper_pos_left = self.model.get_transformed_node_pos(noisy_actions_left, transform=False)
            current_gripper_pos_right = self.model.get_transformed_node_pos(noisy_actions_right, transform=False)

            mode_output_left = preds[..., 3:6] + current_gripper_pos_left + torch.mean(preds[..., :3], dim=-2, keepdim=True)
            mode_output_right = preds[..., 9:12] + current_gripper_pos_right + torch.mean(preds[..., 6:9], dim=-2, keepdim=True)

            ## Diffusion step
            pred_gripper_left_pos = self.noise_scheduler.step(
                model_output=mode_output_left,
                sample=current_gripper_pos_left,
                timestep=k,
            ).prev_sample
            pred_gripper_right_pos = self.noise_scheduler.step(
                model_output=mode_output_right,
                sample=current_gripper_pos_right,
                timestep=k,
            ).prev_sample

            ## Get Transformed matrix for the gripper.
            T_e_e_left = get_rigid_transforms(
                current_gripper_pos_left.view(-1, pred_gripper_left_pos.shape[-2], 3),
                pred_gripper_left_pos.view(-1, pred_gripper_left_pos.shape[-2], 3)
            ).view(batch_size, -1, 4, 4)

            T_e_e_right = get_rigid_transforms(
                current_gripper_pos_right.view(-1, pred_gripper_right_pos.shape[-2], 3),
                pred_gripper_right_pos.view(-1, pred_gripper_right_pos.shape[-2], 3)
            ).view(batch_size, -1, 4, 4)


            noisy_actions_left = torch.matmul(noisy_actions_left, T_e_e_left)
            noisy_actions_right = torch.matmul(noisy_actions_right, T_e_e_right)

            noisy_grip_left = self.noise_scheduler.step(
                model_output=preds[..., -2:-1].mean(dim=-2),
                sample=noisy_grip_left,
                timestep=k,
            ).prev_sample
            noisy_grip_right = self.noise_scheduler.step(
                model_output=preds[..., -1:].mean(dim=-2),
                sample=noisy_grip_right,
                timestep=k,
            ).prev_sample
            noisy_grip_left = torch.clamp(noisy_grip_left, -1, 1)
            noisy_grip_right = torch.clamp(noisy_grip_right, -1, 1)

            ## Convert to 6d, normalize, clamp and denormalize
            noisy_actions_6d_left = transforms_to_actions(noisy_actions_left.view(-1, 4, 4)).view(batch_size, -1, 6)
            noisy_actions_6d_left = self.normalizer.normalize_actions(noisy_actions_6d_left)
            noisy_actions_6d_left = torch.clamp(noisy_actions_6d_left, -1, 1)
            noisy_actions_6d_left = self.normalizer.denormalize_actions(noisy_actions_6d_left)
            noisy_actions_6d_right = transforms_to_actions(noisy_actions_right.view(-1, 4, 4)).view(batch_size, -1, 6)
            noisy_actions_6d_right = self.normalizer.normalize_actions(noisy_actions_6d_right)
            noisy_actions_6d_right = torch.clamp(noisy_actions_6d_right, -1, 1)
            noisy_actions_6d_right = self.normalizer.denormalize_actions(noisy_actions_6d_right)

            noisy_actions_left = actions_to_transforms(noisy_actions_6d_left.view(-1, 6)).view(batch_size, -1, 4, 4)
            noisy_actions_right = actions_to_transforms(noisy_actions_6d_right.view(-1, 6)).view(batch_size, -1, 4, 4)

        return noisy_actions_left, noisy_actions_right, torch.sign(noisy_grip_left), torch.sign(noisy_grip_right)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'],
                                      weight_decay=self.config['weight_decay'])
        if self.use_lr_scheduler:
            lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=self.config['num_warmup_steps'],
                num_training_steps=self.config['num_iters'],
            )
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]
        return optimizer

    def on_validation_epoch_end(self, *args, **kwargs):
        mean_trans_err = torch.tensor(self.val_losses).mean()
        self.val_losses = []
        if self.best_trans_loss > mean_trans_err and self.record:
            # TODO: Could be smarter.
            self.save_model(f'{self.save_dir}/best.pt')
            self.best_trans_loss = mean_trans_err

    def save_model(self, path, save_compiled=False):
        self.trainer.save_checkpoint(path)
        if self.config['compile_models']:
            repair_checkpoint(path, save_path=path)
            if save_compiled:
                path_compiled = path.replace('.pt', '_compiled.pt')
                self.trainer.save_checkpoint(path_compiled)


    def on_train_batch_end(self, *args, **kwargs):
        if self.global_step % self.save_every == 0 and self.record:
            self.save_model(f'{self.save_dir}/{self.global_step}.pt', save_compiled=False)

        # TODO: Can run evals and log results here.


    def on_train_epoch_end(self, *args, **kwargs):
        if self.record:
            self.save_model(f'{self.save_dir}/last.pt', save_compiled=True)
