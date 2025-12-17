import torch
import torch_geometric

from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_batch

from ip.models.scene_encoder import SceneEncoder
from ip.models.graph_transformer import GraphTransformer

from models.graph_rep import BimanualGraphRep

from ip.utils.common_utils import dfs_freeze

"""
This Code is built on top of Instant-Policy (IP).
And is heavily inspired from the IP codebase.
"""

class BimanualAGI(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_demos = config['num_demos'] ## 2
        self.num_demos_in_use = config['num_demos'] ## 2
        self.traj_horizon = config['traj_horizon'] ## 10
        self.local_embd_dim = config['local_nn_dim'] ## 512
        self.batch_size = config['batch_size'] ## 16
        self.num_scenes_nodes = config['num_scenes_nodes'] ## 16 Point cloud ke kitne nodes banane hai.
        self.pred_horizon = config['pred_horizon'] ## 8
        self.num_layers = config['num_layers'] ## 2
        compile_models = config['compile_models'] ## False, model ko compile karne se inference fast hota hai.

        self.scene_encoder = SceneEncoder(
            num_freqs=10,
            embd_dim=config['local_nn_dim'],
        ).to(config['device'])

        if self.config['pre_trained_encoder']: ## This will be True, we would like to use a pre-trained scene encoder.
            self.scene_encoder.load_state_dict(torch.load(config['scene_encoder_path']))
            if self.config['freeze_encoder']:
                dfs_freeze(self.scene_encoder) ## Ye sach m depth first search krke saare parameters ko freeze kr dega.

        self.graph = BimanualGraphRep(config)
        self.graph.initialise_graph()

        in_channels = self.local_embd_dim
        if config['pos_in_nodes']:
            in_channels += self.graph.edge_dim // 2

        self.local_encoder = GraphTransformer(
            in_channels=in_channels,
            hidden_channels=config['hidden_dim'],
            heads=config['hidden_dim'] // 64,
            num_layers=self.num_layers,
            metadata=(
                ['scene', 'gripper_left', 'gripper_right', 'task'],
                [
                    ('scene', 'rel', 'scene'),
                    ('gripper_left', 'rel', 'gripper_left'),
                    ('gripper_right', 'rel', 'gripper_right'),
                    ('scene', 'rel', 'gripper_left'),
                    ('scene', 'rel', 'gripper_right'),
                    ('gripper_left', 'rel', 'gripper_right'),
                    ('gripper_right', 'rel', 'gripper_left'),
                    ('task', 'rel', 'task'),
                    ('gripper_left', 'to_task', 'task'),
                    ('gripper_right', 'to_task', 'task'),
                    ('task', 'from_task', 'gripper_left'),
                    ('task', 'from_task', 'gripper_right'),
                ]
            ),
            edge_dim=self.graph.edge_dim,
            dropout=0.0,
            norm='layer',
        ).to(config['device'])

        self.cond_encoder = GraphTransformer(
            in_channels=config['hidden_dim'],
            hidden_channels=config['hidden_dim'],
            heads=config['hidden_dim'] // 64,
            num_layers=self.num_layers,
            metadata=(
                ['scene', 'gripper_left', 'gripper_right', 'task'],
                [
                    ('gripper_left', 'cond', 'gripper_left'),
                    ('gripper_right', 'cond', 'gripper_right'),
                    ('gripper_left', 'demo', 'gripper_left'),
                    ('gripper_right', 'demo', 'gripper_right'),
                    ('scene', 'rel_demo', 'gripper_left'),
                    ('scene', 'rel_demo', 'gripper_right'),
                    ('scene', 'rel_demo', 'scene'),
                    ('task', 'cond', 'task'),
                ]
            ),
            edge_dim=self.graph.edge_dim,
            dropout=0.0,
            norm='layer',
        ).to(config['device'])

        self.action_encoder = GraphTransformer(
            in_channels=config['hidden_dim'],
            hidden_channels=config['hidden_dim'],
            heads=config['hidden_dim'] // 64,
            num_layers=self.num_layers,
            metadata=(
                ['scene', 'gripper_left', 'gripper_right', 'task'],
                [
                    ('gripper_left', 'time_action', 'gripper_left'),
                    ('gripper_right', 'time_action', 'gripper_right'),
                    ('gripper_left', 'rel_cond', 'gripper_left'),
                    ('gripper_right', 'rel_cond', 'gripper_right'),
                    ('scene', 'rel_action', 'gripper_left'),
                    ('scene', 'rel_action', 'gripper_right'),
                    ('scene', 'rel_action', 'scene'),
                    ('task', 'time_action', 'task'),
                ]
            ),
            edge_dim=self.graph.edge_dim,
            dropout=0.0,
            norm='layer',
        ).to(config['device'])

        ## Setup Output heads
        ## TODO: Mohit, think about the output heads, do we need separate heads for left and right gripper?
        self.prediction_head_left = MLP(
            [config['hidden_dim'], self.local_embd_dim, 3], 
            act='GELU',
            plain_last=True,
            norm='layer_norm',
        )
        self.prediction_head_right = MLP(
            [config['hidden_dim'], self.local_embd_dim, 3], 
            act='GELU',
            plain_last=True,
            norm='layer_norm',
        )
        self.prediction_head_rot_left = MLP(
            [config['hidden_dim'], self.local_embd_dim, 3], 
            act='GELU',
            plain_last=True,
            norm='layer_norm',
        )
        self.prediction_head_rot_right = MLP(
            [config['hidden_dim'], self.local_embd_dim, 3], 
            act='GELU',
            plain_last=True,
            norm='layer_norm',
        )
        self.prediction_head_g_left = MLP(
            [config['hidden_dim'], self.local_embd_dim, 1], 
            act='GELU',
            plain_last=True,
            norm='layer_norm',
        )
        self.prediction_head_g_right = MLP(
            [config['hidden_dim'], self.local_embd_dim, 1], 
            act='GELU',
            plain_last=True,
            norm='layer_norm',
        )

    def reinit_graphs(self, batch_size, num_demos=None):
        self.batch_size = batch_size
        if num_demos is not None:
            self.num_demos = num_demos
            self.graph.num_demos = num_demos
        self.graph.batch_size = batch_size
        self.graph.initialise_graph()

    def compile_models(self):
        print("Compiling models with torch.compile...")
        self.local_encoder = torch.compile(self.local_encoder)
        self.cond_encoder = torch.compile(self.cond_encoder)
        self.action_encoder = torch.compile(self.action_encoder)

    def get_labels(self, gt_actions_left, gt_actions_right, noisy_actions_left, noisy_actions_right,
                   gt_grips_left, gt_grips_right, noisy_grips_left, noisy_grips_right,
                   delta_grip=False, sep_rot=True):
        ## gt_actions_left: [bs, pred_horizon, 4, 4]
        ## gt_actions_right: [bs, pred_horizon, 4, 4]
        ## noisy_actions_left: [bs, pred_horizon, 4, 4]
        ## noisy_actions_right: [bs, pred_horizon, 4, 4]
        ## gt_grips_left: [bs, pred_horizon, 1]
        ## gt_grips_right: [bs, pred_horizon, 1]
        ## noisy_grips_left: [bs, pred_horizon, 1]
        ## noisy_grips_right: [bs, pred_horizon, 1]

        ## Get the gripper node positions from the current graph i.e. the base 6 positions.
        ## We just want these no transformation required here, they are just placeholders.
        gripper_points = self.graph.gripper_node_pos[None, None, :].repeat(
            gt_actions_left.shape[0], gt_actions_left.shape[1], 1, 1)
        
        if sep_rot:
            ## This is fine, my actions are in the Gripper Frame, so the calculation make sense.
            ## Noise and model output are also expected to be relative to the gripper frame.
            T_w_n_left = noisy_actions_left.view(-1, 4, 4)
            T_w_n_right = noisy_actions_right.view(-1, 4, 4)
            T_n_w_left = torch.inverse(T_w_n_left)
            T_n_w_right = torch.inverse(T_w_n_right)
            T_w_g_left = gt_actions_left.view(-1, 4, 4)
            T_w_g_right = gt_actions_right.view(-1, 4, 4)
            T_n_g_left = torch.bmm(T_n_w_left, T_w_g_left)
            T_n_g_right = torch.bmm(T_n_w_right, T_w_g_right)
            T_n_g_left = T_n_g_left.view(gt_actions_left.shape[0], gt_actions_left.shape[1], 4, 4)
            T_n_g_right = T_n_g_right.view(gt_actions_right.shape[0], gt_actions_right.shape[1], 4, 4)

            labels_trans_left = T_n_g_left[..., :3, 3][:, :, None, :].repeat(
                1, 1, gripper_points.shape[-2], 1
            )
            labels_trans_right = T_n_g_right[..., :3, 3][:, :, None, :].repeat(
                1, 1, gripper_points.shape[-2], 1
            )
            T_n_g_left[..., :3, 3] = 0
            T_n_g_right[..., :3, 3] = 0
            ## Gripper points can only be in zero zero here.
            labels_rot_left = self.graph.transform_gripper_nodes(gripper_points, T_n_g_left) - gripper_points
            labels_rot_right = self.graph.transform_gripper_nodes(gripper_points, T_n_g_right) - gripper_points
            labels = torch.cat([labels_trans_left, labels_rot_left, labels_trans_right, labels_rot_right], dim=-1)
        else:
            gripper_points_gt_left = self.graph.transform_gripper_nodes(gripper_points, gt_actions_left)
            gripper_points_gt_right = self.graph.transform_gripper_nodes(gripper_points, gt_actions_right)
            gripper_points_noisy_left = self.graph.transform_gripper_nodes(gripper_points, noisy_actions_left)
            gripper_points_noisy_right = self.graph.transform_gripper_nodes(gripper_points, noisy_actions_right)
            labels = torch.cat([
                gripper_points_noisy_left - gripper_points_gt_left,
                gripper_points_noisy_right - gripper_points_gt_right,
            ])

        if delta_grip:
            labels_grip_left = gt_grips_left - noisy_grips_left
            labels_grip_right = gt_grips_right - noisy_grips_right
        else:
            labels_grip_left = gt_grips_left
            labels_grip_right = gt_grips_right
        labels_grip_left = labels_grip_left[:, :, None, :].repeat(1, 1, gripper_points.shape[-2], 1)
        labels_grip_right = labels_grip_right[:, :, None, :].repeat(1, 1, gripper_points.shape[-2], 1)
        labels_grip = torch.cat([labels_grip_left, labels_grip_right], dim=-1)
        labels = torch.cat([labels, labels_grip], dim=-1)  # [bs, pred_horizon, num_gripper_nodes, 6]
        return labels

    def get_transformed_node_pos(self, actions, transform=True):
        gripper_points = self.graph.gripper_node_pos[None, None, :].repeat(actions.shape[0], actions.shape[1], 1, 1)
        if transform:
            gripper_points = self.graph.transform_gripper_nodes(gripper_points, actions)
        return gripper_points
    

    def forward(self, data):
        ## below two lines are not supposed to run ever in our code. For now.
        if not hasattr(data, 'demo_scene_node_embds'):
            data.demo_scene_node_embds, data.demo_scene_node_pos = self.get_demo_scene_emb(data)
        if not hasattr(data, 'live_scene_node_embds'):
            data.live_scene_node_embds, data.live_scene_node_pos = self.get_live_scene_emb(data)
        
        ## Copy the current observation to the future observations.
        ## As we have the obs in the world frame, we don't need to do this.
        ## Our Scene Embedding remains tha same as the current obs.
        # current_obs = to_dense_batch(data.pos_obs, data.batch_pos_obs, fill_value=0)[0]
        # current_obs = current_obs[:, None, ...].repeat(1, self.pred_horizon, 1, 1)
        # current_obs = current_obs.view(self.batch_size * self.pred_horizon, -1, 3)
        # actions_left = data.actions_left.view(-1, 4, 4)
        # actions_right = data.actions_right.view(-1, 4, 4)

        # ## We want things in left gripper frame
        # ## I think this is where we copy the current PC to future PCs, But see that 
        # ## The action_left might be noisy so the PC will be wrong
        # current_obs = torch.bmm(actions_left[:, :3, :3].transpose(1, 2), current_obs.permute(0, 2, 1)).permute(0, 2, 1)
        # current_obs -= actions_left[:, :3, 3][:, None, :]

        # ## I am unsure if this will work, Transformation to the left gripper frame might be tricky.
        # action_batch = torch.arange(current_obs.shape[0], device=current_obs.device)[:, None].repeat(
        #     1, current_obs.shape[1]
        # )
        # action_batch = action_batch.view(-1)
        # current_obs = current_obs.reshape(-1, 3)

        # pos_obs_old = data.pos_obs.clone()
        # batch_pos_obs_old = data.batch_pos_obs.clone()

        # data.pos_obs = current_obs
        # data.batch_pos_obs = action_batch

        # action_scene_node_embds, action_scene_node_pos = self.get_live_scene_emb(data)

        # data.pos_obs = pos_obs_old
        # data.batch_pos_obs = batch_pos_obs_old

        data.action_scene_node_embds = data.live_scene_node_embds[:, None, :, :].repeat(
            1, self.pred_horizon, 1, 1
        )
        data.action_scene_node_pos = data.live_scene_node_pos[:, None, :, :].repeat(
            1, self.pred_horizon, 1, 1
        )

        # data.action_scene_node_embds = action_scene_node_embds.view(
        #     self.batch_size, self.pred_horizon, -1, self.local_embd_dim
        # )
        # data.action_scene_node_pos = action_scene_node_pos.view(
        #     self.batch_size, self.pred_horizon, -1, 3
        # )

        ########################################################################
        self.graph.update_graph(data)

        ## Might want to remove this
        # torch.compiler.cudagraph_mark_step_begin()

        ## Check this from 
        x_dict = self.local_encoder(
            self.graph.graph.x_dict,
            self.graph.graph.edge_index_dict,
            self.graph.graph.edge_attr_dict,
        )
        x_dict = self.cond_encoder(
            x_dict,
            self.graph.graph.edge_index_dict,
            self.graph.graph.edge_attr_dict,
        )
        x_dict = self.action_encoder(
            x_dict,
            self.graph.graph.edge_index_dict,
            self.graph.graph.edge_attr_dict,
        )
        ###########################################################################
        x_gripper_left = x_dict['gripper_left'][self.graph.graph.gripper_left_time > self.traj_horizon].view(
            self.batch_size,
            self.pred_horizon,
            self.graph.num_g_nodes,
            -1
        )
        x_gripper_right = x_dict['gripper_right'][self.graph.graph.gripper_right_time > self.traj_horizon].view(
            self.batch_size,
            self.pred_horizon,
            self.graph.num_g_nodes,
            -1
        )
        preds_t_left = self.prediction_head_left(x_gripper_left)
        preds_t_right = self.prediction_head_right(x_gripper_right)
        preds_rot_left = self.prediction_head_rot_left(x_gripper_left)
        preds_rot_right = self.prediction_head_rot_right(x_gripper_right)
        preds_g_left = self.prediction_head_g_left(x_gripper_left)
        preds_g_right = self.prediction_head_g_right(x_gripper_right)
        ## Below should match the output setup of the get_labels function.
        preds = torch.cat([preds_t_left, preds_rot_left, preds_t_right, preds_rot_right, preds_g_left, preds_g_right], dim=-1)
        return preds
    
    def get_demo_scene_emb(self, data):
        bs = data.actions_left.shape[0]
        demo_scene_node_embds, demo_scene_node_pos, demo_scene_node_batch = \
            self.scene_encoder(
                None,
                data.pos_demos,
                data.batch_demos,
            )
        demo_scene_node_embds = to_dense_batch(demo_scene_node_embds, demo_scene_node_batch, fill_value=0)[0]
        demo_scene_node_embds = demo_scene_node_embds.view(bs, self.num_demos, self.traj_horizon, -1,
                                                           self.local_embd_dim)
        demo_scene_node_pos = to_dense_batch(demo_scene_node_pos, demo_scene_node_batch, fill_value=0)[0]
        demo_scene_node_pos = demo_scene_node_pos.view(bs, self.num_demos, self.traj_horizon, -1, 3)
        return demo_scene_node_embds, demo_scene_node_pos
    
    def get_live_scene_emb(self, data):
        current_scene_node_embds, current_scene_node_pos, current_scene_node_batch = \
            self.scene_encoder(
                None,
                data.pos_obs,
                data.batch_pos_obs,
            )
        current_scene_node_embds = to_dense_batch(current_scene_node_embds, current_scene_node_batch, fill_value=0)[0]
        current_scene_node_pos = to_dense_batch(current_scene_node_pos, current_scene_node_batch, fill_value=0)[0]
        return current_scene_node_embds, current_scene_node_pos