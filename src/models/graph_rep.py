import torch
from torch import nn
from torch_geometric.data import HeteroData

from ip.utils.common_utils import printarr, PositionalEncoder, SinusoidalPosEmb


class BimanualGraphRep(nn.Module):
    def __init__(self, config):
        super().__init__()
        ########################################
        # STORE PARAMETERS
        ######################################## 
        self.batch_size = config['batch_size']
        self.num_demos = config['num_demos'] ## 2
        self.traj_horizon = config['traj_horizon'] ## 10
        self.num_scenes_nodes = config['num_scenes_nodes'] ## 16
        self.num_freqs = config['local_num_freq'] ## 10, kine frequency band use karne he position encoding ya sinusoidal embedding ke liye
        self.device = config['device']
        self.embd_dim = config['local_nn_dim'] ## 512
        self.pred_horizon = config['pred_horizon']
        self.pos_in_nodes = config['pos_in_nodes'] ## True, nodes me position bhi he, xyz
        ##########################################

        self.gripper_node_pos = torch.tensor([
            [0.0, 0.0, 0.0],  # Middle
            [0.0, 0.0, -0.03], # Tail
            [0.0, 0.03, 0.0],  # Side
            [0.0, -0.03, 0.0],  # Side
            [0.0, 0.03, 0.03],  # finger
            [0.0, -0.03, 0.03]  # finger
        ], dtype=torch.float32, device=self.device) * 2  ## multiplied by 2 for reasons known to god
        self.num_g_nodes = len(self.gripper_node_pos) ## 6
        self.g_state_dim = 64
        self.d_time_dim = 64

        #############################################
        self.sine_pose_embd = SinusoidalPosEmb(self.d_time_dim) ## 64
        ## Learable embeddings
        self.pos_embd = PositionalEncoder(3, self.num_freqs, log_space=True, add_original_x=True, scale=1.0) ## N, 63 as input 3, for each input sin and cos and 10 freq
        self.edge_dim = self.pos_embd.d_output * 2 ## d_output = 63, * 2 = 126
        self.gripper_left_proj = nn.Linear(1, self.g_state_dim) ## gripper open close state, 1 dim to 64 dim
        self.gripper_right_proj = nn.Linear(1, self.g_state_dim)

        self.gripper_left_embds = nn.Embedding(
            len(self.gripper_node_pos) * (self.pred_horizon + 1), ## TODO: Why +1
            self.embd_dim - self.g_state_dim, 
            device=self.device ## 6 * 11 = 66, 512 - 64 = 448
        )
        self.gripper_right_embds = nn.Embedding(
            len(self.gripper_node_pos) * (self.pred_horizon + 1), ##
            self.embd_dim - self.g_state_dim,
            device=self.device ## 6 * 11 = 66, 512 - 64 = 448
        )
        ## Make seperate embeddings for left and right grippers to give model freedom
        ## single edge embd for conditioning, same as in the original IP
        self.gripper_cond_gripper_embds_left = nn.Embedding(1, self.edge_dim, device=self.device) ## 1, 126
        self.gripper_cond_gripper_embds_right = nn.Embedding(1, self.edge_dim, device=self.device) ## 1, 126
        ## This is not even used rigth now, but I am keeping it for future use
        self.gripper_da_gripper_embds_left = nn.Embedding(1, self.edge_dim, device=self.device) ## 1, 126
        self.gripper_da_gripper_embds_right = nn.Embedding(1, self.edge_dim, device=self.device)

        #################################################
        ## Structure of the graph
        self.node_types = ['scene', 'gripper_left', 'gripper_right']
        self.edge_types = [
            ########################################
            ### Connect Local Obs
            # intra-connections
            ('scene', 'rel', 'scene'),
            ('gripper_left', 'rel', 'gripper_left'),
            ('gripper_right', 'rel', 'gripper_right'),
            # inter-connections  scene to grippers
            ('scene', 'rel', 'gripper_left'),
            ('scene', 'rel', 'gripper_right'),
            # inter-connections gripper to other gripper
            ## For me, I want the bidirectional information between the gripper to communicate between them
            ('gripper_left', 'rel', 'gripper_right'),
            ('gripper_right', 'rel', 'gripper_left'),

            ### Gripper Conditioning
            ('gripper_left', 'cond', 'gripper_left'), ## Condition gripper to current gripper
            ('gripper_right', 'cond', 'gripper_right'), ## Condition gripper to
            ## Gripper in demo
            ('gripper_left', 'demo', 'gripper_left'), ## Demo gripper to current gripper
            ('gripper_right', 'demo', 'gripper_right'), ## Demo gripper to current gripper
            ## Grippers in action
            ('gripper_left', 'time_action', 'gripper_left'), ## Current gripper
            ('gripper_right', 'time_action', 'gripper_right'), ## Current gripper
            ## Gripper in demo to action
            ('gripper_left', 'demo_action', 'gripper_left'), ## Demo gripper to current gripper
            ('gripper_right', 'demo_action', 'gripper_right'), ## Demo gripper to current gripper
        ]
        self.graph = None

    def create_dense_edge_idx(self, num_nodes_source, num_nodes_dest):
        return torch.cartesian_prod(
            torch.arange(num_nodes_source, dtype=torch.int64, device=self.device),
            torch.arange(num_nodes_dest, dtype=torch.int64, device=self.device)).contiguous().t()
    
    def transform_gripper_nodes(self, gripper_nodes, T):
        # gripper_nodes - [B, D, T, N, 3]
        # T - [B, D, T, 4, 4]
        has_demo = len(gripper_nodes.shape) == 5
        if not has_demo:
            gripper_nodes = gripper_nodes.unsqueeze(1)
        b, d, t, n, _ = gripper_nodes.shape
        gripper_nodes = gripper_nodes.reshape(-1, gripper_nodes.shape[-2], gripper_nodes.shape[-1]).permute(0, 2, 1)
        gripper_nodes = torch.bmm(T[..., :3, :3].reshape(-1, 3, 3), gripper_nodes)
        gripper_nodes += T[..., :3, 3].reshape(-1, 3, 1)
        gripper_nodes = gripper_nodes.permute(0, 2, 1).view(b, d, t, n, 3)
        if not has_demo:
            gripper_nodes = gripper_nodes.squeeze(1)
        return gripper_nodes
    
    def get_node_info(self):
        # A bunch of arange operations to store information which node in the graph belongs to which batch, timestep etc.
        # First the scene nodes. [bs, nd, th, sn, 3] + [bs, sn, 3]
        ################################################################################################################
        ## Batch info for scene nodes
        sb = torch.arange(self.batch_size, device=self.device) ## [B] , [0, 1, 2, ..., B-1]
        ## Assing Batch, Demo, Traj, Node Num
        scene_batch = sb[:, None, None, None].repeat(
            1,
            self.num_demos, ## 2
            self.traj_horizon, ## 10
            self.num_scenes_nodes ## 16
        ).view(-1) ## [B, 2, 10, 16] -> [B*2*10*16] ## 
        ## BATCH, NUM SCENE NODES
        sb_current = sb[:, None].repeat(1, self.num_scenes_nodes).view(-1) ## [B, 16] -> [B*16]
        ## BATCH, DEMO, TRAJ, NUM SCENE NODES + BATCH, NUM SCENE NODES
        scene_batch = torch.cat([scene_batch, sb_current], dim=0) ## [B*2*10*16 + B*16]

        ## Traj, Batch, Demo, BATCH , Num Scene Nodes
        scene_traj = torch.arange(self.traj_horizon, device=self.device)[None, None, :, None].repeat(
            self.batch_size,
            self.num_demos,
            1,
            self.num_scenes_nodes
        ).view(-1) ## [B, 2, 10, 16] -> [B*2*10*16]
        scene_traj = torch.cat([scene_traj, self.traj_horizon * torch.ones_like(sb_current)], dim=0)

        scene_demo = torch.arange(self.num_demos, device=self.device)[None, :, None, None].repeat(
            self.batch_size,
            1,
            self.traj_horizon,
            self.num_scenes_nodes
        ).view(-1)
        scene_current = self.num_demos * torch.ones(self.batch_size * self.num_scenes_nodes, device=self.device)
        scene_demo = torch.cat([scene_demo, scene_current], dim=0)

        ## Scene action nodes
        ## BATCH, PRED HORIZON, NUM SCENE NODES
        scene_batch_action = sb[:, None, None].repeat(1, self.pred_horizon, self.num_scenes_nodes).view(-1)
        scene_batch = torch.cat([scene_batch, scene_batch_action], dim=0)

        scene_traj_action = torch.arange(self.pred_horizon, device=self.device)[None, :, None].repeat(
            self.batch_size,
            1,
            self.num_scenes_nodes
        ).view(-1) + self.traj_horizon + 1
        scene_traj = torch.cat([scene_traj, scene_traj_action], dim=0)
        scene_demo = torch.cat([scene_demo, self.num_demos * torch.ones_like(scene_traj_action)], dim=0)

        ################################################################################################################
        ## Bimanual Gripper Nodes
        # Now the gripper nodes. [bs, nd, th, gn, 3] + [bs, gn, 3] + [bs, ph, gn, 3]
        ## These are just indices, we don't actually need seperate for left and right grippers but
        ## I am doint this for clairity
        ## Left
        gripper_left_batch = sb[:, None, None, None].repeat(1, self.num_demos, self.traj_horizon, self.num_g_nodes).view(-1)
        gripper_left_batch_current = sb[:, None].repeat(1, self.num_g_nodes).view(-1)
        gripper_left_batch_action = sb[:, None, None].repeat(1, self.pred_horizon, self.num_g_nodes).view(-1)
        gripper_left_batch = torch.cat([gripper_left_batch, gripper_left_batch_current, gripper_left_batch_action], dim=0)

        ## Right
        gripper_right_batch = sb[:, None, None, None].repeat(1, self.num_demos, self.traj_horizon, self.num_g_nodes).view(-1)
        gripper_right_batch_current = sb[:, None].repeat(1, self.num_g_nodes).view(-1)
        gripper_right_batch_action = sb[:, None, None].repeat(1, self.pred_horizon, self.num_g_nodes).view(-1)
        gripper_right_batch = torch.cat([gripper_right_batch, gripper_right_batch_current, gripper_right_batch_action], dim=0)

        ## Time
        ## Left
        gripper_left_time = torch.arange(self.traj_horizon, device=self.device, dtype=torch.long)[None, None, :,
                            None].repeat(self.batch_size, self.num_demos, 1, self.num_g_nodes).view(-1)
        gripper_left_time_current = self.traj_horizon * torch.ones(self.batch_size * self.num_g_nodes, device=self.device,
                                                                   dtype=torch.long)
        gripper_left_time_action = torch.arange(self.pred_horizon, device=self.device, dtype=torch.long)[None, :,
                                 None].repeat(self.batch_size, 1, self.num_g_nodes).view(-1)
        gripper_left_time = torch.cat([gripper_left_time,
                                      gripper_left_time_current,
                                        gripper_left_time_action + self.traj_horizon + 1], dim=0)
        ## Right
        gripper_right_time = torch.arange(self.traj_horizon, device=self.device, dtype=torch.long)[None, None, :,
                             None].repeat(self.batch_size, self.num_demos, 1, self.num_g_nodes).view(-1)
        gripper_right_time_current = self.traj_horizon * torch.ones(self.batch_size * self.num_g_nodes, device=self.device,
                                                                    dtype=torch.long)
        gripper_right_time_action = torch.arange(self.pred_horizon, device=self.device, dtype=torch.long)[None, :,
                                  None].repeat(self.batch_size, 1, self.num_g_nodes).view(-1)
        gripper_right_time = torch.cat([gripper_right_time,
                                       gripper_right_time_current,
                                        gripper_right_time_action + self.traj_horizon + 1], dim=0)

        ## Gripper Node
        ## Left
        gripper_left_node = torch.arange(self.num_g_nodes, device=self.device)[None, None, None, :].repeat(
            self.batch_size, self.num_demos, self.traj_horizon, 1).view(-1)
        gripper_left_node_current = torch.arange(self.num_g_nodes, device=self.device)[None, :].repeat(
            self.batch_size, 1).view(-1)
        gripper_left_node_action = torch.arange(self.num_g_nodes, device=self.device)[None, None, :].repeat(
            self.batch_size, self.pred_horizon, 1).view(-1)
        gripper_left_node = torch.cat([gripper_left_node, gripper_left_node_current, gripper_left_node_action], dim=0)

        ## Right
        gripper_right_node = torch.arange(self.num_g_nodes, device=self.device)[None, None, None, :].repeat(
            self.batch_size, self.num_demos, self.traj_horizon, 1).view(-1)
        gripper_right_node_current = torch.arange(self.num_g_nodes, device=self.device)[None, :].repeat(
            self.batch_size, 1).view(-1)
        gripper_right_node_action = torch.arange(self.num_g_nodes, device=self.device)[None, None, :].repeat(
            self.batch_size, self.pred_horizon, 1).view(-1)
        gripper_right_node = torch.cat([gripper_right_node, gripper_right_node_current, gripper_right_node_action], dim=0)

        ## Embd
        ## Left
        gripper_left_embd = gripper_left_node
        gripper_left_embd[gripper_left_time > self.traj_horizon] += self.num_g_nodes * gripper_left_time_action
        ## Right
        gripper_right_embd = gripper_right_node
        gripper_right_embd[gripper_right_time > self.traj_horizon] += self.num_g_nodes * gripper_right_time_action

        ## Demo
        ## Left
        gripper_left_demo = torch.arange(self.num_demos, device=self.device)[None, :, None, None].repeat(
            self.batch_size, 1, self.traj_horizon, self.num_g_nodes).view(-1)
        gripper_left_current = self.num_demos * torch.ones(self.batch_size * (self.pred_horizon + 1) * self.num_g_nodes,
                                                         device=self.device)
        gripper_left_demo = torch.cat([gripper_left_demo, gripper_left_current], dim=0)
        ## Right
        gripper_right_demo = torch.arange(self.num_demos, device=self.device)[None, :, None, None].repeat(
            self.batch_size, 1, self.traj_horizon, self.num_g_nodes).view(-1)
        gripper_right_current = self.num_demos * torch.ones(self.batch_size * (self.pred_horizon + 1) * self.num_g_nodes,
                                                          device=self.device)
        gripper_right_demo = torch.cat([gripper_right_demo, gripper_right_current], dim=0)

        return {
            'scene': {
                'batch': scene_batch,
                'traj': scene_traj,
                'demo': scene_demo,
            },
            'gripper_left': {
                'batch': gripper_left_batch,
                'time': gripper_left_time,
                'node': gripper_left_node,
                'embd': gripper_left_embd,
                'demo': gripper_left_demo,
            },
            'gripper_right': {
                'batch': gripper_right_batch,
                'time': gripper_right_time,
                'node': gripper_right_node,
                'embd': gripper_right_embd,
                'demo': gripper_right_demo,
            }
        }

    def initialise_graph(self):
        # Manually connecting different nodes in the graph to achieve our desired graph representation.
        # Probably could be re-written to be more beautiful. Most definitely could.
        ## Extend to bimanual
        self.graph = HeteroData()
        node_info = self.get_node_info()

        ## Scene
        dense_s_s = self.create_dense_edge_idx(node_info['scene']['batch'].shape[0],
                                                  node_info['scene']['batch'].shape[0])
        ## Grippers
        dense_gl_gl = self.create_dense_edge_idx(node_info['gripper_left']['embd'].shape[0],
                                                 node_info['gripper_left']['embd'].shape[0])
        dense_gr_gr = self.create_dense_edge_idx(node_info['gripper_right']['embd'].shape[0],
                                                 node_info['gripper_right']['embd'].shape[0])
        dense_gl_gr = self.create_dense_edge_idx(node_info['gripper_left']['embd'].shape[0],
                                                 node_info['gripper_right']['embd'].shape[0])
        dense_gr_gl = self.create_dense_edge_idx(node_info['gripper_right']['embd'].shape[0],
                                                    node_info['gripper_left']['embd'].shape[0])
        ## Scene <--> Grippers
        dense_s_gl = self.create_dense_edge_idx(node_info['scene']['batch'].shape[0],
                                                node_info['gripper_left']['embd'].shape[0])
        dense_s_gr = self.create_dense_edge_idx(node_info['scene']['batch'].shape[0],
                                                node_info['gripper_right']['embd'].shape[0])
        
        ################################################################################################################
        ## Scene to Scene Mask
        s_rel_s_mask = node_info['scene']['batch'][dense_s_s[0, :]] == node_info['scene']['batch'][dense_s_s[1, :]]
        s_rel_s_mask = s_rel_s_mask & (
                node_info['scene']['traj'][dense_s_s[0, :]] == node_info['scene']['traj'][dense_s_s[1, :]])
        s_rel_s_mask = s_rel_s_mask & (
                node_info['scene']['demo'][dense_s_s[0, :]] == node_info['scene']['demo'][dense_s_s[1, :]])
        ################################################################################################################
        ## Scene to Scene Action Mask
        s_rel_s_action_mask = s_rel_s_mask & (
                node_info['scene']['traj'][dense_s_s[0, :]] > self.traj_horizon)
        s_rel_s_action_mask = s_rel_s_action_mask & (
                node_info['scene']['traj'][dense_s_s[1, :]] > self.traj_horizon)
        s_rel_s_mask_demo = s_rel_s_mask & torch.logical_not(s_rel_s_action_mask)
        ################################################################################################################
        ## Gripper Left to Gripper Left Mask
        gl_rel_gl_mask = node_info['gripper_left']['batch'][dense_gl_gl[0, :]] == node_info['gripper_left']['batch'][dense_gl_gl[1, :]]
        gl_rel_gl_mask = gl_rel_gl_mask & (
                node_info['gripper_left']['time'][dense_gl_gl[0, :]] == node_info['gripper_left']['time'][dense_gl_gl[1, :]])
        gl_rel_gl_mask = gl_rel_gl_mask & (
                node_info['gripper_left']['demo'][dense_gl_gl[0, :]] == node_info['gripper_left']['demo'][dense_gl_gl[1, :]])
        ################################################################################################################
        ## Gripper Right to Gripper Right Mask
        gr_rel_gr_mask = node_info['gripper_right']['batch'][dense_gr_gr[0, :]] == node_info['gripper_right']['batch'][dense_gr_gr[1, :]]
        gr_rel_gr_mask = gr_rel_gr_mask & (
                node_info['gripper_right']['time'][dense_gr_gr[0, :]] == node_info['gripper_right']['time'][dense_gr_gr[1, :]])
        gr_rel_gr_mask = gr_rel_gr_mask & (
                node_info['gripper_right']['demo'][dense_gr_gr[0, :]] == node_info['gripper_right']['demo'][dense_gr_gr[1, :]])
        ################################################################################################################
        ## Gripper Left to Gripper Right Mask
        gl_rel_gr_mask = node_info['gripper_left']['batch'][dense_gl_gr[0, :]] == node_info['gripper_right']['batch'][dense_gl_gr[1, :]]
        gl_rel_gr_mask = gl_rel_gr_mask & (
                node_info['gripper_left']['time'][dense_gl_gr[0, :]] == node_info['gripper_right']['time'][dense_gl_gr[1, :]])
        gl_rel_gr_mask = gl_rel_gr_mask & (
                node_info['gripper_left']['demo'][dense_gl_gr[0, :]] == node_info['gripper_right']['demo'][dense_gl_gr[1, :]])
        ## Gripper Right to Gripper Left Mask
        gr_rel_gl_mask = node_info['gripper_right']['batch'][dense_gr_gl[0, :]] == node_info['gripper_left']['batch'][dense_gr_gl[1, :]]
        gr_rel_gl_mask = gr_rel_gl_mask & (
                node_info['gripper_right']['time'][dense_gr_gl[0, :]] == node_info['gripper_left']['time'][dense_gr_gl[1, :]])  
        gr_rel_gl_mask = gr_rel_gl_mask & (
                node_info['gripper_right']['demo'][dense_gr_gl[0, :]] == node_info['gripper_left']['demo'][dense_gr_gl[1, :]])
        ################################################################################################################
        ## Scene to Gripper Left Mask
        s_rel_gl_mask = node_info['scene']['batch'][dense_s_gl[0, :]] == node_info['gripper_left']['batch'][dense_s_gl[1, :]]
        s_rel_gl_mask = s_rel_gl_mask & (
                node_info['scene']['traj'][dense_s_gl[0, :]] == node_info['gripper_left']['time'][dense_s_gl[1, :]])
        s_rel_gl_mask = s_rel_gl_mask & (
                node_info['scene']['demo'][dense_s_gl[0, :]] == node_info['gripper_left']['demo'][dense_s_gl[1, :]])
        ################################################################################################################
        ## Scene to Gripper Right Mask
        s_rel_gr_mask = node_info['scene']['batch'][dense_s_gr[0, :]] == node_info['gripper_right']['batch'][dense_s_gr[1, :]]
        s_rel_gr_mask = s_rel_gr_mask & (
                node_info['scene']['traj'][dense_s_gr[0, :]] == node_info['gripper_right']['time'][dense_s_gr[1, :]])
        s_rel_gr_mask = s_rel_gr_mask & (
                node_info['scene']['demo'][dense_s_gr[0, :]] == node_info['gripper_right']['demo'][dense_s_gr[1, :]])
        ################################################################################################################
        ## Scene to Gripper Left Action Mask
        s_rel_gl_action_mask = s_rel_gl_mask & (
                node_info['scene']['traj'][dense_s_gl[0, :]] > self.traj_horizon)
        s_rel_gl_action_mask = s_rel_gl_action_mask & (
                node_info['gripper_left']['time'][dense_s_gl[1, :]] > self.traj_horizon)
        s_rel_gl_mask_demo = s_rel_gl_mask & torch.logical_not(s_rel_gl_action_mask)
        ################################################################################################################
        ## Scene to Gripper Right Action Mask
        s_rel_gr_action_mask = s_rel_gr_mask & (
                node_info['scene']['traj'][dense_s_gr[0, :]] > self.traj_horizon)
        s_rel_gr_action_mask = s_rel_gr_action_mask & (
                node_info['gripper_right']['time'][dense_s_gr[1, :]] > self.traj_horizon)
        s_rel_gr_mask_demo = s_rel_gr_mask & torch.logical_not(s_rel_gr_action_mask)
        ################################################################################################################
        ## Gripper Left to Gripper Left Conditioning Mask
        gl_c_gl_mask = node_info['gripper_left']['batch'][dense_gl_gl[0, :]] == node_info['gripper_left']['batch'][dense_gl_gl[1, :]]
        gl_c_gl_mask = gl_c_gl_mask & (
                node_info['gripper_left']['time'][dense_gl_gl[0, :]] < self.traj_horizon)
        gl_c_gl_mask = gl_c_gl_mask & (node_info['gripper_left']['time'][dense_gl_gl[1, :]] == self.traj_horizon)
        ################################################################################################################
        ## Gripper Right to Gripper Right Conditioning Mask
        gr_c_gr_mask = node_info['gripper_right']['batch'][dense_gr_gr[0, :]] == node_info['gripper_right']['batch'][dense_gr_gr[1, :]]
        gr_c_gr_mask = gr_c_gr_mask & (
                node_info['gripper_right']['time'][dense_gr_gr[0, :]] < self.traj_horizon)
        gr_c_gr_mask = gr_c_gr_mask & (node_info['gripper_right']['time'][dense_gr_gr[1, :]] == self.traj_horizon)
        ################################################################################################################
        ## Gripper Left to Gripper Left Time Action Mask
        gl_t_gl_mask = node_info['gripper_left']['batch'][dense_gl_gl[0, :]] == node_info['gripper_left']['batch'][dense_gl_gl[1, :]]
        gl_t_gl_mask = gl_t_gl_mask & (node_info['gripper_left']['time'][dense_gl_gl[0, :]] >= self.traj_horizon)
        gl_t_gl_mask = gl_t_gl_mask & (node_info['gripper_left']['time'][dense_gl_gl[1, :]] > self.traj_horizon)
        gl_t_gl_mask = gl_t_gl_mask & (
                node_info['gripper_left']['time'][dense_gl_gl[1, :]] != node_info['gripper_left']['time'][dense_gl_gl[0, :]])
        gl_tc_gl = gl_t_gl_mask & (node_info['gripper_left']['time'][dense_gl_gl[0, :]] == self.traj_horizon)
        gl_t_gl_mask = gl_t_gl_mask & torch.logical_not(gl_tc_gl)
        ################################################################################################################
        ## Gripper Right to Gripper Right Time Action Mask
        gr_t_gr_mask = node_info['gripper_right']['batch'][dense_gr_gr[0, :]] == node_info['gripper_right']['batch'][dense_gr_gr[1, :]]
        gr_t_gr_mask = gr_t_gr_mask & (node_info['gripper_right']['time'][dense_gr_gr[0, :]] >= self.traj_horizon)
        gr_t_gr_mask = gr_t_gr_mask & (node_info['gripper_right']['time'][dense_gr_gr[1, :]] > self.traj_horizon)
        gr_t_gr_mask = gr_t_gr_mask & (
                node_info['gripper_right']['time'][dense_gr_gr[1, :]] != node_info['gripper_right']['time'][dense_gr_gr[0, :]])
        gr_tc_gr = gr_t_gr_mask & (node_info['gripper_right']['time'][dense_gr_gr[0, :]] == self.traj_horizon)
        gr_t_gr_mask = gr_t_gr_mask & torch.logical_not(gr_tc_gr)
        ################################################################################################################
        ## Gripper Left to Gripper Left Demo Mask
        gl_d_gl_mask = node_info['gripper_left']['batch'][dense_gl_gl[0, :]] == node_info['gripper_left']['batch'][dense_gl_gl[1, :]]
        gl_d_gl_mask = gl_d_gl_mask & (node_info['gripper_left']['time'][dense_gl_gl[0, :]] < self.traj_horizon)
        gl_d_gl_mask = gl_d_gl_mask & (node_info['gripper_left']['time'][dense_gl_gl[1, :]] < self.traj_horizon)
        gl_d_gl_mask = gl_d_gl_mask & (
                node_info['gripper_left']['time'][dense_gl_gl[0, :]] != node_info['gripper_left']['time'][dense_gl_gl[1, :]])
        gl_d_gl_mask = gl_d_gl_mask & (
                node_info['gripper_left']['demo'][dense_gl_gl[0, :]] == node_info['gripper_left']['demo'][dense_gl_gl[1, :]])
        gl_d_gl_mask = gl_d_gl_mask & (node_info['gripper_left']['time'][dense_gl_gl[1, :]] - node_info['gripper_left']['time'][
            dense_gl_gl[0, :]] == -1)
        ################################################################################################################
        ## Gripper Right to Gripper Right Demo Mask
        gr_d_gr_mask = node_info['gripper_right']['batch'][dense_gr_gr[0, :]] == node_info['gripper_right']['batch'][dense_gr_gr[1, :]]
        gr_d_gr_mask = gr_d_gr_mask & (node_info['gripper_right']['time'][dense_gr_gr[0, :]] < self.traj_horizon)
        gr_d_gr_mask = gr_d_gr_mask & (node_info['gripper_right']['time'][dense_gr_gr[1, :]] < self.traj_horizon)
        gr_d_gr_mask = gr_d_gr_mask & (
                node_info['gripper_right']['time'][dense_gr_gr[0, :]] != node_info['gripper_right']['time'][dense_gr_gr[1, :]])
        gr_d_gr_mask = gr_d_gr_mask & (
                node_info['gripper_right']['demo'][dense_gr_gr[0, :]] == node_info['gripper_right']['demo'][dense_gr_gr[1, :]])
        gr_d_gr_mask = gr_d_gr_mask & (node_info['gripper_right']['time'][dense_gr_gr[1, :]] - node_info['gripper_right']['time'][
            dense_gr_gr[0, :]] == -1)
        ################################################################################################################
        ## Set values in graph
        self.graph.gripper_left_batch = node_info['gripper_left']['batch']
        self.graph.gripper_left_time = node_info['gripper_left']['time']
        self.graph.gripper_left_node = node_info['gripper_left']['node']
        self.graph.gripper_left_embd = node_info['gripper_left']['embd'].long()
        self.graph.gripper_left_demo = node_info['gripper_left']['demo']
        self.graph.gripper_right_batch = node_info['gripper_right']['batch']
        self.graph.gripper_right_time = node_info['gripper_right']['time']
        self.graph.gripper_right_node = node_info['gripper_right']['node']
        self.graph.gripper_right_embd = node_info['gripper_right']['embd'].long()
        self.graph.gripper_right_demo = node_info['gripper_right']['demo']
        self.graph.scene_batch = node_info['scene']['batch']
        self.graph.scene_traj = node_info['scene']['traj']
        self.graph.scene_demo = node_info['scene']['demo']  

        ## Add edges
        ## Local
        self.graph[('scene', 'rel', 'scene')].edge_index = dense_s_s[:, s_rel_s_mask]
        self.graph[('scene', 'rel', 'gripper_left')].edge_index = dense_s_gl[:, s_rel_gl_mask]
        self.graph[('scene', 'rel', 'gripper_right')].edge_index = dense_s_gr[:, s_rel_gr_mask]
        self.graph[('gripper_left', 'rel', 'gripper_left')].edge_index = dense_gl_gl[:, gl_rel_gl_mask]
        self.graph[('gripper_right', 'rel', 'gripper_right')].edge_index = dense_gr_gr[:, gr_rel_gr_mask]
        self.graph[('gripper_left', 'rel', 'gripper_right')].edge_index = dense_gl_gr[:, gl_rel_gr_mask]
        self.graph[('gripper_right', 'rel', 'gripper_left')].edge_index = dense_gr_gl[:, gr_rel_gl_mask]
    
        ## Conditioning
        self.graph[('gripper_left', 'cond', 'gripper_left')].edge_index = dense_gl_gl[:, gl_c_gl_mask]
        self.graph[('gripper_right', 'cond', 'gripper_right')].edge_index = dense_gr_gr[:, gr_c_gr_mask]
        ## Time Action
        self.graph[('gripper_left', 'time_action', 'gripper_left')].edge_index = dense_gl_gl[:, gl_t_gl_mask]
        self.graph[('gripper_right', 'time_action', 'gripper_right')].edge_index = dense_gr_gr[:, gr_t_gr_mask]
        ## Demo
        self.graph[('gripper_left', 'demo', 'gripper_left')].edge_index = dense_gl_gl[:, gl_d_gl_mask]
        self.graph[('gripper_right', 'demo', 'gripper_right')].edge_index = dense_gr_gr[:, gr_d_gr_mask]

        ## Demo Action
        self.graph[('scene', 'rel_action', 'gripper_left')].edge_index = dense_s_gl[:, s_rel_gl_action_mask]
        self.graph[('scene', 'rel_action', 'gripper_right')].edge_index = dense_s_gr[:, s_rel_gr_action_mask]
        self.graph[('scene', 'rel_demo', 'gripper_left')].edge_index = dense_s_gl[:, s_rel_gl_mask_demo]
        self.graph[('scene', 'rel_demo', 'gripper_right')].edge_index = dense_s_gr[:, s_rel_gr_mask_demo]
        self.graph[('scene', 'rel_action', 'scene')].edge_index = dense_s_s[:, s_rel_s_action_mask]
        self.graph[('scene', 'rel_demo', 'scene')].edge_index = dense_s_s[:, s_rel_s_mask_demo]
        self.graph[('gripper_left', 'rel_cond', 'gripper_left')].edge_index = dense_gl_gl[:, gl_tc_gl]
        self.graph[('gripper_right', 'rel_cond', 'gripper_right')].edge_index = dense_gr_gr[:, gr_tc_gr]

    """ Below will need some understanding of data processing the train function I guess. """
    def update_graph(self, data):
        # Adding information to the graph structure create in initialise_graph.
        # scene_node_pos: # [B, N, T, S, 3]
        gripper_node_pos = self.gripper_node_pos[None, None, None, :, :].repeat(
            self.batch_size, self.num_demos, self.traj_horizon, 1, 1,)
        ########################################################################
        # demo_T_w_es: [B, D, T, 4, 4]
        # T_w_e: [B, 4, 4]
        # T_w_n: [B, P, 4, 4]
        # Create identity matrix like T_w_e for gripper nodes
        # ## Below can be used by both grippers, this is like the current position
        # I_w_e = torch.eye(4, device=self.device)[None, :, :].repeat(self.batch_size, 1, 1)

        # all_T_w_e_left = torch.cat([
        #     data.demo_T_w_es_left[:, :self.num_demos, :, None, :, :].repeat(1, 1, 1, 6, 1, 1).view(-1, 4, 4),
        #     I_w_e[:, None, :, :].repeat(1, 6, 1, 1).view(-1, 4, 4),
        #     data.actions_left[:, :, None, :, :].repeat(1, 1, 6, 1, 1).view(-1, 4, 4)
        # ])
        # ## I is good for left, but where is the right?
        # all_T_left_right_e = torch.cat([
        #     data.demo_T_left_es_right[:, :self.num_demos, :, None, :, :].repeat(1, 1, 1, 6, 1, 1).view(-1, 4, 4),
        #     I_w_e[:, None, :, :].repeat(1, 6, 1, 1).view(-1, 4, 4),
        #     data.actions_right[:, :, None, :, :].repeat(1, 1, 6, 1, 1).view(-1, 4, 4)
        # ])
        # all_T_e_w_left = all_T_w_e_left.inverse()
        # all_T_right_left_e = all_T_left_right_e.inverse()
        # ################################################################################
        gripper_node_pos_current = self.gripper_node_pos[None, None, None, :, :].repeat(
            self.batch_size, 1, 1, 1, 1)
        gripper_node_pos_action = self.gripper_node_pos[None, None, :, :].repeat(
            self.batch_size, 1, self.pred_horizon, 1, 1)
        
        # for k, v in data.items():
        #     print(f"{k}: {v.shape}") if isinstance(v, torch.Tensor) else None

        ## Move the gripper poses to the world frame.
        gripper_node_pos_demo_left = self.transform_gripper_nodes(gripper_node_pos, data.demo_T_w_es_left)
        gripper_node_pos_demo_right = self.transform_gripper_nodes(gripper_node_pos, data.demo_T_w_es_right)

        ## Transform the current gripper pose to the world frame.
        gripper_node_pos_current_left = self.transform_gripper_nodes(gripper_node_pos_current, data.T_w_es_left[:, None, None, ...])
        gripper_node_pos_current_right = self.transform_gripper_nodes(gripper_node_pos_current, data.T_w_es_right[:, None, None, ...])

        ## Action are relative motions.
        ## TODO: @mohit check this, we might want to use the rel edges properly. 
        ## Below will right now add rel edges based on the current gripper while the actions are
        ## relative. Also, note that actions are delta with respect to each other.
        gripper_node_pos_action_left = self.transform_gripper_nodes(gripper_node_pos_action, data.actions_left[:, None, ...])
        gripper_node_pos_action_right = self.transform_gripper_nodes(gripper_node_pos_action, data.actions_right[:, None, ...])

        gripper_node_pos_left = torch.cat([
            gripper_node_pos_demo_left.reshape(-1, 3),
            gripper_node_pos_current_left.reshape(-1, 3),
            gripper_node_pos_action_left.reshape(-1, 3)
        ], dim=0)
        gripper_node_pos_right = torch.cat([
            gripper_node_pos_demo_right.reshape(-1, 3),
            gripper_node_pos_current_right.reshape(-1, 3),
            gripper_node_pos_action_right.reshape(-1, 3)
        ], dim=0)
        
        # gripper_node_pos = torch.cat([gripper_node_pos.reshape(-1, 3),
        #                               gripper_node_pos_current.reshape(-1, 3),
        #                               gripper_node_pos_action.reshape(-1, 3)], dim=0)
        
        # data.grasp_demo [B, D, T, 1]
        #########################################
        ### Below is for grippers states
        #########################################
        gripper_left_states = self.gripper_left_proj(data.grasp_left_demo[:, :self.num_demos])[..., None, :].repeat(
            1, 1, 1, self.num_g_nodes, 1
        )
        gripper_left_states = gripper_left_states.view(-1, self.g_state_dim)
        gripper_left_states_current = self.gripper_left_proj(data.current_grip_left.unsqueeze(-1))[..., None, :].repeat(
            1, self.num_g_nodes, 1
        )
        gripper_left_states_current = gripper_left_states_current.view(-1, self.g_state_dim)
        gripper_left_states_action = self.gripper_left_proj(data.actions_grip_left.unsqueeze(-1))[..., None, :].repeat(
            1, 1, self.num_g_nodes, 1
        )
        gripper_left_states_action = gripper_left_states_action.view(-1, self.g_state_dim)
        gripper_left_states = torch.cat([
            gripper_left_states, 
            gripper_left_states_current, 
            gripper_left_states_action
        ], dim=0)
        gripper_left_embd = self.gripper_left_embds(self.graph.gripper_left_embd)

        gripper_right_states = self.gripper_right_proj(data.grasp_right_demo[:, :self.num_demos])[..., None, :].repeat(
            1, 1, 1, self.num_g_nodes, 1
        )
        gripper_right_states = gripper_right_states.view(-1, self.g_state_dim)
        gripper_right_states_current = self.gripper_right_proj(data.current_grip_right.unsqueeze(-1))[..., None, :].repeat(
            1, self.num_g_nodes, 1
        )
        gripper_right_states_current = gripper_right_states_current.view(-1, self.g_state_dim)
        gripper_right_states_action = self.gripper_right_proj(data.actions_grip_right.unsqueeze(-1))[..., None, :].repeat(
            1, 1, self.num_g_nodes, 1
        )
        gripper_right_states_action = gripper_right_states_action.view(-1, self.g_state_dim)
        gripper_right_states = torch.cat([
            gripper_right_states, 
            gripper_right_states_current, 
            gripper_right_states_action
        ], dim=0)
        gripper_right_embd = self.gripper_right_embds(self.graph.gripper_right_embd)

        ## Add diffusion time step information to gripper action nodes.
        d_time_embd = self.sine_pose_embd(data.diff_time)[:, None, ...].repeat(
            1, self.pred_horizon, self.num_g_nodes, 1
        ).view(-1, self.d_time_dim)
        gripper_left_embd[self.graph.gripper_left_time > self.traj_horizon][:, -self.d_time_dim:] = d_time_embd
        gripper_right_embd[self.graph.gripper_right_time > self.traj_horizon][:, -self.d_time_dim:] = d_time_embd

        gripper_left_embd = torch.cat([gripper_left_embd, gripper_left_states], dim=-1)
        gripper_right_embd = torch.cat([gripper_right_embd, gripper_right_states], dim=-1)
        #########################################
        #########################################

        ## Scene node position and embedding
        scene_node_pos = torch.cat([
            data.demo_scene_node_pos[:, :self.num_demos].reshape(-1, 3),
            data.live_scene_node_pos.view(-1, 3),
            data.action_scene_node_pos.view(-1, 3)
        ], dim=0)
        scene_node_embd = torch.cat([
            data.demo_scene_node_embds[:, :self.num_demos].reshape(-1, self.embd_dim),
            data.live_scene_node_embds.view(-1, self.embd_dim),
            data.action_scene_node_embds.view(-1, self.embd_dim)
        ], dim=0)

        self.graph['gripper_left'].pos = gripper_node_pos_left
        self.graph['gripper_left'].x = gripper_left_embd
        self.graph['gripper_right'].pos = gripper_node_pos_right
        self.graph['gripper_right'].x = gripper_right_embd
        self.graph['scene'].pos = scene_node_pos
        self.graph['scene'].x = scene_node_embd

        if self.pos_in_nodes:
            self.graph['gripper_left'].x = \
                torch.cat([self.graph['gripper_left'].x, self.pos_embd(self.graph['gripper_left'].pos)], dim=-1)
            self.graph['gripper_right'].x = \
                torch.cat([self.graph['gripper_right'].x, self.pos_embd(self.graph['gripper_right'].pos)], dim=-1)
            self.graph['scene'].x = \
                torch.cat([self.graph['scene'].x, self.pos_embd(self.graph['scene'].pos)], dim=-1)

        ## Add relative edge attributes for gripper nodes.
        self.add_rel_edge_attr('scene', 'scene')
        self.add_rel_edge_attr('gripper_left', 'gripper_left')
        self.add_rel_edge_attr('gripper_right', 'gripper_right')
        self.add_rel_edge_attr('gripper_left', 'gripper_right')
        self.add_rel_edge_attr('gripper_right', 'gripper_left')
        self.add_rel_edge_attr('scene', 'gripper_left')
        self.add_rel_edge_attr('scene', 'gripper_right')

        self.graph[('gripper_left', 'cond', 'gripper_left')].edge_attr = \
            self.gripper_cond_gripper_embds_left(
                torch.zeros(len(self.graph[('gripper_left', 'cond', 'gripper_left')].edge_index[0]),
                            device=self.device).long()
            )
        self.graph[('gripper_right', 'cond', 'gripper_right')].edge_attr = \
            self.gripper_cond_gripper_embds_right(
                torch.zeros(len(self.graph[('gripper_right', 'cond', 'gripper_right')].edge_index[0]),
                            device=self.device).long()
            )
        
        self.add_rel_edge_attr('scene', 'gripper_left', edge='rel_action')
        self.add_rel_edge_attr('scene', 'gripper_right', edge='rel_action')
        self.add_rel_edge_attr('scene', 'gripper_left', edge='rel_demo')
        self.add_rel_edge_attr('scene', 'gripper_right', edge='rel_demo')

        self.add_rel_edge_attr('scene', 'scene', edge='rel_action')
        self.add_rel_edge_attr('scene', 'scene', edge='rel_demo')

        self.add_rel_edge_attr('gripper_left', 'gripper_left', edge='time_action')
        self.add_rel_edge_attr('gripper_left', 'gripper_left', edge='rel_cond')
        self.add_rel_edge_attr('gripper_left', 'gripper_left', edge='demo')
        self.add_rel_edge_attr('gripper_right', 'gripper_right', edge='time_action')
        self.add_rel_edge_attr('gripper_right', 'gripper_right', edge='rel_cond')
        self.add_rel_edge_attr('gripper_right', 'gripper_right', edge='demo')

        # self.add_rel_edge_attr('gripper_left', 'gripper_left', edge='time_action',
        #                        all_T_w_e = all_T_w_e_left, all_T_e_w = all_T_e_w_left)
        # self.add_rel_edge_attr('gripper_left', 'gripper_left', edge='rel_cond',
        #                        all_T_w_e = all_T_w_e_left, all_T_e_w = all_T_e_w_left)
        # self.add_rel_edge_attr('gripper_left', 'gripper_left', edge='demo',
        #                        all_T_w_e = all_T_w_e_left, all_T_e_w = all_T_e_w_left)
        # self.add_rel_edge_attr('gripper_right', 'gripper_right', edge='time_action',
        #                        all_T_w_e = all_T_left_right_e, all_T_e_w = all_T_right_left_e)
        # self.add_rel_edge_attr('gripper_right', 'gripper_right', edge='rel_cond',
        #                        all_T_w_e = all_T_left_right_e, all_T_e_w = all_T_right_left_e)
        # self.add_rel_edge_attr('gripper_right', 'gripper_right', edge='demo',
        #                        all_T_w_e = all_T_left_right_e, all_T_e_w = all_T_right_left_e)  
        
    def add_rel_edge_attr(self, source, dest, edge='rel', all_T_w_e=None, all_T_e_w=None):
        if all_T_w_e is None:
            pos_dest = self.graph[dest].pos[self.graph[(source, edge, dest)].edge_index[1]]
            pos_source = self.graph[source].pos[self.graph[(source, edge, dest)].edge_index[0]]
            pos_dest_rot = pos_dest
        else:
            pos_source = self.graph[source].pos[self.graph[(source, edge, dest)].edge_index[0]]
            T_i_j = torch.bmm(all_T_e_w[self.graph[(source, edge, dest)].edge_index[0]],
                              all_T_w_e[self.graph[(source, edge, dest)].edge_index[1]])
            pos_dest_rot = torch.bmm(T_i_j[..., :3, :3], pos_source[..., None]).squeeze(-1)
            pos_dest = pos_source + T_i_j[..., :3, 3]
        self.graph[(source, edge, dest)].edge_attr = torch.cat([
            self.pos_embd(pos_dest - pos_source),
            self.pos_embd(pos_dest_rot - pos_source)
        ], dim=-1)