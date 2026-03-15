from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import math
import numpy as np

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class Elflx(LeggedRobot):
    #----------------------------------------
    
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.time_to_stand_still = torch.zeros(self.num_envs,
                            dtype=torch.float,
                            device=self.device,
                            requires_grad=False)
        self.static_delay = self.cfg.commands.static_delay

    # # 定义参考相位与参考轨迹
    # def  _get_phase(self):
    #     cycle_time = self.cfg.rewards.cycle_time
    #     phase = torch.fmod(self.episode_length_buf * self.dt / cycle_time, 1.0)
    #     level = (self.time_to_stand_still.unsqueeze(1) / self.cfg.commands.static_delay).clip(max=1.)
    #     # level = torch.zeros_like(self.time_to_stand_still.unsqueeze(1))
    #     return phase, level
    
    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        level = (self.time_to_stand_still.unsqueeze(1) / self.cfg.commands.static_delay).clip(max=1.)
        return phase, level

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase, level = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 4), device=self.device)
        
        # foot0&3 stance
        stance_mask[:, [0, 3]] = (sin_pos > 0).float().unsqueeze(1)  # 需要升维保证形状匹配
        # foot1&2 stance
        stance_mask[:, [1, 2]] = (sin_pos < 0).float().unsqueeze(1)

        # 4_leg support phase
        stance_mask = torch.where(
            level == 1.,
            torch.ones((self.num_envs, 4), device=self.device),
            stance_mask)

        return stance_mask

    def step(self, actions):
        return super().step(actions)
    
    # # progress 仿真与样机的控制频率与decimation相关
    # def step(self, actions, cmd):
    #     """ Apply actions, simulate, call self.post_physics_step()

    #     Args:
    #         actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
    #     """
    #     # 执行命令
    #     clip_actions = self.cfg.normalization.clip_actions
    #     self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
    #     self.cmd = torch.clip(cmd, -clip_actions, clip_actions).to(self.device)
    #     # step physics and render each frame
    #     self.render()
    #     for _ in range(self.cfg.control.decimation):
    #         self.torques = self._compute_torques(self.cmd).view(self.torques.shape)
    #         self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
    #         self.gym.simulate(self.sim)
    #         if self.device == 'cpu':
    #             self.gym.fetch_results(self.sim, True)
    #         self.gym.refresh_dof_state_tensor(self.sim)
    #     self.post_physics_step()

    #     # return clipped obs, clipped states (None), rewards, dones and infos
    #     clip_obs = self.cfg.normalization.clip_observations
    #     self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
    #     if self.privileged_obs_buf is not None:
    #         self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
    #     return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    # def check_termination(self):
    #     """ Check if environments need to be reset
    #     """
    #     self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
    #     self.time_out_buf = self.episode_length_buf > 10 * self.max_episode_length # no terminal reward for time-outs
    #     self.reset_buf |= self.time_out_buf
    

    # 创建智能体
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        self.mass_mask = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        print("linkname: ", self.cfg.asset.foot_name)
        print("feet_name: ", feet_names)
        print("body_name: ", body_names)

    
    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")

        if self.cfg.domain_rand.randomize_all_mass:
            randomize_indices = {2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16}
            for s in randomize_indices:
            # for s in range(len(props)):
                rng = self.cfg.domain_rand.rd_mass_range
                rd_num = np.random.uniform(rng[0], rng[1])
                self.mass_mask[env_id, s] = rd_num
                props[s].mass += rd_num

        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[1].mass += np.random.uniform(rng[0], rng[1]) 
        
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body1 {i}: {p.mass} (after randomization)")
        #     print(f"Total mass1 {sum} (after randomization)")

        return props
    
    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        command_mode = torch.rand((len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 0] *= command_mode >= 0.4 # x vel
        self.commands[env_ids, 1] *= ((command_mode >= 0.2) & (command_mode < 0.4)) | (command_mode >= 0.6) # y vel
        self.commands[env_ids, 2] *= ((command_mode >0.1) & (command_mode < 0.2)) | (command_mode >= 0.6) # ang vel 

        self.time_to_stand_still[env_ids] = 0.
        contacts = self.contact_forces[:, self.feet_indices, 2] > 5.
        double_contact = torch.sum(1.*contacts, dim=-1) == 4
        self.time_to_stand_still[env_ids] += self.static_delay * double_contact[env_ids] * \
            (torch.norm(self.commands[env_ids, :3], dim=-1) == 0) * \
            (torch.norm(self.base_euler_xyz[env_ids, :2], dim=-1) < 0.3)

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    # 检查奖励 检查终止
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = torch.clone(self.last_actions[:]) 
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def compute_observations(self):
        """ Computes observations
        """
        phase, level = self._get_phase()
        condition = (level == 1.0).bool()
    
        # 确保 `torch.sin` 的结果是 float32
        sin_vals = torch.sin(2 * torch.pi * phase).unsqueeze(1).to(torch.float32)
        sin_pos = torch.where(condition, torch.tensor(0.0, device=self.device, dtype=torch.float32), sin_vals)
        
        cos_vals = torch.cos(2 * torch.pi * phase).unsqueeze(1).to(torch.float32)
        cos_pos = torch.where(condition, torch.tensor(0.0, device=self.device, dtype=torch.float32), cos_vals)
        # sin_pos = torch.where(level == 1., 0, torch.sin(2 * torch.pi * phase).unsqueeze(1))
        # cos_pos = torch.where(level == 1., 0, torch.cos(2 * torch.pi * phase).unsqueeze(1))
        
        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)

        # base_lin_vel_obs = self.base_lin_vel * self.obs_scales.lin_vel
        # print("Base Linear Velocity:", base_lin_vel_obs)

        base_ang_vel_obs = self.base_ang_vel * self.obs_scales.ang_vel
        # base_ang_vel_obs = torch.tensor([[1., 1., 1.]], device='cuda:0')
        # print("Base Angular Velocity:", base_ang_vel_obs)

        projected_gravity_obs = self.projected_gravity
        # print(projected_gravity_obs)
        # projected_gravity_obs = torch.tensor([[1., 1., -1.]], device='cuda:0')
        # print("Projected Gravity:", projected_gravity_obs)

        # 设置速度命令为0 0 0
        # self.commands[:, :3] = 0
        commands_obs = self.commands[:, :3] * self.commands_scale
        # commands_obs = torch.tensor([[ 0., 0., 0.]], device='cuda:0')
        # print("Velocity Commands:", commands_obs)

        dof_pos_obs = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        # angle = 1 - math.cos(2 * math.pi / 2500 * t)
        # dof_pos_obs = torch.tensor([[ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')
        # dof_pos_obs = torch.tensor([[ angle, angle, angle, angle, angle, angle, angle, angle, angle, angle, angle, angle]], device='cuda:0')
        # print("Joint States:", dof_pos_obs)

        dof_vel_obs = self.dof_vel * self.obs_scales.dof_vel
        # dof_vel_obs = torch.tensor([[ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')
        # print("Joint Velocities:", dof_vel_obs)

        actions_obs = self.actions
        # actions_obs = torch.tensor([[ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')
        # print("Joint Actions:", actions_obs)

        self.obs_buf = torch.cat((  
                                    self.base_ang_vel  * self.obs_scales.ang_vel,                   # [3] Base angular velocity [5:47]
                                    self.base_euler_xyz * self.obs_scales.quat,                                         # [3] Projected gravity
                                    self.command_input,                                             # [2 + 3] sin cos and Velocity commands  只取前三个[x vel, y vel, yaw vel, heading]
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,# [12] Joint states
                                    self.dof_vel * self.obs_scales.dof_vel,                         # [12] Joint velocities
                                    self.actions                                                    # [12] Joint actions
                                    ),dim=-1)
        
        self.privileged_obs_buf = torch.cat((  
                                    self.base_lin_vel * self.obs_scales.lin_vel,                    # [3] Base linear velocity
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.base_euler_xyz * self.obs_scales.quat,
                                    self.command_input,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)

        # print("Observation Buffer (obs_buf):", self.obs_buf.cpu().numpy())
        # add perceptive inputs if not blind
        
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[45:232] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #------------ reward functions----------------
    
    # 步态辅助奖励 对角触底奖励
    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1.0, -0.3)
        # print('stancemask&contact', stance_mask, contact)
        return torch.mean(reward, dim=1)
    
    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    # # 惩罚hip关节侧翻    
    # def _reward_hip_pos(self):
    #     return torch.sum(torch.square(self.dof_pos[:,[0,3,6,9]]), dim=1)
    def _reward_hip_pos(self):
        """
        0.009 为hip关节打开程度，可根据期望程度调整
        0.27为足端距离{base}xoy平面距离，后续可加在cfg中定义
        """
        hip_pos = torch.sum(torch.square(self.dof_pos[:,[0,3,6,9]]), dim=1) 
        hip_pos_rad = 0.060 * (((self.commands[:, 1])) ** 2 + ((0.36 * self.commands[:, 2])) ** 2)  # 第一个参数：与y轴速度相关 第二个参数：与yaw速度相关 为hip关节打开程度，可根据期望程度调整
        hip_pos = torch.clamp(hip_pos - hip_pos_rad, 0, 10)
        return hip_pos

    # 惩罚关节位置接近其限制的情况
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    # 惩罚trot行动不一致, 需对应calf thigh对角关节 
    def _reward_tracking_trot(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]] - self.dof_pos[:,[10,11,4,5]]), dim=1)

    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
    
    # 奖励机器人在执行长步时的空中时间
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time.clamp(0, 0.32) - 0.1) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        
        self.feet_air_time *= ~contact_filt
        return rew_airTime
        
    # def _reward_feet_swing_height(self):
    #     contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
    #     pos_error = torch.square(self.feet_pos[:, :, 2] - 0.1) * ~contact
    #     return torch.sum(pos_error, dim=(1))
    
    # def _reward_feet_swing_height(self):
    #     """
    #     0.1为期望摆线高度
    #     todo 可在cfg.reward中定义
    #     """
    #     # 计算接触状态
    #     contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
    #     # 初始化最大高度变量（如果尚未初始化）
    #     if not hasattr(self, 'max_feet_height'):
    #         self.max_feet_height = torch.zeros(self.feet_pos.shape[0], self.feet_pos.shape[1], dtype=self.feet_pos.dtype, device=self.feet_pos.device)
    #     # 更新最大高度
    #     current_height = self.feet_pos[:, :, 2]
    #     max_height_condition = current_height > self.max_feet_height
    #     self.max_feet_height = torch.where(max_height_condition, current_height, self.max_feet_height)
    #     # 仅在摆动结束时计算惩罚
    #     # 假设摆动结束时脚部接触地面
    #     end_of_swing = ~contact
    #     pos_error = torch.square(self.max_feet_height - 0.05) * end_of_swing
    #     # 重置最大高度变量以准备下一个摆动周期
    #     self.max_feet_height = torch.where(end_of_swing, torch.zeros_like(self.max_feet_height), self.max_feet_height)
    #     # 计算惩罚
    #     return torch.sum(pos_error, dim=1)
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        trot_thigh_calf_1 = joint_diff[:,[7,8,10,11]]           # 左后 右后
        trot_thigh_calf_2 = joint_diff[:,[4,5,1,2]]           # 右前 左前
        trot_thigh_calf = torch.norm(trot_thigh_calf_1, dim=1) + torch.norm(trot_thigh_calf_2, dim=1)
        trot_thigh_calf = torch.clamp(trot_thigh_calf - 0.1, 0, 50)
        return torch.exp(-trot_thigh_calf * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_joint_power_limit(self):
        """
        惩罚电机功率过大，
        当电机功率大于103时进行惩罚 joint_power - 103
        """
        joint_power = torch.abs(self.dof_vel) * torch.abs(self.torques)
        excessive_power = (joint_power - 103).clip(min=0.)
        return torch.sum(excessive_power, dim=1)
    
    # def _reward_punish_one_leg_air(self):
    #     """
    #     惩罚机器人一条腿一直不着地的行为
    #     """
    #     # 检测足端是否触地
    #     contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        
    #     # 初始化单腿不触地时间变量（如果尚未初始化）
    #     if not hasattr(self, 'single_leg_air_time'):
    #         self.single_leg_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        
    #     # 更新单腿不触地时间
    #     # 使用矢量化操作代替显式循环
    #     self.single_leg_air_time = torch.where(
    #         contact,  # 条件：如果触地
    #         torch.zeros_like(self.single_leg_air_time),  # 如果触地，重置为0
    #         self.single_leg_air_time + self.dt  # 如果不触地，增加时间
    #     )
        
    #     # 惩罚阈值（超过这个时间就进行惩罚）
    #     punish_threshold = 0.5  # 单位：秒
        
    #     # 计算惩罚值
    #     punish = torch.clamp(self.single_leg_air_time - punish_threshold, min=0.) * 10  # 惩罚系数可以根据需要调整
        
    #     # 总惩罚值
    #     total_punish = torch.sum(punish, dim=1)
        
    #     return total_punish
    
    # 惩罚机器人的一条腿长时间不触地

    def _reward_long_air_time(self):
        # 定义一个最大允许的空中时间阈值
        max_air_time = 0.7  # 可以根据实际情况调整这个值
        # 检查当前帧哪些腿没有接触地面
        in_air = self.contact_forces[:, self.feet_indices, 2] <= 1.
        # 构造一个掩码，触地的腿对应的值为0，未触地的腿对应的值为1
        mask = in_air.float()
        # 更新空中时间：如果腿在空中，空中时间加1；如果腿接触地面，空中时间重置为0
        self.air_time = torch.where(in_air, self.air_time + self.dt, torch.zeros_like(self.air_time))
        # print(self.air_time)
        # 检查每条腿的空中时间是否超过阈值
        long_air = self.air_time > max_air_time
        # 计算惩罚值，对超过阈值的腿进行惩罚
        punish = torch.sum(long_air * self.dt, dim=1)  # 惩罚值与超过的时间成正比
        # 只有当有腿长时间不触地时才给予惩罚
        punish = punish * (torch.sum(long_air, dim=1) > 0) * 10
        return punish  # 返回惩罚值
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (self.time_to_stand_still > self.static_delay)