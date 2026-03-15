from legged_gym.envs import ElflxRoughCfg, ElflxRoughCfgPPO

class ElflxFlatCfg( ElflxRoughCfg ):
    class env(ElflxRoughCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        episode_length_s = 20               # episode length in seconds
        num_envs = 2048
        use_ref_actions = False

    class terrain(ElflxRoughCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False
        static_friction = 0.8
        dynamic_friction = 0.6

    class asset (ElflxRoughCfg.asset):
        fix_base_link = False                 # 固定机身关节
        self_collisions = 1
        flip_visual_attachments = False

    class rewards(ElflxRoughCfg.rewards):
        only_positive_rewards = True       # if true negative total rewards are clipped at zero (avoids early termination problems)
        max_contact_force = 300.            # forces above this value are penalized
        base_height_target = 0.28
        soft_torque_limit = 1               # 扭矩的软限制，这里设置为 1.0，超过限制的扭矩将受到惩罚。

        # 定义参考周期 相位 参考转角 足端参考高度
        cycle_time = 0.4 #sec
        target_joint_pos_scale = 0.25    # thigh单位为rad calf需要乘以2rad
        
        target_feet_height = 0.06       # m  ?

        class scales (ElflxRoughCfg.rewards.scales):   
            termination = -1.
            orientation = 1

            base_height = -60               # 0321 40 基础高度的惩罚权重
            tracking_lin_vel = 1.8          # 0321 1.3 跟踪线速度的奖励
            tracking_ang_vel = 0.8          # 0321 0.5 跟踪角速度的奖励权重
            lin_vel_z = -5.0                # Z轴线速度的惩罚权重

            dof_pos_limits = -1.2              # 0321 -1.2 5
            joint_power_limit = -0
            torques = -0.00020              # 0321 0.0002 0.000040
            feet_air_time = 12.              # 0321 1.

            # dof_acc = -2.5e-7                # 关节加速度的惩罚权重 -2.5e-7
            dof_vel = -0.000005                 # 0321 0.00001 关节速度的惩罚权重
            # stand_still = -0.5               # 静止站立的惩罚权重
            action_rate = -0.006               # 动作频率太快的惩罚

            hip_pos = -0.7                     # 0321 -1.5 hip侧翻惩罚，根据elflx实际尺寸设计奖励函数，如果修改狗则需要修改奖励函数 0.65
            # alive = 0.05                       # 0.25 0321 无
            action_smoothness = -0.003          # 0321 0.003
            tracking_trot = -0.2            #0321 0  trot 对角关节跟踪惩罚，如果关节索引发生变化需要修改奖励函数
            # feet_swing_height = -100         # 摆动腿足抬腿高度
            contact_no_vel = -0.001
            # default_joint_pos = 1.2
            # vel_mismatch_exp = -0.05
            long_air_time = -0.1
            feet_contact_number = 0.6
            track_vel_hard = 0.5
            stand_still = -0.5
            punish_xy_angle_vel = -0.05

            joint_pos = 1.4
            feet_clearance = 0.8

    class commands(ElflxRoughCfg.commands):
        heading_command = False
        resampling_time = 4.
        static_delay = 180.
        class ranges(ElflxRoughCfg.commands.ranges):
            lin_vel_x = [-1.4, 1.4]         # min max [m/s]
            lin_vel_y = [-0.5, 0.5]         # min max [m/s]
            ang_vel_yaw = [-1, 1]       # min max [rad/s]

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.87

    class domain_rand(ElflxRoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-0.5,2.5]
        randomize_all_mass = True
        rd_mass_range = [0.01, 0.6]

        randomize_com = False
        rd_com_range = [-0.05, 0.05]

        randomize_base_com = True
        rd_base_com_range = [-0.05, 0.05]

        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.7
        
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]

        randomize_calculated_torque = True
        torque_multiplier_range = [0.8, 1.2]

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035] # Offset to add to the motor angles

        randomize_joint_friction = True
        joint_friction_range = [0.01, 1.15]

        randomize_joint_damping = True
        joint_damping_range = [0.03, 1.5]

        randomize_joint_armature = True
        joint_armature_range = [0.008, 0.06]

        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [1, 10]
        
        add_obs_latency = True # no latency for obs_action
        randomize_obs_motor_latency = True
        randomize_obs_imu_latency = True
        range_obs_motor_latency = [1, 10]
        range_obs_imu_latency = [1, 10]
    
    class noise(ElflxRoughCfg.noise):
        add_noise = True
        noise_level = 1. # scales other values
        class noise_scales(ElflxRoughCfg.noise.noise_scales):
            dof_pos = 0.07
            dof_vel = 2.0
            lin_vel = 0.1
            ang_vel = 0.2
            quat = 0.15
            gravity = 0.05
            height_measurements = 0.1

class ElflxFlatCfgPPO(ElflxRoughCfgPPO):
    class policy(ElflxRoughCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'

    class algorithm(ElflxRoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner (ElflxRoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'elflx_flat'
        load_run = -1
        max_iterations = 300