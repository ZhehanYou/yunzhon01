from legged_gym.envs import ElflxspineRoughCfg, ElflxspineRoughCfgPPO

class ElflxspineFlatCfg( ElflxspineRoughCfg ):
    class env(ElflxspineRoughCfg.env):
        num_observations = 51
        num_privileged_obs = 54
        episode_length_s = 20               # episode length in seconds
        num_envs = 4096

    class terrain(ElflxspineRoughCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False
        static_friction = 1.0
        dynamic_friction = 0.9

    class asset (ElflxspineRoughCfg.asset):
        fix_base_link = False               # 固定机身关节
        self_collisions = 1
        flip_visual_attachments = False

    class rewards(ElflxspineRoughCfg.rewards):
        only_positive_rewards = False       # if true negative total rewards are clipped at zero (avoids early termination problems)
        max_contact_force = 300.            # forces above this value are penalized
        base_height_target = 0.28
        soft_torque_limit = 1               # 扭矩的软限制，这里设置为 1.0，超过限制的扭矩将受到惩罚。

        class scales (ElflxspineRoughCfg.rewards.scales):   
            termination = -0.4      
            orientation = -0.5

            base_height = -60               # 0321 40 基础高度的惩罚权重
            tracking_lin_vel = 1.9          # 0321 1.3 跟踪线速度的奖励
            tracking_ang_vel = 0.8          # 0321 0.5 跟踪角速度的奖励权重
            lin_vel_z = -2.0                # Z轴线速度的惩罚权重

            dof_pos_limits = -1.2              # 0321 -1.2 5
            joint_power_limit = -0
            torques = -0.00020              # 0321 0.0002 0.000040
            feet_air_time = 12.              # 0321 1.

            # dof_acc = -2.5e-7                # 关节加速度的惩罚权重 -2.5e-7
            dof_vel = -0.000005                 # 0321 0.00001 关节速度的惩罚权重
            # stand_still = -0.5               # 静止站立的惩罚权重
            action_rate = -0.006               # 动作频率太快的惩罚

            hip_pos = -0.45                     # 0321 -1.5 hip侧翻惩罚，根据elflx实际尺寸设计奖励函数，如果修改狗则需要修改奖励函数 0.65
            # alive = 0.05                       # 0.25 0321 无
            action_smoothness = -0.003          # 0321 0.003
            tracking_trot = -0.2            #0321 0  trot 对角关节跟踪惩罚，如果关节索引发生变化需要修改奖励函数
            # feet_swing_height = -100         # 摆动腿足抬腿高度
            contact_no_vel = -0.001
            # default_joint_pos = 1.2
            # vel_mismatch_exp = -0.05
            # track_vel_hard = 0.05
            long_air_time = -0.1

    class commands(ElflxspineRoughCfg.commands):
        heading_command = False
        resampling_time = 4.
        class ranges(ElflxspineRoughCfg.commands.ranges):
            lin_vel_x = [-1.5, 1.5]         # min max [m/s]
            lin_vel_y = [-0.5, 0.5]         # min max [m/s]
            ang_vel_yaw = [-1, 1]       # min max [rad/s]


    class domain_rand(ElflxspineRoughCfg.domain_rand):
        friction_range = [0. , 1.5]
        

class ElflxspineFlatCfgPPO(ElflxspineRoughCfgPPO):
    class policy(ElflxspineRoughCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'


    class algorithm(ElflxspineRoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner (ElflxspineRoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'elflxspine_flat'
        load_run = -1
        max_iterations = 300