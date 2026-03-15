from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class ElflxRoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_envs = 2048
        num_actions = 12
        use_ref_actions = False
        
    class terrain(LeggedRobotCfg.terrain):
        mesg_type = 'trimesh'
        static_friction = 0.8
        dynamic_friction = 0.6
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 0 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.5, 0.5, 0, 0, 0, 0.0] # added = 1
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.34]  # 初始位置，z轴高度设置为0.34米，可以根据实际情况调整
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'AL_hip_Roll': 0.02,
            'AL_thigh_PitchT': 0.7854,
            'AL_calf_PitchC': 0.0,

            'AR_hip_Roll': -0.02,
            'AR_thigh_PitchT': 0.7854,
            'AR_calf_PitchC': 0.0,

            'BL_hip_Roll': 0.02,
            'BL_thigh_PitchT': 0.7854,
            'BL_calf_PitchC': 0.0,
            
            'BR_hip_Roll': -0.02,
            'BR_thigh_PitchT': 0.7854,
            'BR_calf_PitchC': 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {
                'hip': 80,
                'thigh': 80,
                'calf': 80,
            }
        damping = {
                'hip': 4.0,
                'thigh': 4.0,
                'calf': 4.2,
            }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25     # 动作缩放因子，用于将动作值转换为关节角度变化
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4          # 控制动作更新的频率

    class sim(LeggedRobotCfg.sim):
        # 周期/频率
        dt =  0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/elflx/meshes/elflx.urdf'  # URDF文件的路径
        name = "elflx"
        foot_name = "foot"  # 脚部链接的名称，用于计算奖励和终止条件 
        penalize_contacts_on = ["hip", "thigh", "calf"]  # 接触惩罚的链接名称
        terminate_after_contacts_on = ["trunk", "spine", "hip"]  # 接触后终止仿真的链接名称
        self_collisions = 1  # 启用自我碰撞检测 1为不启用 0为启用
        max_angular_velocity = 15. # by default set to 1000. in legged_robot_config.py
        max_linear_velocity = 15.
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-0.5,2.]
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

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.07
            dof_vel = 2.0
            lin_vel = 0.1
            ang_vel = 0.2
            quat = 0.15
            gravity = 0.05
            height_measurements = 0.1

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True       # if true negative total rewards are clipped at zero (avoids early termination problems)
        max_contact_force = 300.            # forces above this value are penalized
        base_height_target = 0.28
        soft_torque_limit = 1               # 扭矩的软限制，这里设置为 1.0，超过限制的扭矩将受到惩罚。

        # 定义参考周期 相位
        cycle_time = 0.4 #sec
        target_joint_pos_scale = 0.25    # thigh单位为rad calf需要乘以2rad
        
        target_feet_height = 0.06       # m  ?

        class scales (LeggedRobotCfg.rewards.scales):   
            termination = -1.
            orientation = 1

            base_height = -60               # 0321 40 基础高度的惩罚权重
            tracking_lin_vel = 1.8          # 0321 1.3 跟踪线速度的奖励
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
    
    class commands(LeggedRobotCfg.commands):
        heading_command = False
        resampling_time = 4.
        static_delay = 180.
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.4, 1.4]         # min max [m/s]
            lin_vel_y = [-0.7, 0.7]         # min max [m/s]
            ang_vel_yaw = [-1, 1]       # min max [rad/s]

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.87

class ElflxRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'elflx_rough'
        # load_run = -1