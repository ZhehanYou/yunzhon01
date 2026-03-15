# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096  # 环境数量
        num_observations = 235  # 状态空间的维度
        num_privileged_obs = None  # 特权状态的维度（用于不对称训练），如果为 None，则不返回特权状态
        num_actions = 12  # 动作空间的维度（通常对应机器人关节数量）
        env_spacing = 3.  # 环境之间的间距（不适用于高度场或三角网格地形）
        send_timeouts = True  # 是否向算法发送超时信息
        episode_length_s = 20  # 每个 episode 的时长（秒）

    class terrain:
        mesh_type = 'trimesh'  # 地形类型：'none', 'plane', 'heightfield', 'trimesh'
        horizontal_scale = 0.1  # 地形的水平缩放比例 [米]
        vertical_scale = 0.005  # 地形的垂直缩放比例 [米]
        border_size = 25  # 地形边界大小 [米]
        curriculum = True  # 是否启用地形课程学习
        static_friction = 1.0  # 静摩擦系数
        dynamic_friction = 1.0  # 动摩擦系数
        restitution = 0.  # 恢复系数
        measure_heights = True  # 是否测量地形高度
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 测量点的 x 坐标
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]  # 测量点的 y 坐标
        selected = False  # 是否选择特定地形类型
        terrain_kwargs = None  # 特定地形的参数字典
        max_init_terrain_level = 5  # 初始地形课程的最大等级
        terrain_length = 8.  # 地形长度 [米]
        terrain_width = 8.  # 地形宽度 [米]
        num_rows = 10  # 地形行数（课程等级）
        num_cols = 20  # 地形列数（地形类型）
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]  # 各种地形类型的比例
        slope_treshold = 0.75  # 三角网格地形的坡度阈值

    class commands:
        curriculum = False  # 是否启用命令课程学习
        max_curriculum = 1.  # 最大课程等级
        num_commands = 4  # 命令的数量（如线速度、角速度等）
        resampling_time = 10.  # 命令重新采样的时间间隔 [秒]
        heading_command = False  # 是否根据航向误差计算角速度命令
        class ranges:
            lin_vel_x = [-1.0, 1.0]  # x 方向线速度范围 [米/秒]
            lin_vel_y = [-1.0, 1.0]  # y 方向线速度范围 [米/秒]
            ang_vel_yaw = [-1, 1]  # 偏航角速度范围 [弧度/秒]
            heading = [-3.14, 3.14]  # 航向角范围 [弧度]

    class init_state:
        pos = [0.0, 0.0, 1.]  # 初始位置 [x, y, z] [米]
        rot = [0.0, 0.0, 0.0, 1.0]  # 初始旋转 [x, y, z, w] [四元数]
        lin_vel = [0.0, 0.0, 0.0]  # 初始线速度 [x, y, z] [米/秒]
        ang_vel = [0.0, 0.0, 0.0]  # 初始角速度 [x, y, z] [弧度/秒]
        default_joint_angles = {  # 动作为 0 时的默认关节角度
            "joint_a": 0., 
            "joint_b": 0.
        }

    class control:
        control_type = 'P'  # 控制类型：'P'（位置控制）、'V'（速度控制）、'T'（力矩控制）
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # 关节刚度 [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}  # 关节阻尼 [N*m*s/rad]
        action_scale = 0.5  # 动作缩放比例
        decimation = 4  # 每个策略时间步的控制更新次数

    class asset:
        file = ""  # 机器人模型文件路径
        name = "legged_robot"  # 机器人名称
        foot_name = "None"  # 足端名称（用于接触力计算）
        penalize_contacts_on = []  # 惩罚接触的部件
        terminate_after_contacts_on = []  # 接触后终止的部件
        disable_gravity = False  # 是否禁用重力
        collapse_fixed_joints = True  # 是否合并固定关节
        fix_base_link = False  # 是否固定机器人基座
        default_dof_drive_mode = 3  # 默认关节驱动模式
        self_collisions = 0  # 是否启用自碰撞检测
        replace_cylinder_with_capsule = True  # 是否用胶囊体替换圆柱体
        flip_visual_attachments = True  # 是否翻转视觉附件
        density = 0.001  # 机器人密度
        angular_damping = 0.  # 角阻尼
        linear_damping = 0.  # 线阻尼
        max_angular_velocity = 1000.  # 最大角速度
        max_linear_velocity = 1000.  # 最大线速度
        armature = 0.  # 转动惯量
        thickness = 0.01  # 碰撞体厚度

    class domain_rand:
        randomize_friction = True  # 是否随机化摩擦系数
        friction_range = [0.5, 1.25]  # 摩擦系数范围
        randomize_base_mass = False  # 是否随机化基座质量
        added_mass_range = [-1., 1.]  # 额外质量范围
        push_robots = True  # 是否随机推机器人
        push_interval_s = 15  # 推机器人的时间间隔 [秒]
        max_push_vel_xy = 1.  # 推力的最大速度 [米/秒]

    class rewards:
        class scales:
            termination = -0.0  # 提前终止的奖励权重
            tracking_lin_vel = 1.0  # 跟踪线速度的奖励权重
            tracking_ang_vel = 0.5  # 跟踪角速度的奖励权重
            lin_vel_z = -2.0  # Z 轴线速度的惩罚权重
            ang_vel_xy = -0.05  # XY 平面角速度的惩罚权重
            orientation = -0.  # 姿态的惩罚权重
            torques = -0.00001  # 扭矩的惩罚权重
            dof_vel = -0.  # 关节速度的惩罚权重
            dof_acc = -2.5e-7  # 关节加速度的惩罚权重
            base_height = -0.  # 基座高度的惩罚权重
            feet_air_time = 1.0  # 足端离地时间的奖励权重
            collision = -1.  # 碰撞的惩罚权重
            feet_stumble = -0.0  # 足端绊倒的惩罚权重
            action_rate = -0.01  # 动作频率的惩罚权重
            stand_still = -0.  # 静止站立的惩罚权重
        
        only_positive_rewards = True  # 是否将负奖励裁剪为 0
        tracking_sigma = 0.25  # 跟踪误差的容忍度
        soft_dof_pos_limit = 1.  # 关节位置的软限制
        soft_dof_vel_limit = 1.  # 关节速度的软限制
        soft_torque_limit = 1.  # 扭矩的软限制
        base_height_target = 1.  # 基座高度目标值
        max_contact_force = 100.  # 最大接触力

    class normalization:
        class obs_scales:
            lin_vel = 2.0  # 线速度的归一化尺度
            ang_vel = 0.25  # 角速度的归一化尺度
            dof_pos = 1.0  # 关节位置的归一化尺度
            dof_vel = 0.05  # 关节速度的归一化尺度
            height_measurements = 5.0  # 高度测量的归一化尺度
            quat = 1.0  # 四元数的归一化尺度
        clip_observations = 100.  # 观测值的裁剪范围
        clip_actions = 100.  # 动作值的裁剪范围

    class noise:
        add_noise = True  # 是否添加噪声
        noise_level = 1.0  # 噪声强度
        class noise_scales:
            dof_pos = 0.01  # 关节位置噪声
            dof_vel = 1.5  # 关节速度噪声
            lin_vel = 0.1  # 线速度噪声
            ang_vel = 0.2  # 角速度噪声
            gravity = 0.05  # 重力噪声
            height_measurements = 0.1  # 高度测量噪声

    class viewer:
        ref_env = 0  # 参考环境索引
        pos = [10, 0, 6]  # 摄像机位置 [米]
        lookat = [11., 5, 3.]  # 摄像机目标点 [米]

    class sim:
        dt = 0.005  # 仿真时间步长 [秒]
        substeps = 1  # 每个时间步的子步数
        gravity = [0., 0., -9.81]  # 重力加速度 [米/秒^2]
        up_axis = 1  # 上轴方向（0 为 y，1 为 z）

        class physx:
            num_threads = 10  # 物理引擎线程数
            solver_type = 1  # 求解器类型（0：PGS，1：TGS）
            num_position_iterations = 4  # 位置迭代次数
            num_velocity_iterations = 0  # 速度迭代次数
            contact_offset = 0.01  # 接触偏移 [米]
            rest_offset = 0.0  # 静止偏移 [米]
            bounce_threshold_velocity = 0.5  # 弹跳阈值速度 [米/秒]
            max_depenetration_velocity = 1.0  # 最大分离速度 [米/秒]
            max_gpu_contact_pairs = 2**18  # 最大 GPU 接触对数
            default_buffer_size_multiplier = 5  # 默认缓冲区大小倍数
            contact_collection = 2  # 接触集合模式（0：从不，1：最后子步，2：所有子步）