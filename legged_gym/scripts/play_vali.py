from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
import numpy as np
import torch
import csv
import math
from datetime import datetime
import time

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()  # 输入
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # 打开CSV文件准备写入
    with open('fast.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # 旧动态生成列标题
        # ac_action_headers = [f'Action{i+1}' for i in range(12)]  # 12个动作
        # observation_headers = [f'Observation{i+1}' for i in range(45)]  # 45个观察值
        # all_headers = ['Step'] + action_headers + observation_headers
        # csv_writer.writerow(all_headers)  # 写入列标题

        # 动态生成列标题
        # specific_action_headers = [f'modified_cmd{i+1}' for i in range(12)]  # 12个动作命令
        dof_pos_headers = [f'motSta{i}' for i in range(12)]  # 观察值的第10到21个 对应电机关节状态
        action_headers = [f'cmdOfl{i}' for i in range(12)]  # 12个动作
        # last_action = [f'last_action{i+1}' for i in range(12)]  # 观察值的第10到21个 对应电机上次action
        all_headers = ['Step']  + dof_pos_headers + action_headers# + last_action
        # all_headers = ['Step'] + specific_action_headers + dof_pos_headers + action_headers + ac_action_headers + observation_headers# + last_action
        csv_writer.writerow(all_headers)  # 写入列标题

        logger = Logger(env.dt)
        robot_index = 0 # which robot is used for logging
        joint_index = 3 # which joint is used for logging
        stop_state_log = 100 # number of steps before plotting states
        stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
        # print("stop_rew_log", stop_rew_log)
        camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
        camera_vel = np.array([1., 1., 0.])
        camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
        img_idx = 0

        print("max_episode_length", env.max_episode_length)
        env.max_episode_length = 10 * env.max_episode_length
        t0 = get_world_time_in_seconds()
        for i in range(10*int(env.max_episode_length)):
            t1 = get_world_time_in_seconds()
            t2 = t1 -t0
            t1 = get_world_time_in_seconds()
            t2 = t1 -t0
            actions = policy(obs.detach())
            # print("actions: " , actions)
            
            # 单腿测试
            # specific_actions = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2], dtype=torch.float32, device=env.device)

        # lcl 添加特定关节动作如下
        # vali 模式下记得设置速度命令为0
            specific_actions = torch.zeros(1, 12, device=env.device)
            amplitude = 0.08 * 4  # 设置正弦波的振幅，可以根据需要调整
            # 每次循环开始时重置动作张量
            specific_actions.zero_()

            frequency = 1   # 频率 频率为一个prd周期内腿运动次数
            j = i % 1001    # 去除1001
            prd = 50       # 设定控制周期5s
            ratio = 0.05    # 设置触底比
            ret = j % prd   # 计算相位

            if (ret >= (ratio * prd) and ret <=(0.5 * prd)):
                angle = (ret - ratio * prd) * 2 * math.pi * frequency / (prd * (0.5 - ratio))
                # 应用正弦信号到特定的关节
                specific_actions[0, [2, 11]] = 0.0
                specific_actions[0, [5, 8]] = -2 * amplitude * (1 - math.cos(angle))
                specific_actions[0, [1, 10]] = 0.0
                specific_actions[0, [4, 7]] = amplitude * (1 - math.cos(angle))
            elif (ret <= (1 * prd) and ret >= (0.5 * prd + ratio * prd)):
                angle = (ret - ratio * prd - 0.5 * prd) * 2 * math.pi * frequency / (prd * (0.5 -ratio))
                # 应用正弦信号到特定的关节
                specific_actions[0, [2, 11]] = -2 * amplitude * (1 - math.cos(angle))
                specific_actions[0, [5, 8]] = 0.0
                specific_actions[0, [1, 10]] = amplitude * (1 - math.cos(angle))
                specific_actions[0, [4, 7]] = 0.0
            else:
                specific_actions[0, [2, 11]] = 0.0
                specific_actions[0, [5, 8]] = 0.0
                specific_actions[0, [1, 10]] = 0.0
                specific_actions[0, [4, 7]] = 0.0
        # lcl 添加特定关节动作结束
            if (ret >= (ratio * prd) and ret <=(0.5 * prd)):
                angle = (ret - ratio * prd) * 2 * math.pi * frequency / (prd * (0.5 - ratio))
                # 应用正弦信号到特定的关节
                specific_actions[0, [2, 11]] = 0.0
                specific_actions[0, [5, 8]] = -2 * amplitude * (1 - math.cos(angle))
                specific_actions[0, [1, 10]] = 0.0
                specific_actions[0, [4, 7]] = amplitude * (1 - math.cos(angle))
            elif (ret <= (1 * prd) and ret >= (0.5 * prd + ratio * prd)):
                angle = (ret - ratio * prd - 0.5 * prd) * 2 * math.pi * frequency / (prd * (0.5 -ratio))
                # 应用正弦信号到特定的关节
                specific_actions[0, [2, 11]] = -2 * amplitude * (1 - math.cos(angle))
                specific_actions[0, [5, 8]] = 0.0
                specific_actions[0, [1, 10]] = amplitude * (1 - math.cos(angle))
                specific_actions[0, [4, 7]] = 0.0
            else:
                specific_actions[0, [2, 11]] = 0.0
                specific_actions[0, [5, 8]] = 0.0
                specific_actions[0, [1, 10]] = 0.0
                specific_actions[0, [4, 7]] = 0.0
        # lcl 添加特定关节动作结束

            # 执行action lcl修改 pa
            obs, _, rews, dones, infos = env.step(actions.detach(), specific_actions) #, t2
            # obs, _, rews, dones, infos = env.step(actions.detach())

            # print("dones: ", env.step(actions.detach()))
            # 将actions、specific_actions和obs的指定元素转换为列表并写入CSV

        # lcl 添加打印如下  文件输出应该在这里哪里，记不太清楚了，应该解开注释就能记录，可能得看看
            # cmdofl
            offset_sequence = [0, 0.7854, 0, 0, 0.7854, 0, 0, 0.7854, 0, 0, 0.7854, 0]
            modified_actions = actions[0].detach().cpu().numpy() * 0.25 
            modified_actions += np.array(offset_sequence)

            # 开环指令cmd
            modified_cmd = specific_actions[0].detach().cpu().numpy() * 0.25
            modified_cmd += np.array(offset_sequence)

            # 电机state
            modified_state = obs[0, 9:21].detach().cpu().numpy()
            modified_state += np.array(offset_sequence)

            # lastact = obs[0, 33:45].detach().cpu().numpy()

            # 将actions和observations转换为列表并写入CSV
            row_data = [t2] + modified_state.tolist() + modified_actions.tolist()
            # row_data = [t2] + modified_cmd.tolist() + modified_state.tolist() + modified_actions.tolist() + actions[0].detach().cpu().numpy().tolist() + obs[0].detach().cpu().numpy().tolist() # + lastact.tolist()
            # row_data = [t2] + actions[0].detach().cpu().numpy().tolist() + obs[0].detach().cpu().numpy().tolist()
            
            csv_writer.writerow(row_data)
        # lcl添加打印结束

            if RECORD_FRAMES:
                if i % 2:
                    filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    img_idx += 1 
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

            if i < stop_state_log:
                logger.log_states(
                    {
                        'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                        'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                        'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                        'dof_torque': env.torques[robot_index, joint_index].item(),
                        'command_x': env.commands[robot_index, 0].item(),
                        'command_y': env.commands[robot_index, 1].item(),
                        'command_yaw': env.commands[robot_index, 2].item(),
                        'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                        'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                        'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                        'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                        'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                    }
                )
            elif i==stop_state_log:
                logger.plot_states()
            if  0 < i < stop_rew_log:
                if infos["episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes>0:
                        logger.log_rewards(infos["episode"], num_episodes)
            elif i==stop_rew_log:
                logger.print_rewards()

def get_world_time_in_seconds():
    # 获取当前UTC时间
    utc_now = datetime.utcnow()
    # 将UTC时间转换为Unix时间戳（秒）
    unix_timestamp = int((utc_now.timestamp() * 1000))
    return unix_timestamp
def get_world_time_in_seconds():
    # 获取当前UTC时间
    utc_now = datetime.utcnow()
    # 将UTC时间转换为Unix时间戳（秒）
    unix_timestamp = int((utc_now.timestamp() * 1000))
    return unix_timestamp

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)