from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class ElflxspineRoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2048
        num_actions = 14

    class terrain(LeggedRobotCfg.terrain):
        mesg_type = 'trimesh'
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.34]  # 初始位置，z轴高度设置为0.34米，可以根据实际情况调整
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'A0_spine_PitchS':0.0,

            'AL_hip_Roll': 0.0,
            'AL_thigh_PitchT': 0.7854,
            'AL_calf_PitchC': 0.0,

            'AR_hip_Roll': 0.0,
            'AR_thigh_PitchT': 0.7854,
            'AR_calf_PitchC': 0.0,

            'B0_spine_PitchS': 0.0,

            'BL_hip_Roll': 0.0,
            'BL_thigh_PitchT': 0.7854,
            'BL_calf_PitchC': 0.0, 
            
            'BR_hip_Roll': 0.0,                       
            'BR_thigh_PitchT': 0.7854,         
            'BR_calf_PitchC': 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {
                'A0_spine_PitchS': 48,
                'B0_spine_PitchS': 48,
                'hip': 48,
                'thigh': 48,
                'calf': 48, 
                # 'hip': 80.1,
                # 'thigh': 80.1,
                # 'calf': 80.1,
            }
        damping = {
                'A0_spine_PitchS': 1.2,
                'B0_spine_PitchS': 1.2,
                'hip': 1.2,
                'thigh': 1.2,
                'calf': 1.2,
                # 'hip': 5.0,
                # 'thigh': 5.0,
                # 'calf': 5.0,
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
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/elflxspine/meshes/elflxspine.urdf'  # URDF文件的路径
        name = "elflxspine"
        foot_name = "foot"  # 脚部链接的名称，用于计算奖励和终止条件 
        penalize_contacts_on = ["hip", "thigh", "calf"]  # 接触惩罚的链接名称
        terminate_after_contacts_on = ["trunk", "spine", "hip", "thigh"]  # 接触后终止仿真的链接名称
        self_collisions = 1  # 启用自我碰撞检测 1为不启用 0为启用
        max_angular_velocity = 15. # by default set to 1000. in legged_robot_config.py
        max_linear_velocity = 15.
        flip_visual_attachments = False
        
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-1,2.5]
        

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.3
        max_contact_force = 300.

        class scales( LeggedRobotCfg.rewards.scales ):
            pass
            
class ElflxspineRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_elflxspine'
        # load_run = -1