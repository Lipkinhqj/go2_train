def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        # PPO超参数：clip范围0.2，目标KL散度0.01，学习率0.001，熵系数0.01
        # 优化设置：5个epochs/次更新，4个minibatch，梯度裁剪1.0
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive", 
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        # 网络结构：Actor-Critic均使用[512,256,128]隐藏层
        # 激活函数：ELU（指数线性单元）
        # 初始动作噪声：1.0（鼓励探索）
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        # 最大迭代次数100（可命令行覆盖）
        # 每环境24步收集一次数据
        # 每100次迭代保存模型
        # 实验名称和日志路径管理
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": True,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": True,
        "seed": 1,
    }

    return train_cfg_dict

#环境配置函数
def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        # 将模型文件中定义的关节映射到模拟器内部的自由度索引
        # 关节命名通常遵循〈腿的位置〉_〈关节名称〉
        # F Front   R Rear      L Left      R Right
        # Hip joint （髋关节） → 负责腿根部水平摆动，类似“左右张合 / 前后摆”
        # Thigh joint（大腿关节）→ 负责上下抬腿、前后摆动（主要决定步幅）
        # Calf joint（小腿关节）→ 负责伸缩小腿（蹬地、落脚高度）
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "feet_link_names": ["foot"],
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "foot_names" : ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        # PD控制参数
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 20,  # degree
        "termination_if_pitch_greater_than": 20,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        # 每过四秒重新生成指令
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        # # 生成随机时间的力
        # "random_push":{
        #     "enabled":True,
        #     "interval_range":[150, 400],   #步长0.02，所以3-8秒推一次
        #     "force_range":[5.0, 10.0],  
        # }
        # domain randomization
        'randomize_friction': True,
        'friction_range': [0.2, 1.5],
        # 'randomize_base_mass': True,
        # 'added_mass_range': [-1., 3.],
        # 'randomize_com_displacement': True,
        # 'com_displacement_range': [-0.01, 0.01],
        # 'randomize_motor_strength': False,
        # 'motor_strength_range': [0.9, 1.1],
        # 'randomize_motor_offset': True,
        # 'motor_offset_range': [-0.02, 0.02],
        # 'randomize_kp_scale': True,
        # 'kp_scale_range': [0.8, 1.2],
        # 'randomize_kd_scale': True,
        # 'kd_scale_range': [0.8, 1.2],
    }
    #观测空间
    # 线速度（缩放2x）
    # 角速度（缩放0.25x）
    # 关节位置/速度（各缩放1x/0.05x）
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    #奖励函数
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "desired_stance_width": 0.28,          # 目标左右脚间距 0.28 m
        "stance_sigma": 0.03,                  # σ 越小越严格
        "desired_feet_front": 0.38,
        "front_sigma": 0.01,
        "poseZ_sigma":0.01,
        # "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.3, #跟踪线速度命令（xy轴）
            "tracking_ang_vel": 0.5, #跟踪角速度命令（偏航）

            "lin_vel_z": -0.25,       #惩罚z轴基础线速度
            "base_height": -25.0,    #惩罚基础高度偏离目标
            "action_rate": -0.003,   #惩罚动作变化
            # "similar_to_default": -0.01, #鼓励机器人姿态与默认姿态相似
            "angle_change": -0.003,     #惩罚zitai角变化
            "feet_pose":0.3,
            #"feet_stance_width": 0.3,       #鼓励机器人直行脚步宽度正常
            #"feet_front_back": 0.3,         #鼓励机器人横向走的时候脚宽度正常
            "similar_to_hip": -0.003,   #鼓励机器人hip与默认相似
            "similar_foot_Zpos": 0.5        #鼓励对角脚姿态相似
        },
    }
    # 命令生成
    command_cfg = {
        "num_commands": 3,
        "simplex_cmd":{
            "lin_vel_x_range": [-1.3, 1.3],
            "lin_vel_y_range": [0, 0],
            "ang_vel_range": [0, 0],
        },
        "simpley_cmd":{
            "lin_vel_x_range": [0, 0],
            "lin_vel_y_range": [-1., 1.],
            "ang_vel_range": [0, 0],
        },
        "complex_cmd":{
            "lin_vel_x_range": [-0.5, 0.5],
            "lin_vel_y_range": [-0.3, 0.3],
            "ang_vel_range": [0, 0],
        },
        "complex_prob": 0.5,
        "max_y":1,
        "max_x":1.3,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg
