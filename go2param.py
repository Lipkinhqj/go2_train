def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        # PPO超参数：clip范围0.2，目标KL散度0.01，学习率0.001，熵系数0.01
        # 优化设置：5个epochs/次更新，4个minibatch，梯度裁剪1.0
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            # 从0.01调到0.02，允许有更大的策略变化
            "desired_kl": 0.01,
            "entropy_coef": 0.001,    #原来为0.01
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-4,
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
        # 每个电机会有一个连续的action
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
        "joint_init_angles": {  # [rad]
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
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "connect_plane_links":[
            "base",
            "FR_calf",   # 前右脚
            "FL_calf",   # 前左脚
            "RR_calf",   # 后右脚
            "RL_calf",   # 后左脚
        ],
        "foot_names" : ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        # PD控制参数
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 30,  # degree
        "termination_if_pitch_greater_than": 30,
        "termination_if_base_connect_plane_than": True, #触底重置
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
        "noise":{
            "use": True,
            #[高斯,随机游走]
            "ang_vel": [0.01,0.01],
            "dof_pos": [0.01,0.01],
            "dof_vel": [0.01,0.01],
            "gravity": [0.01,0.01],
        }
    }
    #奖励函数
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "desired_stance_width": 0.28,          # 目标左右脚间距 0.28 m
        "stance_sigma": 0.03,                  # σ 越小越严格
        "desired_feet_front": 0.40,
        "front_sigma": 0.01,
        "action_sym_sigma":1.0,
        # "leg_posture_sigma":0.01,
        # "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.3, #跟踪线速度命令（xy轴）
            "tracking_ang_vel": 1.0, #跟踪角速度命令（偏航）

            "lin_vel_z": -0.25,       #惩罚z轴基础线速度
            "base_height": -25.0,    #惩罚基础高度偏离目标
            "action_rate": -0.003,   #惩罚动作变化
            "similar_to_default": -0.05, #鼓励机器人姿态与默认姿态相似
            "angle_change": -0.003,     #惩罚zitai角变化
            "feet_pose":0.3,
            # "similar_to_hip": -0.003,   #鼓励机器人hip与默认相似
            # "collision": -0.0015,      #base接触地面碰撞力越大越惩罚，数值太大会摆烂
            "dof_force":-1e-5,            #惩罚每个关节力矩过大 
            "actions_symmetry":-0.005,     #鼓励机器人y走路的时候，腿部相似
            # "ycmd_sim_to_default": -0.05,
            # "MSE": 0.015,
            # "yspd_feetpos": 0.1,
            # "contact_time:-0.001,"
            # "similar_to_leg_pos": 0.3        #鼓励对角脚姿态相似
        },
    }
    # 课程学习，奖励循序渐进 待优化
    curriculum_cfg = {
        "curriculum_lin_vel_step":0.015,   #比例
        "curriculum_ang_vel_step":0.00015,   #比例
        "curriculum_lin_vel_min_range":0.3,   #比例
        "curriculum_ang_vel_min_range":0.1,   #比例
        "lin_vel_err_range":[0.25,0.45],  #课程误差阈值
        "ang_vel_err_range":[0.25,0.45],  #课程误差阈值 连续曲线>方波>不波动
        "damping_descent":False,
        "dof_damping_descent":[0.2, 0.005, 0.001, 0.4],#[damping_max,damping_min,damping_step（比例）,damping_threshold（存活步数比例）]
    }

    # 命令生成
    command_cfg = {
        "num_commands": 3,
        "modes":{
            "simplex_cmd":{
                "lin_vel_x_range": [0.55, 1.3],
                "lin_vel_y_range": [0, 0],
                "ang_vel_range": [0, 0],
            },
            "rotate_spd_cmd":{
                "lin_vel_x_range": [0, 0],
                "lin_vel_y_range": [0, 0],
                "ang_vel_range": [-1., 1.],
            },
            "low_spdx_cmd":{
                "lin_vel_x_range": [-0.5, 0.5],
                "lin_vel_y_range": [0, 0],
                "ang_vel_range": [0, 0],
            },
            "simpley_cmd":{
                "lin_vel_x_range": [0, 0],
                "lin_vel_y_range": [-0.5, 0.5],
                "ang_vel_range": [0, 0],
            },
            # "complex_cmd":{
            #     "lin_vel_x_range": [-0.5, 0.5],
            #     "lin_vel_y_range": [-0.3, 0.3],
            #     "ang_vel_range": [0, 0],
            # },
        },
        "eval_seq_cmd":[
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.7],
            [0.3, 0.0, 0.0],
        ],

        "mode_prob":[0.2, 0.3, 0.2, 0.3],
        "max_y":0.7,
        "max_x":1.3,
        "max_ang":1.0,
    }
    #域随机化 friction_ratio是范围波动 mass和com是偏移波动 等到模型存活达到70%再开启域随机化
    domain_rand_cfg = { 
        "friction_ratio_range":[0.2 , 1.6], #地面摩擦力范围
        "random_base_mass_shift_range":[-3 , 8.], #质量偏移量
        "random_other_mass_shift_range":[-0.1, 0.1],  #质量偏移量
        "random_base_com_shift":0.05, #位置偏移量 xyz
        "random_other_com_shift":0.01, #位置偏移量 xyz
        "random_KP":[0.8, 1.2], #比例
        "random_KV":[0.8, 1.2], #比例
        "random_default_joint_angles":[-0.03,0.03], #rad
        "damping_range":[0.8, 1.2], #比例
        "dof_stiffness_range":[0.0 , 0.0], #范围 不包含轮 [0.0 , 0.0]就是关闭，关闭的时候把初始值也调0
        "dof_armature_range":[0.0 , 0.008], #范围 额外惯性 类似电机减速器惯性 有助于仿真稳定性
    }
    #地形配置
    terrain_cfg = {
        "terrain":True, #是否开启地形
        "train":"agent_train_gym",
        "eval":"agent_eval_gym",    # agent_eval_gym/circular
        "respawn_points":[
            [-5.0, -5.0, 0.0],    #plane地形坐标，一定要有，为了远离其他地形
            [5.0, 5.0, 0.0],
            [15.0, 5.0, 0.08],
        ],
        "horizontal_scale":0.1,
        "vertical_scale":0.001,
        "vertical_stairs":True,
        "v_stairs_height":0.1,  # 阶梯高度
        "v_stairs_width":0.25,  # 阶梯宽度
        "v_plane_size":0.8,  # 平台尺寸
        "v_stairs_num":10       # 阶梯数量
    }


    return env_cfg, obs_cfg, reward_cfg, command_cfg, domain_rand_cfg, terrain_cfg
