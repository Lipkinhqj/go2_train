import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import go2param as gp

# 用于生成随机float数
def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


# 将环境数量、模型映射、观测空间、奖励函数、是否可视传入
class go2EnvCreate:
    # 初始化环境信息等
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        # 观测空间的维数是45，就是输入是45
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.force = dict()

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        self.init_euler_flag = 0

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )    

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # 添加机器人实体
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                merge_fixed_links = False,
            ),
        )

        # 创建4096个仿真环境
        self.scene.build(n_envs=num_envs)    

        # 映射电机和配置文件中的关节
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # PD control parameters,配置仿真环境每个机器人的kpd
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions = dict()
        self.episode_sums = dict()
        
        #计算实际周期内的奖励
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # 初始化每个环境的速度角速度，都为0
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        # 初始化观测空间，每个环境有45个观测空间
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        # 初始化指令，每个四足有3个指令，即x、y速度，角速度
        # 缩放系数后续用于归一化
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        # 初始化action，即12个关节
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        # 初始每个机器人只有一个三维的初始位置
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        # 姿态角
        self.last_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        # self.init_base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )

        # 接触力
        self.connect_force = torch.zeros((self.num_envs,self.robot.n_links, 3), device=self.device, dtype=gs.tc_float)

        #足端位置
        self.FL_foot = self.robot.get_link(self.env_cfg["foot_names"][0])
        self.FR_foot = self.robot.get_link(self.env_cfg["foot_names"][1])
        self.RL_foot = self.robot.get_link(self.env_cfg["foot_names"][2])
        self.RR_foot = self.robot.get_link(self.env_cfg["foot_names"][3])
        self.FL_foot_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.FR_foot_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.RL_foot_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.RR_foot_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.FL_foot_pos[:] = self.FL_foot.get_pos()
        self.FR_foot_pos[:] = self.FR_foot.get_pos()
        self.RL_foot_pos[:] = self.RL_foot.get_pos()   
        self.RR_foot_pos[:] = self.RR_foot.get_pos()  
        self.FL_foot_base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.FR_foot_base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.RL_foot_base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.RR_foot_base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    # 随机生成指令信息
    def _resample_commands(self, envs_idx):
        complex_prob = self.command_cfg["complex_prob"]
        use_complex = torch.rand(len(envs_idx), device=self.device) < complex_prob

            # ---- 先采样两组指令 ----
        cmd_simple  = torch.stack([
            gs_rand_float(*self.command_cfg["simplex_cmd"]["lin_vel_x_range"],  (len(envs_idx),), self.device),
            gs_rand_float(*self.command_cfg["simplex_cmd"]["lin_vel_y_range"],  (len(envs_idx),), self.device),
            gs_rand_float(*self.command_cfg["simplex_cmd"]["ang_vel_range"],    (len(envs_idx),), self.device),
        ], dim=-1)

        cmd_complex = torch.stack([
            gs_rand_float(*self.command_cfg["simpley_cmd"]["lin_vel_x_range"], (len(envs_idx),), self.device),
            gs_rand_float(*self.command_cfg["simpley_cmd"]["lin_vel_y_range"], (len(envs_idx),), self.device),
            gs_rand_float(*self.command_cfg["simpley_cmd"]["ang_vel_range"],   (len(envs_idx),), self.device),
        ], dim=-1)

        # ---- 根据掷硬币结果选指令 ----
        self.commands[envs_idx] = torch.where(
            use_complex.unsqueeze(-1),        # (N,1) 做 broadcast
            cmd_complex,
            cmd_simple,
        )
    #每一步的与环境交互
    def step(self, actions):
        # 剪裁动作，将他限制在一定范围
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        # 考虑真实机器人控制信号传递延时，如果等于true就使用上一次动作
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        # 计算目标关节的位置
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        # 执行位置控制
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        # 推进仿真世界一帧
        self.scene.step()

        # 位置、四元素储存加一
        self.episode_length_buf += 1
        # 获取机器人在世界系下的位置
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            # rpy=True,
            # degrees=True,
        )

        if self.init_euler_flag == 0:
            self.init_euler = self.base_euler
            self.init_euler_flag = 1

        # 计算机器人的反向四元数
        inv_base_quat = inv_quat(self.base_quat)
        # 获取机器人系的线速度、角速度、中立投影、电机关节位置、速度
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        #足端位置
        self.FL_foot_pos[:] = self.FL_foot.get_pos()
        self.FR_foot_pos[:] = self.FR_foot.get_pos()
        self.RL_foot_pos[:] = self.RL_foot.get_pos()
        self.RR_foot_pos[:] = self.RR_foot.get_pos()
        self.FL_foot_base_pos[:] = transform_by_quat(self.FL_foot_pos, inv_base_quat) + self.base_pos
        self.FR_foot_base_pos[:] = transform_by_quat(self.FR_foot_pos, inv_base_quat) + self.base_pos
        self.RL_foot_base_pos[:] = transform_by_quat(self.RL_foot_pos, inv_base_quat) + self.base_pos
        self.RR_foot_base_pos[:] = transform_by_quat(self.RR_foot_pos, inv_base_quat) + self.base_pos

        # resample commands
        # 每过4秒重新生成控制指令，目前只有x、y方向行走
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        # 当达到最大步数、时间到、或者姿态不稳定，就重置
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())     

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            # 计算当前步所有rew
            self.rew_buf += rew
            self.episode_sums[name] += rew  

        # compute observations
        # 观测量拼接
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 角速度3维
                self.projected_gravity,  # 投影后的重力方向3维
                self.commands * self.commands_scale,  # 当前目标指令3维
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 关节位置，相对于默认12维
                self.dof_vel * self.obs_scales["dof_vel"],  # 关节速度12维
                self.actions,  # 上一个动作12维
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        # 返回给agent
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # def _apply_random_push(self, force_range, pos, env_ids):

    #     for i in env_ids:
    #         directions = torch.randint(1, 5, size=(1,)).item()
    #         if directions == 1:
    #             force_dir = pos[0].cpu() - 0.2
    #         elif directions == 2:
    #             force_dir = pos[0].cpu() + 0.2
    #         elif directions == 3:
    #             force_dir = pos[1].cpu() - 0.2
    #         elif directions == 4:
    #             force_dir = pos[1].cpu() + 0.2

    #         force = torch.rand(1).item() * (force_range[1] - force_range[0]) + force_range[0]
    #         force_field = gs.engine.force_fields.Point(force, force_dir, 12, 0)
    #         force_field.activate()

            # if not hasattr(self, 'force_fields '):
            #     self.force_fields = {}
            # self.force_fields[i] = {'field': force_field, 'timer': 5}

    # 
    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_feet_pose(self):
        
        # 惩罚左右脚在机体坐标系下的横向间距偏离目标值
        # - 用 (y_left_avg, y_right_avg) 计算实际间距
        # - 奖励形式:  exp(- (error²) / σ² )
        # 读取机体系 y 坐标
        y_FL = self.FL_foot_base_pos[:, 1]   # (N_env,)
        y_FR = self.FR_foot_base_pos[:, 1]
        y_RL = self.RL_foot_base_pos[:, 1]
        y_RR = self.RR_foot_base_pos[:, 1]

        # 左右侧平均，使奖励更平滑
        y_left  = 0.5 * (y_FL + y_RL)        # 左脚平均 y (>0)
        y_right = 0.5 * (y_FR + y_RR)        # 右脚平均 y (<0)

        vy_abs = torch.abs(self.commands[:, 1])        # 取 vy
        alpha_y  = 1.0 - (vy_abs / self.command_cfg["max_y"]).clamp(max=1.0)

        # 实际间距（绝对差）
        stance_width = torch.abs(y_left - y_right)   # (N_env,)
        # 误差及奖励
        target_y = self.reward_cfg["desired_stance_width"]
        sigma_y  = self.reward_cfg["stance_sigma"]
        error_y  = torch.square(stance_width - target_y)     # 二次误差
        reward_y = torch.exp(-error_y / (2 * sigma_y * sigma_y)) # 高斯式映射到 (0,1]


        x_FL = self.FL_foot_base_pos[:, 0]   # (N_env,)
        x_FR = self.FR_foot_base_pos[:, 0]
        x_RL = self.RL_foot_base_pos[:, 0]
        x_RR = self.RR_foot_base_pos[:, 0]

        # x_front = 0.5 * (x_FL + x_FR)
        # x_rear = 0.5 * (x_RL + x_RR)
        x_front_dif = x_FL - x_RL
        x_rear_dif = x_FR - x_RR

        vx_abs = torch.abs(self.commands[:, 0])        # 取 vx
        alpha_x  = 1.0 - (vx_abs / self.command_cfg["max_x"]).clamp(max=1.0)

        # 实际间距（绝对差）
        # vx_gap = torch.abs(x_front - x_rear)   # (N_env,)
        # 误差及奖励
        target_x = self.reward_cfg["desired_feet_front"]
        sigma_x  = self.reward_cfg["front_sigma"]
        error_x  = torch.square(x_front_dif - target_x) + torch.square(x_rear_dif - target_x)     # 二次误差
        reward_x = torch.exp(-error_x / (2 * sigma_x * sigma_x)) # 高斯式映射到 (0,1]

        return reward_y * alpha_y + reward_x * alpha_x        

    # def _reward_similar_to_default(self):
    #     # Penalize joint poses far away from default pose
    #     return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_similar_foot_Zpos(self):
        
        z_FL = self.FL_foot_base_pos[:, 2]   # (N_env,)
        z_FR = self.FR_foot_base_pos[:, 2]
        z_RL = self.RL_foot_base_pos[:, 2]
        z_RR = self.RR_foot_base_pos[:, 2]

        dif_13 = torch.abs(z_FL - z_RR)
        dif_24 = torch.abs(z_FR - z_RL)

        sigma_z = self.reward_cfg["poseZ_sigma"]
        reward_z = torch.exp(- (dif_13 + dif_24) / (2 * sigma_z * sigma_z))
        return reward_z


    def _reward_similar_to_hip(self):
        hip_idx = [0, 3, 6, 9]
        dif = torch.abs(self.dof_pos[:, hip_idx] - self.default_dof_pos[hip_idx])
        return dif.sum(dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
    
    def _reward_angle_change(self):
        # 惩罚角度变化过大
        return torch.square(self.base_euler[:, 0] - self.init_euler[:, 0]) + torch.square(self.base_euler[:, 1] - self.init_euler[:, 1])
            