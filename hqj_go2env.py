import torch
import math
import genesis as gs
import numpy as np
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import go2param as gp

# 用于生成随机float数
def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


# 将环境数量、模型映射、观测空间、奖励函数、是否可视传入
class go2EnvCreate:
    # 初始化环境信息等
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, 
                  command_cfg, domain_rand_cfg, terrain_cfg, 
                 show_viewer=False, train_mode=True):
        self.num_envs = num_envs
        # 观测空间的维数是45，就是输入是45
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.eval_cmds = command_cfg["eval_seq_cmd"]
        self.device = gs.device
        self.mode = train_mode
        self.eval_cmd_num = 0
        self.domain_rand_cfg = domain_rand_cfg
        self.terrain_cfg = terrain_cfg

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
        self.noise = obs_cfg["noise"]

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
        self.joint_dof_idx    = self.motors_dof_idx.copy()
        self.joint_dof_idx_np = np.arange(len(self.joint_dof_idx))

        # PD control parameters,配置仿真环境每个机器人的kpd
        self.kp = self.env_cfg["kp"]  
        self.kv = self.env_cfg["kd"]
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions = dict()
        self.episode_sums = dict()

        # 存活比例
        self.survive_ratio = 0.0
        
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
        self.dof_force = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        # 初始每个机器人只有一个三维的初始位置
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        # 姿态角
        self.last_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        # self.init_base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        # 机体默认初始位置，这个值不变
        self.basic_default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )        
        
        default_dof_pos_list = [[self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]]] * self.num_envs
        self.default_dof_pos = torch.tensor(default_dof_pos_list,device=self.device,dtype=gs.tc_float,)
        init_dof_pos_list = [[self.env_cfg["joint_init_angles"][name] for name in self.env_cfg["joint_names"]]] * self.num_envs
        self.init_dof_pos = torch.tensor(init_dof_pos_list,device=self.device,dtype=gs.tc_float,)

        # self.default_dof_pos = torch.tensor(
        #     [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
        #     device=gs.device,
        #     dtype=gs.tc_float,
        # )

        # 接触力
        self.connect_force = torch.zeros((self.num_envs,self.robot.n_links, 3), device=self.device, dtype=gs.tc_float)
        self.robot_mass = self.robot.get_mass()
        self.extras = dict()  # extra information for logging

        # 跪地重启   注意是idx_local不需要减去base_idx
        if(self.env_cfg["termination_if_base_connect_plane_than"]&self.mode):
            self.reset_links = [(self.robot.get_link(name).idx_local) for name in self.env_cfg["connect_plane_links"]]

        # 足端id用于检测碰地长度
        self.foot_link_ids = torch.as_tensor(
            [self.robot.get_link(n).idx 
             for n in self.env_cfg["foot_names"]],
            device=self.device,
            dtype=torch.long
        )  # shape (N_foot,)

        # 足端位置
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

        self.FL_foot_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.FR_foot_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.RL_foot_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.RR_foot_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.FL_foot_vel[:] = self.FL_foot.get_vel()
        self.FR_foot_vel[:] = self.FR_foot.get_vel()
        self.RL_foot_vel[:] = self.RL_foot.get_vel()   
        self.RR_foot_vel[:] = self.RR_foot.get_vel()  
        self.FL_foot_base_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.FR_foot_base_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.RL_foot_base_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.RR_foot_base_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        #域随机化 domain_rand_cfg
        # 地面摩擦力范围
        self.friction_ratio_low = self.domain_rand_cfg["friction_ratio_range"][0]
        self.friction_ratio_range = self.domain_rand_cfg["friction_ratio_range"][1] - self.friction_ratio_low
        # 机体base质量偏移量，大幅度
        self.base_mass_low = self.domain_rand_cfg["random_base_mass_shift_range"][0]
        self.base_mass_range = self.domain_rand_cfg["random_base_mass_shift_range"][1] - self.base_mass_low  
        # 躯干以外的质量偏移量，小幅度
        self.other_mass_low = self.domain_rand_cfg["random_other_mass_shift_range"][0]
        self.other_mass_range = self.domain_rand_cfg["random_other_mass_shift_range"][1] - self.other_mass_low            
        # 关节阻尼系数比例随机化
        self.dof_damping_low = self.domain_rand_cfg["damping_range"][0]
        self.dof_damping_range = self.domain_rand_cfg["damping_range"][1] - self.dof_damping_low
        # 电机转动惯量随机化
        self.dof_armature_low = self.domain_rand_cfg["dof_armature_range"][0]
        self.dof_armature_range = self.domain_rand_cfg["dof_armature_range"][1] - self.dof_armature_low
        # 关节PD控制器比例缩放
        self.kp_low = self.domain_rand_cfg["random_KP"][0]
        self.kp_range = self.domain_rand_cfg["random_KP"][1] - self.kp_low
        self.kv_low = self.domain_rand_cfg["random_KV"][0]
        self.kv_range = self.domain_rand_cfg["random_KV"][1] - self.kv_low
        # 默认关节零位偏移
        self.joint_angle_low = self.domain_rand_cfg["random_default_joint_angles"][0]
        self.joint_angle_range = self.domain_rand_cfg["random_default_joint_angles"][1] - self.joint_angle_low
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    # 随机生成指令信息
    def _resample_commands(self, envs_idx):

        # if not self.mode:
        #     seq = torch.as_tensor(self.eval_cmds,
        #         device=self.device, dtype=torch.float32)  # (M,3
        #     M = seq.shape[0]
        #     idx = self.eval_cmd_num % M
            
        #     self.commands[envs_idx] = seq[idx, :].repeat(len(envs_idx), 1)
        #     self.eval_cmd_num += 1
        if not self.mode:
            num_envs   = len(envs_idx)
            if num_envs == 0:
                return
            seq = torch.as_tensor(self.eval_cmds,
                device=self.device, dtype=torch.float32)  # (M,3
            
            self.commands[envs_idx] = seq[self.eval_cmd_num, :].repeat(len(envs_idx), 1)
            
            self.eval_cmd_num += 1
            if self.eval_cmd_num > 3:
                self.eval_cmd_num = 0
            return
        
        num_envs   = len(envs_idx)
        if num_envs == 0:
            return
        mode_names = list(self.command_cfg["modes"].keys())        # ['forward', 'sideways', ...]

        # ---------- ① 抽取每个 env 的模式编号 ----------
        probs      = torch.tensor(self.command_cfg["mode_prob"], device=self.device)
        mode_idxs  = torch.multinomial(probs, num_envs, replacement=True)   # (N,)

        # ---------- ② 为所有模式各自生成一批指令 ----------
        cmd_tensors = []
        for m in mode_names:
            cfg = self.command_cfg["modes"][m]
            cmd = torch.stack([
                gs_rand_float(*cfg["lin_vel_x_range"], (num_envs,), self.device),
                gs_rand_float(*cfg["lin_vel_y_range"], (num_envs,), self.device),
                gs_rand_float(*cfg["ang_vel_range"]   , (num_envs,), self.device),
            ], dim=-1)                                             # (N, 3)
            cmd_tensors.append(cmd)

        # shape: (N, num_modes, 3)
        all_cmds = torch.stack(cmd_tensors, dim=1)

        # ---------- ③ 根据 mode_idxs 选出每个 env 的指令 ----------
        # 把 mode_idxs 扩成 (N,1,1) 便于 gather
        idx = mode_idxs.view(num_envs, 1, 1).expand(-1, 1, 3)
        selected_cmds = torch.gather(all_cmds, 1, idx).squeeze(1)  # (N,3)

        # ---------- ④ 写回 ----------
        self.commands[envs_idx] = selected_cmds

    #每一步的与环境交互
    def step(self, actions):
        # 剪裁动作，将他限制在一定范围
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        # 考虑真实机器人控制信号传递延时，如果等于true就使用上一次动作
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        # 计算目标关节的位置
        target_dof_pos = exec_actions[:,self.joint_dof_idx_np] * self.env_cfg["action_scale"] + self.default_dof_pos[:,self.joint_dof_idx_np]
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
        self.dof_force[:] = self.robot.get_dofs_force(self.motors_dof_idx)

        #足端位置
        self.FL_foot_pos[:] = self.FL_foot.get_pos()
        self.FR_foot_pos[:] = self.FR_foot.get_pos()
        self.RL_foot_pos[:] = self.RL_foot.get_pos()
        self.RR_foot_pos[:] = self.RR_foot.get_pos()
        self.FL_foot_base_pos[:] = transform_by_quat(self.FL_foot_pos, inv_base_quat) + self.base_pos
        self.FR_foot_base_pos[:] = transform_by_quat(self.FR_foot_pos, inv_base_quat) + self.base_pos
        self.RL_foot_base_pos[:] = transform_by_quat(self.RL_foot_pos, inv_base_quat) + self.base_pos
        self.RR_foot_base_pos[:] = transform_by_quat(self.RR_foot_pos, inv_base_quat) + self.base_pos

        self.FL_foot_vel[:] = self.FL_foot.get_vel()
        self.FR_foot_vel[:] = self.FR_foot.get_vel()
        self.RL_foot_vel[:] = self.RL_foot.get_vel()
        self.RR_foot_vel[:] = self.RR_foot.get_vel()
        self.FL_foot_base_vel[:] = transform_by_quat(self.FL_foot_vel, inv_base_quat)
        self.FR_foot_base_vel[:] = transform_by_quat(self.FR_foot_vel, inv_base_quat)
        self.RL_foot_base_vel[:] = transform_by_quat(self.RL_foot_vel, inv_base_quat)
        self.RR_foot_base_vel[:] = transform_by_quat(self.RR_foot_vel, inv_base_quat)

        # 给传感器添加上高斯噪声、和随机噪声
        if self.noise["use"]:
            self.base_ang_vel[:] += torch.randn_like(self.base_ang_vel) * self.noise["ang_vel"][0] + (torch.rand_like(self.base_ang_vel)*2-1) * self.noise["ang_vel"][1]
            self.projected_gravity += torch.randn_like(self.projected_gravity) * self.noise["gravity"][0] + (torch.rand_like(self.projected_gravity)*2-1) * self.noise["gravity"][1]
            self.dof_pos[:] += torch.randn_like(self.dof_pos) * self.noise["dof_pos"][0] + (torch.rand_like(self.dof_pos)*2-1) * self.noise["dof_pos"][1]
            self.dof_vel[:] += torch.randn_like(self.dof_vel) * self.noise["dof_vel"][0] + (torch.rand_like(self.dof_vel)*2-1) * self.noise["dof_vel"][1]
 
        #碰撞力
        self.connect_force = self.robot.get_links_net_contact_force()
        self.foot_connect_force_z = self.connect_force[:, self.foot_link_ids, 2]
        self._update_contact_mask()

        #步数
        self.episode_lengths += 1

        # resample commands
        # 每过4秒重新生成控制指令，目前只有x、y方向行走
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )

        if(self.mode):
            self.check_termination()

        # 时间过长就重置
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        if(self.mode):
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())     

        # 计算reward
        if(self.mode):
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
                (self.dof_pos[:,self.joint_dof_idx_np] - self.default_dof_pos[:,self.joint_dof_idx_np]) * self.obs_scales["dof_pos"],  # 关节位置，相对于默认12维
                self.dof_vel * self.obs_scales["dof_vel"],  # 关节速度12维
                self.actions,  # 上一个动作12维
            ],
            axis=-1,
        )
     
        self._resample_commands(envs_idx)

        if not self.mode :
            self.eval_debug()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        # 返回给agent
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def check_termination(self):
        # check termination and reset
        # 当达到最大步数、时间到、或者姿态不稳定，就重置
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        if(self.env_cfg["termination_if_base_connect_plane_than"]):
            for idx in self.reset_links:
                self.reset_buf |= torch.abs(self.connect_force[:,idx,:]).sum(dim=1) > 0

    # 域随机
    def domain_rand(self, envs_idx):
        # 设置不同摩擦
        friction_ratio = self.friction_ratio_low + self.friction_ratio_range * torch.rand(len(envs_idx), self.robot.n_links)
        self.robot.set_friction_ratio(friction_ratio=friction_ratio,
                                      link_indices=np.arange(0, self.robot.n_links),
                                      envs_idx = envs_idx)
        
        # 机体Base和各个link的质量增加或减少
        base_mass_shift = self.base_mass_low + self.base_mass_range * torch.rand(len(envs_idx), 1, device=self.device)
        other_mass_shift =-self.other_mass_low + self.other_mass_range * torch.rand(len(envs_idx), self.robot.n_links - 1, device=self.device)
        mass_shift = torch.cat((base_mass_shift, other_mass_shift), dim=1)
        self.robot.set_mass_shift(mass_shift=mass_shift,
                                  link_indices=np.arange(0, self.robot.n_links),
                                  envs_idx = envs_idx)

        # 质心坐标偏移
        base_com_shift = -self.domain_rand_cfg["random_base_com_shift"] / 2 + self.domain_rand_cfg["random_base_com_shift"] * torch.rand(len(envs_idx), 1, 3, device=self.device)
        other_com_shift = -self.domain_rand_cfg["random_other_com_shift"] / 2 + self.domain_rand_cfg["random_other_com_shift"] * torch.rand(len(envs_idx), self.robot.n_links - 1, 3, device=self.device)
        com_shift = torch.cat((base_com_shift, other_com_shift), dim=1)
        self.robot.set_COM_shift(com_shift=com_shift,
                                 link_indices=np.arange(0, self.robot.n_links),
                                 envs_idx = envs_idx)

        # 随机关节阻尼
        # kp_shift = (self.kp_low + self.kp_range * torch.rand(len(envs_idx), self.num_actions, device=self.device)) * self.kp
        # self.robot.set_dofs_kp(kp_shift, self.motors_dof_idx, envs_idx=envs_idx)

        # kv_shift = (self.kv_low + self.kv_range * torch.rand(len(envs_idx), self.num_actions, device=self.device)) * self.kv
        # self.robot.set_dofs_kv(kv_shift, self.motors_dof_idx, envs_idx = envs_idx)

        #随机初始关节位置
        dof_pos_shift = self.joint_angle_low + self.joint_angle_range * torch.rand(len(envs_idx),self.num_actions,device=self.device,dtype=gs.tc_float)
        self.default_dof_pos[envs_idx] = dof_pos_shift + self.basic_default_dof_pos

        # #damping下降，如果存活时间长就减少阻尼，短的话就增加阻尼
        # if self.is_damping_descent:
        #     if self.episode_lengths[envs_idx].mean()/(self.env_cfg["episode_length_s"]/self.dt) > self.damping_threshold:
        #         self.damping_base -= self.damping_step
        #         if self.damping_base < self.damping_min:
        #             self.damping_base = self.damping_min
        #     else:
        #         self.damping_base += self.damping_step
        #         if self.damping_base > self.damping_max:
        #             self.damping_base = self.damping_max      
        # damping = (self.dof_damping_low+self.dof_damping_range * torch.rand(len(envs_idx), self.robot.n_dofs)) * self.damping_base
        # damping[:,:6] = 0
        # self.robot.set_dofs_damping(damping=damping, 
        #                            dofs_idx_local=np.arange(0, self.robot.n_dofs), 
        #                            envs_idx=envs_idx)

        # 给各个关节额外的转动惯量
        # armature = (self.dof_armature_low+self.dof_armature_range * torch.rand(len(envs_idx), self.robot.n_dofs))
        # armature[:,:6] = 0
        # self.robot.set_dofs_armature(armature=armature, 
        #                            dofs_idx_local=np.arange(0, self.robot.n_dofs), 
        #                            envs_idx=envs_idx)

    def _update_contact_mask(self):
        # 1) 获取上一帧全部接触信息（本机器人相关）
        contact_info = self.robot.get_contacts()             # dict, shape (N_env, N_ctc, …)

        # 2) 提取 link 索引并搬到同一 device
        link_a = torch.as_tensor(contact_info["link_a"], device=self.device)   # (N_env, N_ctc)
        link_b = torch.as_tensor(contact_info["link_b"], device=self.device)

        # 3) 判断每个触点的 A/B link 是否属于足尖
        foot_ids = self.foot_link_ids                    # (N_foot,)
        mask_a   = (link_a.unsqueeze(-1) == foot_ids)    # (N_env, N_ctc, N_foot)
        mask_b   = (link_b.unsqueeze(-1) == foot_ids)

        # 4) 对“触点数”维度做 any() → 得到 (N_env, N_foot) 布尔矩阵
        contact_mask = (mask_a | mask_b).any(dim=1)      # True = 本帧有接触

        # 5) 并行场景若有 valid_mask，过滤填充行（可选）
        # 把真实接触赛选出来
        if "valid_mask" in contact_info:
            valid = torch.as_tensor(contact_info["valid_mask"], device=self.device)  # (N_env, N_ctc)
            contact_mask &= valid.any(dim=1, keepdim=True)

        # 6) 存成员，后续奖励/观测直接用
        self.contact_mask = contact_mask            # shape (N_env, N_foot)

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.survive_ratio = self.episode_length_buf[envs_idx].float().mean() / self.max_episode_length
        # reset dofs
        self.dof_pos[envs_idx] = self.init_dof_pos[envs_idx]
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
        if self.mode :
            if self.survive_ratio > 0.7:
                self.domain_rand(envs_idx)
        #步数
        self.episode_lengths[envs_idx] = 0.0

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
        alpha_y  = (vy_abs / self.command_cfg["max_y"]).clamp(max=1.0)
        # alpha_y  = 1.0 - (vy_abs / self.command_cfg["max_y"]).clamp(max=1.0)

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
        alpha_x  = (vx_abs / self.command_cfg["max_x"]).clamp(max=1.0)
        # alpha_x  = 1.0 - (vx_abs / self.command_cfg["max_x"]).clamp(max=1.0)

        # 实际间距（绝对差）
        # vx_gap = torch.abs(x_front - x_rear)   # (N_env,)
        # 误差及奖励
        target_x = self.reward_cfg["desired_feet_front"]
        sigma_x  = self.reward_cfg["front_sigma"]
        error_x  = torch.square(x_front_dif - target_x) + torch.square(x_rear_dif - target_x)     # 二次误差
        reward_x = torch.exp(-error_x / (2 * sigma_x * sigma_x)) # 高斯式映射到 (0,1]

        # 改成根据速度给
        return reward_y * alpha_x + reward_x * alpha_y   

    def _reward_actions_symmetry(self):
        #鼓励机器人更加具有动作对称性    
        #     # "joint_names": [
        #     #     "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        #     #     "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        #     #     "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        #     #     "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        #     # ],)
        sigma = self.reward_cfg["action_sym_sigma"]
        vy_abs = torch.abs(self.commands[:, 1])        # 取 vy
        alpha_y  = (vy_abs / self.command_cfg["max_y"]).clamp(max=1.0)
        actions_diff = torch.square(self.actions[:, 0] + self.actions[:, 3])
        actions_diff += torch.square(self.actions[:, 1:3] - self.actions[:, 4:6]).sum(dim=-1)
        actions_diff += torch.square(self.actions[:, 6] + self.actions[:, 9])
        actions_diff += torch.square(self.actions[:, 7:9] - self.actions[:, 10:12]).sum(dim=-1)
        # normed_diff = torch.exp(-actions_diff * alpha_y / sigma)
        
        return actions_diff * alpha_y
        # return actions_diff * alpha_y     

    # def _reward_similar_to_leg_pos(self):
    #     # 脚完全相同不科学，应该让小腿位置相似

    #     # "joint_names": [
    #     #     "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    #     #     "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    #     #     "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    #     #     "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    #     # ],)
    #     idx_FL = [5]   # FL_hip, FL_thigh, FL_calf
    #     idx_RR = [8]   # RR_hip, RR_thigh, RR_calf
    #     idx_FR = [2]
    #     idx_RL = [11]

    #     angles_FL = self.dof_pos[:, idx_FL]   # (N_env, 2)
    #     angles_RR = self.dof_pos[:, idx_RR]   # (N_env, 2)
    #     angles_FR = self.dof_pos[:, idx_FR]   # (N_env, 2)
    #     angles_RL = self.dof_pos[:, idx_RL]   # (N_env, 2)

    #     diff_leg_12 = torch.norm(angles_FL - angles_FR, dim=1)  # (N_env,)
    #     diff_leg_34 = torch.norm(angles_RL - angles_RR, dim=1)  # (N_env,)

    #     # 3. 映射为 [0,1] 的相似度 reward
    #     sigma_leg = self.reward_cfg["leg_posture_sigma"]
    #     reward_leg_13 = torch.exp(-diff_leg_12 / (2 * sigma_leg * sigma_leg))
    #     reward_leg_24 = torch.exp(-diff_leg_34 / (2 * sigma_leg * sigma_leg))

    #     vy_abs = torch.abs(self.commands[:, 1])
    #     alpha_y = (vy_abs / self.command_cfg["max_y"]).clamp(max=1.0)

    #     reward_leg = 0.5 * reward_leg_13 + 0.5 * reward_leg_24

    #     return reward_leg * alpha_y
    
    def _reward_collision(self):
        # 接触地面惩罚，对地面力越大惩罚越大
        collision = torch.zeros(self.num_envs,device=self.device,dtype=gs.tc_float)
        for idx in self.reset_links:
            collision += torch.square(self.connect_force[:,idx,:]).sum(dim=1)
        return collision

    # def _reward_similar_to_hip(self):
    #     hip_idx = [0, 3, 6, 9]
    #     dif = torch.abs(self.dof_pos[:, hip_idx] - self.default_dof_pos[hip_idx])
    #     return dif.sum(dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
    
    # def _reward_yspd_feetpos(self):
    #     # 试图让四足横移的时候步态正常
    #     x_FL = self.FL_foot_base_pos[:, 0]   # (N_env,)
    #     x_FR = self.FR_foot_base_pos[:, 0]
    #     x_RL = self.RL_foot_base_pos[:, 0]
    #     x_RR = self.RR_foot_base_pos[:, 0]

    #     # x_front = 0.5 * (x_FL + x_FR)
    #     # x_rear = 0.5 * (x_RL + x_RR)
    #     x_front_dif = x_FL - x_FR
    #     x_rear_dif = x_RL - x_RR

    #     vx_abs = torch.abs(self.commands[:, 0])        # 取 vx
    #     alpha_x  = 1.0 - (vx_abs / self.command_cfg["max_x"]).clamp(max=1.0)

    #     # 实际间距（绝对差）
    #     # vx_gap = torch.abs(x_front - x_rear)   # (N_env,)
    #     # 误差及奖励
    #     target_x = self.reward_cfg["desired_feet_front"]
    #     sigma_x  = self.reward_cfg["front_sigma"]
    #     error_x  = torch.square(x_front_dif - target_x) + torch.square(x_rear_dif - target_x)     # 二次误差
    #     reward_x = torch.exp(-error_x / (2 * sigma_x * sigma_x)) # 高斯式映射到 (0,1]

    #     return alpha_x
    
    # def _reward_contact_time(self):
    #     # contact_timer: 每足累计着地步数，触地 +1，离地置 0
    #     self.contact_timer[self.contact_mask.bool()] += 1
    #     self.contact_timer[~self.contact_mask.bool()]  = 0

    #     T = int(0.6 / self.dt)               # 0.6 s ≈ 70 步
    #     long_stance = (self.contact_timer > T).float()   # (N,4)
    #     pen_long_stance = long_stance.mean(dim=1)        # 0~1

    #     reward += -1.0 * pen_long_stance

    def _reward_dof_force(self):
        # 惩罚关节电机的扭矩，防止电机力矩过大
        # self.dof_force,(N_env, N_dof)
        return torch.sum(torch.square(self.dof_force), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.basic_default_dof_pos), dim=1)
    
    # #希望能让y指令的时候大腿小腿去模仿基础位置
    # def _reward_ycmd_sim_to_default(self):
    #     idx = [7, 8, 10, 11]
    #     return torch.sum(torch.abs(self.dof_pos[:, idx] - self.default_dof_pos[idx]), dim=1)
    
    def _reward_angle_change(self):
        # 惩罚机体角度变化过大
        return torch.square(self.base_euler[:, 0] - self.init_euler[:, 0]) + torch.square(self.base_euler[:, 1] - self.init_euler[:, 1])

    # def _reward_MSE(self):
    #     mean_fz  = torch.mean(self.foot_connect_force_z, dim=1, keepdim=True)             # (N_env,1)
    #     mse_load = torch.mean((self.foot_connect_force_z - mean_fz)**2, dim=1)            # (N_env,)

    #     # σ 取 0.05×TotalWeight ⇒ 允许 ±5 % 体重误差
    #     sigma2   = (0.05 * self.robot_mass * 9.81 / 4)**2
    #     rew_bal  = torch.exp(-mse_load / sigma2)                   # (0,1]

    #     return rew_bal

    def eval_debug(self):
        print(     
            # 发送位置 
            # print(f"eval_cmd_num={self.eval_cmd_num}")

            f"FR_hip_joint={self.dof_pos[0,0]:+.3f}  "                  
            f"FR_thigh_joint={self.dof_pos[0,1]:+.3f}  "      
            f"FR_calf_joint={self.dof_pos[0,2]:+.3f}  "               
            f"FL_hip_joint={self.dof_pos[0,3]:+.3f}  "    
            f"FL_thigh_joint={self.dof_pos[0,4]:+.3f}  "                  
            f"FL_calf_joint={self.dof_pos[0,5]:+.3f}  "      
            f"RR_hip_joint={self.dof_pos[0,6]:+.3f}  "               
            f"RR_thigh_joint={self.dof_pos[0,7]:+.3f}  "  
            f"RR_calf_joint={self.dof_pos[0,8]:+.3f}  "                  
            f"RL_hip_joint={self.dof_pos[0,9]:+.3f}  "      
            f"RL_thigh_joint={self.dof_pos[0,10]:+.3f}  "               
            f"RL_calf_joint={self.dof_pos[0,11]:+.3f}  "              
        )
        # print(     
        #     # 发送速度 
        #     f"FR_hip_joint={self.dof_vel[0,0]:+.3f}  "                  
        #     f"FR_thigh_joint={self.dof_vel[0,1]:+.3f}  "      
        #     f"FR_calf_joint={self.dof_vel[0,2]:+.3f}  "               
        #     f"FL_hip_joint={self.dof_vel[0,3]:+.3f}  "    
        #     f"FL_thigh_joint={self.dof_vel[0,4]:+.3f}  "                  
        #     f"FL_calf_joint={self.dof_vel[0,5]:+.3f}  "      
        #     f"RR_hip_joint={self.dof_vel[0,6]:+.3f}  "               
        #     f"RR_thigh_joint={self.dof_vel[0,7]:+.3f}  "  
        #     f"RR_calf_joint={self.dof_vel[0,8]:+.3f}  "                  
        #     f"RL_hip_joint={self.dof_vel[0,9]:+.3f}  "      
        #     f"RL_thigh_joint={self.dof_vel[0,10]:+.3f}  "               
        #     f"RL_calf_joint={self.dof_vel[0,11]:+.3f}  "              
        # )
        # print(     
        #     # 发送力矩 
        #     f"FR_hip_joint={self.dof_force[0,0]:+.3f}  "                  
        #     f"FR_thigh_joint={self.dof_force[0,1]:+.3f}  "      
        #     f"FR_calf_joint={self.dof_force[0,2]:+.3f}  "               
        #     f"FL_hip_joint={self.dof_force[0,3]:+.3f}  "    
        #     f"FL_thigh_joint={self.dof_force[0,4]:+.3f}  "                  
        #     f"FL_calf_joint={self.dof_force[0,5]:+.3f}  "      
        #     f"RR_hip_joint={self.dof_force[0,6]:+.3f}  "               
        #     f"RR_thigh_joint={self.dof_force[0,7]:+.3f}  "  
        #     f"RR_calf_joint={self.dof_force[0,8]:+.3f}  "                  
        #     f"RL_hip_joint={self.dof_force[0,9]:+.3f}  "      
        #     f"RL_thigh_joint={self.dof_force[0,10]:+.3f}  "               
        #     f"RL_calf_joint={self.dof_force[0,11]:+.3f}  "              
        # )
        # print(     
            # 发送力矩 
            # for i, link_id in enumerate(self.foot_link_ids):
            #     fz = self.connect_force[0, link_id, 2].item()   # env 0 的竖直力
            #     name = ["FL","FR","RL","RR"][i]
            # print(f"{name}_foot Fz = {fz:+.3f} N")

            # f"FR_hip_joint={self.connect_force[:, self.foot_link_ids[1], 0]}  "  
            # f"FR_hip_joint={self.connect_force[:, self.foot_link_ids[1], 1]}  " 
            # f"FR_hip_joint={self.connect_force[:, self.foot_link_ids[1], 2]}  "                 
            # f"FR_thigh_joint={self.foot_connect_force_z[0,1]:+.3f}  "      
            # f"FR_calf_joint={self.foot_connect_force_z[0,2]:+.3f}  "                             
        # )
            