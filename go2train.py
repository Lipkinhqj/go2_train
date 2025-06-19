import argparse
#处理命令行参数（如实验名称、并行环境数、训练迭代次数）
import os
#管理日志目录（创建/删除）
import pickle
#保存实验配置（环境、奖励等参数）
import shutil
from importlib import metadata
import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
#导入仿真框架
from hqj_go2env import go2EnvCreate
#自定义机器人仿真环境
import go2param as gp


def main():
    #创建解析器对象
    parser = argparse.ArgumentParser(description="train go2.")

    # 添加命令行参数
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking-AddFeetZpos")
    parser.add_argument("-B", "--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=3001)
    args = parser.parse_args()

    # 初始化 Genesis
    gs.init(backend=gs.gpu, logging_level="warning")

    # 配置路径与参数
    log_dir = f"logs/{args.exp_name}"
    checkpoint_path = os.path.join(log_dir, "model_1000.pt")
    auto_resume = os.path.exists(checkpoint_path)

    # 加载参数
    env_cfg, obs_cfg, reward_cfg, command_cfg = gp.get_cfgs()
    train_cfg = gp.get_train_cfg(args.exp_name, args.max_iterations)

    # 如果不是 resume，清理旧目录，保存配置
    if not auto_resume:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        # # 保存训练配置
        # with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        #     pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)
    else:
        print(f"[INFO] Detected existing checkpoint: {checkpoint_path}, loading...")

    # 保存训练配置（只有首次训练时保存）
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # if os.path.exists(log_dir):
    #     shutil.rmtree(log_dir)
    # os.makedirs(log_dir, exist_ok=True)
    #环境创建
    env = go2EnvCreate(args.num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # 如果需要恢复，则加载模型
    if auto_resume:
        # runner.load(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=gs.device)

        net_attr = "actor_critic"  
        if not hasattr(runner.alg, net_attr):
            raise AttributeError(f"runner.alg has no attribute '{net_attr}'")
        getattr(runner.alg, net_attr).load_state_dict(ckpt["model_state_dict"])

        # —— 3. 加载归一化状态 —— 
        runner.obs_normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(ckpt["critic_obs_norm_state_dict"])
    # 启动训练
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True,
    )

if __name__ == "__main__":
    main()

    # tensorboard --logdir logs/go2-walking --port 6006
    # python train.py -e go2-walking --resume_from logs/go2-walking/model_100.pt
    # /home/lipkin/anaconda3/envs/RL/bin/python /home/lipkin/RL/hqj_go2/go2train.py -e go2-walking --resume_from logs/go2-walking/model_100.pt

