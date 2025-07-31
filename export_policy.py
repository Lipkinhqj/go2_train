# export_policy.py  (放在项目根：~/RL/hqj_go2/)
import os, torch

# ---------- 配置 ----------
CKPT_PATH = "logs/go2-walking-Adddomain/model_9000.pt"   # ← 你的 checkpoint
OUT_JIT   = "logs/go2-walking-Adddomain/policy.pt"       # 导出路径
OBS_DIM, ACT_DIM = 45, 12
HIDDEN = [512, 256, 128]          # 与 train_cfg 中 actor_hidden_dims 一致
# ---------------------------

# ① 读取 checkpoint
ckpt = torch.load(CKPT_PATH, map_location="cpu")
print("✔ 读取 checkpoint，包含 keys ->", ckpt.keys())

# ② 导入 ActorCritic 类并实例化
from rsl_rl.modules.actor_critic import ActorCritic   # 路径按实际改
ac = ActorCritic(OBS_DIM, OBS_DIM, ACT_DIM,
                 actor_hidden_dims=HIDDEN,
                 critic_hidden_dims=HIDDEN,
                 activation="elu")

# ③ 加载权重（整包 state-dict）
ac.load_state_dict(ckpt["model_state_dict"], strict=True)
print("✔ model_state_dict 已加载")

# ④ 导出 TorchScript 策略（仅 actor）
ac.actor.eval()
dummy = torch.randn(1, OBS_DIM)
traced = torch.jit.trace(ac.actor, dummy)
traced.save(OUT_JIT)
print(f"✅ 已导出 TorchScript 策略 → {OUT_JIT}")
