import gymnasium as gym
import Mag_Env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList



import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch as th


# 重新创建环境
env = gym.make('MagnetEnv_OSC-v0', gui=1, mode='P', P_sens=1, P_max_force=60)
env = Monitor(env, 'monitor_point')

# 设置回调
eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=65000, render=False)
checkpoint_callback = CheckpointCallback(save_freq=97500, save_path='./logs/', name_prefix='rl_model')
callback = CallbackList([checkpoint_callback, eval_callback])

# 加载之前训练的模型
model = PPO.load("./logs/best_model", env)

# 继续训练
model.learn(total_timesteps=1000000, callback=callback, reset_num_timesteps=False)

# 保存模型
model.save("MagnetEnv_OSC_continued")

# 清理资源
del model