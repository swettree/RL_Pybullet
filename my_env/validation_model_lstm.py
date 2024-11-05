import gymnasium as gym
import Mag_Env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import csv
from scipy.interpolate import CubicSpline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('MagnetEnv_OSC-v0', gui=1, mode='P')

min_action = torch.tensor([-1, -1, -1, -1, -1, -1], dtype=torch.float32).to(device)
max_action = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32).to(device)
num_observations = 39
num_actions = 6

clip_ob_min = -5.0
clip_ob_max = 5.0

# 加载保存的checkpoint
checkpoint = torch.load('UR10eMagStack.pth', map_location=device)
state_dict = checkpoint['model']

# 定义模型结构
class ActorNet(nn.Module):
    def __init__(self, num_observations, num_actions):
        super(ActorNet, self).__init__()
        # MLP部分
        self.fc1 = nn.Linear(num_observations, 256)
        self.fc2 = nn.Linear(256 + num_observations, 512)  # D2RL: 输入包括上一层的输出和原始输入

        # LSTM部分
        self.lstm = nn.LSTM(512 + num_observations, 256, batch_first=True)  # Concat input with fc2 output

        # 输出层
        self.mean = nn.Linear(256, num_actions)
        self.logstd = nn.Parameter(torch.zeros(num_actions))  # 使用参数而不是线性层

    def forward(self, x):
        # MLP 部分
        x_fc1 = F.elu(self.fc1(x))
        x_fc2 = F.elu(self.fc2(torch.cat([x_fc1, x], dim=-1)))  # D2RL: 将上一层的输出和原始输入拼接

        # LSTM 部分
        x_concat = torch.cat([x_fc2, x], dim=-1).unsqueeze(0)  # Concat input with fc2 output
        lstm_out, _ = self.lstm(x_concat)
        lstm_out = lstm_out.squeeze(0)  # 移除batch维度，得到形状为 (256,)

        # 输出层
        mean = self.mean(lstm_out)
        std = torch.exp(self.logstd)  # 确保std为正数
        return mean, std

class Actor():
    def __init__(self, mean, var, epsilon=1e-5):
        self.pi = ActorNet(num_observations, num_actions).to(device)
        self.mean = mean.to(device)
        self.var = var.to(device)
        self.epsilon = epsilon

    def normalize_obs(self, obs):
        return (obs - self.mean) / torch.sqrt(self.var + self.epsilon)

    def choose_action(self, s):
        state = torch.tensor(s, dtype=torch.float32).to(device)
        state = torch.clamp(state, clip_ob_min, clip_ob_max)
        state = self.normalize_obs(state)
        mean, std = self.pi(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, min_action, max_action)
        return action

# 设置设备
mean = torch.tensor(state_dict['running_mean_std.running_mean'], dtype=torch.float32)
var = torch.tensor(state_dict['running_mean_std.running_var'], dtype=torch.float32)

# 实例化模型并将其移动到相应设备
actor = Actor(mean, var)

# 提取并加载模型参数
# actor.pi.fc1.weight.data = state_dict['a2c_network.actor_mlp.0.weight']
# actor.pi.fc1.bias.data = state_dict['a2c_network.actor_mlp.0.bias']

# actor.pi.fc2.weight.data = state_dict['a2c_network.actor_mlp.2.weight']
# actor.pi.fc2.bias.data = state_dict['a2c_network.actor_mlp.2.bias']

# actor.pi.lstm.weight_ih_l0.data = state_dict['a2c_network.a_rnn.rnn.weight_ih_l0']
# actor.pi.lstm.weight_hh_l0.data = state_dict['a2c_network.a_rnn.rnn.weight_hh_l0']
# actor.pi.lstm.bias_ih_l0.data = state_dict['a2c_network.a_rnn.rnn.bias_ih_l0']
# actor.pi.lstm.bias_hh_l0.data = state_dict['a2c_network.a_rnn.rnn.bias_hh_l0']

# actor.pi.mean.weight.data = state_dict['a2c_network.mu.weight']
# actor.pi.mean.bias.data = state_dict['a2c_network.mu.bias']
# actor.pi.logstd.data = state_dict['a2c_network.sigma']

# 提取并加载模型参数
actor.pi.fc1.weight.data = state_dict['a2c_network.actor_mlp.linears.0.weight']
actor.pi.fc1.bias.data = state_dict['a2c_network.actor_mlp.linears.0.bias']

actor.pi.fc2.weight.data = state_dict['a2c_network.actor_mlp.linears.1.weight']
actor.pi.fc2.bias.data = state_dict['a2c_network.actor_mlp.linears.1.bias']

actor.pi.lstm.weight_ih_l0.data = state_dict['a2c_network.a_rnn.rnn.weight_ih_l0']
actor.pi.lstm.weight_hh_l0.data = state_dict['a2c_network.a_rnn.rnn.weight_hh_l0']
actor.pi.lstm.bias_ih_l0.data = state_dict['a2c_network.a_rnn.rnn.bias_ih_l0']
actor.pi.lstm.bias_hh_l0.data = state_dict['a2c_network.a_rnn.rnn.bias_hh_l0']

actor.pi.mean.weight.data = state_dict['a2c_network.mu.weight']
actor.pi.mean.bias.data = state_dict['a2c_network.mu.bias']
actor.pi.logstd.data = state_dict['a2c_network.sigma']


rew = []
dis_error = []
lasttime = None
target_position = []


# 出热点图
# for k in range(5):
#     for j in range(11):
#         for i in range(11):
#             terminated = False
#             truncated = False
#             obs, _ = env.reset()

#             target_position = np.array([0.1 +0.02*(i), -0.1 + 0.02*(j),1.1+ 0.05*(k)])
#             while not (terminated or truncated):
#                 obs[33] = target_position[0]
#                 obs[34] = target_position[1]
#                 obs[35] = target_position[2]

#                 obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)  
#                 action = actor.choose_action(obs_tensor).detach().cpu().numpy()
#                 # print(action)  
#                 obs, reward, terminated, truncated, info = env.step(action)

#                 current_step = info["step_counter"]
#                 max_length = info["max_length"]
                
#                 if current_step >= (max_length-100):
                    
#                     obj_position = info["obj_position"]
#                     distance = np.linalg.norm(obj_position - target_position)
#                     dis_error.append(distance)

#                 # obs[33] target_x , obs[34] target_y ,obs[35] target_z 
#                 # current_time = time.time()
#                 # if lasttime is not None:
#                 #     interval = current_time - lasttime
#                 #     # print(f"间隔时间: {interval:.6f} 秒")
#                 # # print(action)
#                 # lasttime = current_time

#                 # if i == 0:
#                 #     rew.append(reward)
#                 # print(info["d"])
                
#             # print(dis_error)
#             if dis_error:
#                 dis_mean_error = np.mean(dis_error)
#                 print(dis_mean_error)
#             else:
#                 dis_mean_error = 0
            
        

            
#             # dis_error.append(np.abs(obj_position[0] - target_position[0]))
#             # dis_error.append(np.abs(obj_position[1] - target_position[1]))
#             # dis_error.append(np.abs(obj_position[2] - target_position[2]))
            
#             #print(info["d"])
#             print(i)
#             print(j)
#             print(k)
#             with open('output_6.csv', mode='a+', newline='') as file:
#                 writer = csv.writer(file)
            
#             # 逐行写入数组数据
#                 writer.writerow([i, j, k, dis_mean_error])
#             target_position = np.array([])  # 重新分配为空数组

#             dis_error.clear()
#             dis_mean_error = np.array([])
# 验证模型
# def smooth_trajectory(start_action, end_action, num_steps):
#     # 创建起点和终点动作之间的时间序列
#     t = np.array([0, 1])  # 起点和终点时间
#     y = np.vstack([start_action, end_action])  # 起点和终点的动作

#     # 使用CubicSpline进行插值
#     cs = CubicSpline(t, y, axis=0)

#     # 生成插值点
#     t_vals = np.linspace(0, 1, num_steps)
#     interpolated_actions = cs(t_vals)

#     return interpolated_actions


for i in range(1):
    terminated = False
    truncated = False
    obs, _ = env.reset()
    while not (terminated or truncated):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        action = actor.choose_action(obs_tensor).detach().cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)

# for i in range(1):
#     terminated = False
#     truncated = False
#     obs, _ = env.reset()
#     last_action = None  # 用于保存上一个动作
#     num_steps = 3  # 选择插值的步数，可以根据需要调整（比如10个step来平滑过渡）
    
#     while not (terminated or truncated):
#         obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
#         current_action = actor.choose_action(obs_tensor).detach().cpu().numpy()
        
#         # 如果有上一个动作，就进行平滑插值
#         if last_action is not None:
#             interpolated_actions = smooth_trajectory(last_action, current_action, num_steps)
            
#             # 在每个插值点执行动作
#             for interpolated_action in interpolated_actions:
#                 obs, reward, terminated, truncated, info = env.step(interpolated_action)
                
#                 # 每隔 1/240 秒执行一步动作
#                 time.sleep(1/240)  # 模拟真实时间步长
                
#                 if terminated or truncated:
#                     break
#         else:
#             # 第一次动作选择时没有上一个动作，直接执行
#             obs, reward, terminated, truncated, info = env.step(current_action)
        
#         # 更新上一个动作
#         last_action = current_action
