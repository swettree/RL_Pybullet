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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('MagnetEnv_OSC-v0', gui=1, mode='P')

min_action = torch.tensor([-1, -1, -1, -1, -1, -1], dtype=torch.float32).to(device)
max_action = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32).to(device)
num_observations=39
num_actions=6

clip_ob_min = -5.0
clip_ob_max = 5.0
# 加载保存的checkpoint
checkpoint = torch.load('UR10eMagStack.pth', map_location=device)
state_dict = checkpoint['model']
# 定义模型结构
class ActorNet(nn.Module):
    def __init__(self, num_observations, num_actions):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(num_observations, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256,256)
        self.mean = nn.Linear(256, num_actions)
        self.logstd = nn.Parameter(torch.zeros(num_actions))  # 使用参数而不是线性层

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))
        mean = self.mean(x)
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

    '''第二步 编写根据状态选择动作的函数'''
    def choose_action(self,s):
        state = torch.tensor(s, dtype=torch.float32).to(device)
        state = torch.clamp(state,clip_ob_min,clip_ob_max)
        state = self.normalize_obs(state)
        mean, std = self.pi(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, min_action, max_action)
      
        return action

# 设置设备

# 用于归一化obs
mean = torch.tensor(state_dict['running_mean_std.running_mean'], dtype=torch.float32)
var = torch.tensor(state_dict['running_mean_std.running_var'], dtype=torch.float32)

# mean = state_dict['running_mean_std.running_mean'].clone().detach()
# var = state_dict['running_mean_std.running_var'].clone().detach()
# 实例化模型并将其移动到相应设备
actor = Actor(mean,var)







# 提取并加载模型参数
actor.pi.fc1.weight.data = state_dict['a2c_network.actor_mlp.0.weight']
actor.pi.fc1.bias.data = state_dict['a2c_network.actor_mlp.0.bias']
actor.pi.fc2.weight.data = state_dict['a2c_network.actor_mlp.2.weight']
actor.pi.fc2.bias.data = state_dict['a2c_network.actor_mlp.2.bias']
actor.pi.fc3.weight.data = state_dict['a2c_network.actor_mlp.4.weight']
actor.pi.fc3.bias.data = state_dict['a2c_network.actor_mlp.4.bias']
actor.pi.fc4.weight.data = state_dict['a2c_network.actor_mlp.6.weight']
actor.pi.fc4.bias.data = state_dict['a2c_network.actor_mlp.6.bias']
actor.pi.mean.weight.data = state_dict['a2c_network.mu.weight']
actor.pi.mean.bias.data = state_dict['a2c_network.mu.bias']
actor.pi.logstd.data = state_dict['a2c_network.sigma']
# 创建环境
# print(state_dict['a2c_network.mu.weight'].shape)
# print(state_dict['a2c_network.mu.bias'].shape)
# print(state_dict['a2c_network.sigma'].shape)


rew = []
dis_error = []
lasttime = None
target_position = []

# 出热点图
# for k in range(10):
#     for j in range(10):
#         for i in range(10):
#             terminated = False
#             truncated = False
#             obs, _ = env.reset()
#             while not (terminated or truncated):
#                 obs[33] += 0.02*(i)
#                 obs[34] += 0.02*(j)
#                 obs[35] += 0.02*(k)

#                 obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)  
#                 action = actor.choose_action(obs_tensor).detach().cpu().numpy()
#                 # print(action)  
#                 obs, reward, terminated, truncated, info = env.step(action)
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
                
#             obj_position = info["obj_position"]
#             target_position = np.array([0.1 +0.02*(i), -0.1 + 0.02*(j),1.1+ 0.02*(k)])
#             distance = np.linalg.norm(obj_position - target_position)
        

#             dis_error.append(distance)
#             dis_error.append(np.abs(obj_position[0] - target_position[0]))
#             dis_error.append(np.abs(obj_position[1] - target_position[1]))
#             dis_error.append(np.abs(obj_position[2] - target_position[2]))
            
            # print(info["d"])
            # print(i)
            # print(j)
            # print(k)
            # with open('output.csv', mode='a+', newline='') as file:
            #     writer = csv.writer(file)
            
            # # 逐行写入数组数据
            #     writer.writerow(dis_error)
            # target_position = np.array([])  # 重新分配为空数组

            # dis_error.clear()

for i in range(100):
    terminated = False
    truncated = False
    obs, _ = env.reset()
    while not (terminated or truncated):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)  
        action = actor.choose_action(obs_tensor).detach().cpu().numpy()
        # print(action)  
        obs, reward, terminated, truncated, info = env.step(action)
        # current_time = time.time()
        # if lasttime is not None:
        #     interval = current_time - lasttime
        #     # print(f"间隔时间: {interval:.6f} 秒")
        # # print(action)
        # lasttime = current_time



        
# t = np.arange(len(rew))
# print(sum(rew))
# fig, ax = plt.subplots()
# ax.plot(t, rew)
# ax.set_title("Testing")
# ax.set_xlabel("Timesteps")
# ax.set_ylabel("Rewards")
# plt.show()