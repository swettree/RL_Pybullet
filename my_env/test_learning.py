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


env = gym.make('MagnetEnv_OSC-v0',gui=1, mode='P',P_sens=1,P_max_force=60)

model = PPO.load("Mag_OSC_model.zip", env=env,custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,})

rew = []

for i in range(5):
    terminated = False
    obs,_ = env.reset()
    while not terminated:
        action, _state =model.predict(obs)
        #action = np.array([0,0,0,0,0,0,0])
        obs,reward,terminated,_,_ = env.step(action)
        print(reward)
        print(obs)
        print(action)
        #print(action)
        if i==0:
            rew.append(reward)

t = np.arange(len(rew))
print(sum(rew))
fig,ax = plt.subplots()
ax.plot(t,rew)
ax.set_title("Testing")
ax.set_xlabel("Timesteps")
ax.set_ylabel("Rewards")
plt.show()