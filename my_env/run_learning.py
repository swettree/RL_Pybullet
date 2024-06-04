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




env = gym.make('MagnetEnv_OSC-v0',gui=0, mode='P',P_sens=1,P_max_force=60)
# Parallel environments

env = Monitor(env,'monitor_Mag')

eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=65000,
                             render=False)
checkpoint_callback = CheckpointCallback(save_freq=97500, save_path='./logs/',
                                         name_prefix='rl_model')
callback = CallbackList([checkpoint_callback, eval_callback])



policy_kwargs = dict(ortho_init=False,activation_fn=th.nn.ReLU,net_arch=dict(pi=[256,128,64], vf=[256,128,64]))
# It will check your custom environment and output additional warnings if needed
model = PPO('MlpPolicy',env,policy_kwargs=policy_kwargs,verbose=1,learning_rate=0.0001,clip_range=0.2, tensorboard_log = "./tensorboard/MagnetEnv_OSC-v0/")
model.learn(6000000,callback=callback,reset_num_timesteps=False)
t = env.get_episode_rewards()
model.save("Mag_OSC_model")


#model = PPO.load('logs/best_model.zip',env)
del model


file_name = "rewards_Mag.pkl"
op_file = open(file_name,'wb')
pickle.dump(t, op_file)
op_file.close()

fi,a = plt.subplots()
a.plot(np.arange(len(t)),t)
plt.show()






