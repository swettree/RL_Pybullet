from gymnasium.envs.registration import register


register(
    id = 'MagnetEnv-v0',
    entry_point = 'Mag_Env.envs.joint_pose_task:MagnetEnv'
)

register(
    id = 'MagnetEnv_OSC-v0',
    entry_point = 'Mag_Env.envs.OSC_task:MagnetEnv_OSC'
)