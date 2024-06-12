from Mag_Env.envs.joint_pos_task import MagnetEnv
from Mag_Env.envs.OSC_task import MagnetEnv_OSC
from Mag_Env.envs.OSC_topoint_task import Env_topoint_OSC
SLEEP = False
import time

env = MagnetEnv_OSC(gui=1)
# env = Env_topoint_OSC(gui=1)


env.reset()
# fixed_joint = env._p.createConstraint(parentBodyUniqueId=env.env_dict["table"],
#                                  parentLinkIndex=-1,
#                                  childBodyUniqueId=env.obj,
#                                  childLinkIndex=-1,
#                                  jointType=env._p.JOINT_FIXED,
#                                  jointAxis=[0, 0, 0],
#                                  parentFramePosition=[0, 0, 0.4],
#                                  childFramePosition=[0, 0, 0])

# 使mc和ma在同一Z轴
# constraint_id = env._p.createConstraint(parentBodyUniqueId=env.tool_id,
#                                     parentLinkIndex=-1,
#                                     childBodyUniqueId=env.obj,
#                                     childLinkIndex=-1,
#                                     jointType=env._p.JOINT_PRISMATIC,
#                                     jointAxis=[0, 0, 1],
#                                     parentFramePosition=[0, 0, 0],  # 约束在 object1 原点
#                                     childFramePosition=[0, 0, 0])  # 约束在 object2 原点

# 获取针尖位置
# info = env._p.getBodyInfo(env.tool_id)
# info2 = env._p.getNumJoints(env.tool_id)
# info3 = env._p.getLinkState(env.tool_id,1)
# print("tool info:", info)
# print("tool info2:", info2)
# print("tool info3:", info3)
#action1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.25]
lasttime = time.time()
while True:






    #print("ma_hat:", env.ma_hat)
    #print("mc_hat:", env.mc_hat)

    #print("target_mc:", env.target_mc_Orientation_quaternion)

    # print(env.D())
    #print("torque:",torque)
    #print("t:", t)
    # env.calculate_mc_hat()
    # env._p.removeConstraint(constraint_id)


    # env._p.applyExternalForce(env.obj, -1, f.flatten(), [0, 0, 0], env._p.LINK_FRAME)
    # env._p.resetBasePositionAndOrientation(env.obj, env.mc_position, env.target_mc_Orientation_quaternion)

    #ma_position, _ = env._p.getBasePositionAndOrientation(env.tool_id)
    #print("ma_position:", ma_position)
    #env._p.applyExternalTorque(env.obj, -1, t.flatten(), env._p.WORLD_FRAME)
    #env.agent.add_debug_param()
    #env.agent.update_debug()
    
    env.test_step()
    v,_ = env.agent.get_tip_vel()
    print(v)
    contact1 = env._p.getContactPoints(env.env_dict["table"], env.obj)
    contact2 = env._p.getContactPoints(env.obj, env.agent.arm)
    if (contact2):
        current_time = time.time()
        if lasttime is not None:
            interval = current_time - lasttime
            # print(f"间隔时间: {interval:.6f} 秒")
        lasttime = current_time
        env.reset()



