import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R
from Mag_Env.envs import utils
from . import manipulators

import time
from gymnasium.utils import seeding
import gymnasium as gym
import csv
u0 = 4 * np.pi * 1e-7
MAX_EPISODE_LEN = 200000

gui = 1
direct = 0


DIM_OBS = 24 # no. of dimensions in observation space
DIM_ACT = 1 # no. of dimensions in action space
#target_position = (1.1,0.0,0.55)



def generate_trajectory_numpy(form, num_points):
    trajectory = []
    
    if form == 0:
        trajectory = generate_circle_trajectory(num_points)
    elif form == 1:
        trajectory = generate_spiral_trajectory(num_points)
    elif form == 2:
        trajectory = generate_yz_wave_trajectory(num_points)
    elif form == 3:
        trajectory = generate_xy_wave_trajectory(num_points)
    else:
        print("incorrect form")
        return None
        
    return np.array(trajectory)

def generate_circle_trajectory(num_points):
    trajectory = []
    radius = 0.1
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = 0.2 + radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 1.2
        trajectory.append([x, y, z])
    return trajectory

def generate_spiral_trajectory(num_points):
    trajectory = []
    radius = 0.1
    num_turns = 3
    height = 0.2
    for i in range(num_points):
        angle = 2 * np.pi * num_turns * i / num_points
        x = 0.2 + radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 1.1 + height * i / num_points
        trajectory.append([x, y, z])
    return trajectory

def generate_yz_wave_trajectory(num_points):
    trajectory = []
    amplitude = 0.16
    wide = 0.04
    frequency = 2
    delta_p = ((2 * frequency + 1) * amplitude + 2 * frequency * wide) / num_points
    segment = num_points / (frequency * 12 + 1)
    cir = 0
    up = 1
    x = 0.2
    y = -0.08
    z = 1.12
    trajectory.append([x, y, z])

    for i in range(num_points):
        if cir % 2 != 0:
            y += delta_p
        else:
            if up == 1:
                z += delta_p
            else:
                z -= delta_p

        if i in [4*segment, 9*segment, 14*segment, 19*segment]:
            cir += 1
            up = 1 - up
        elif i in [5*segment, 10*segment, 15*segment, 20*segment]:
            cir += 1

        trajectory.append([x, y, z])
    return trajectory

def generate_xy_wave_trajectory(num_points):
    trajectory = []
    amplitude = 0.16
    wide = 0.04
    frequency = 2
    delta_p = ((2 * frequency + 1) * amplitude + 2 * frequency * wide) / num_points
    segment = num_points / (frequency * 12 + 1)
    cir = 0
    up = 1
    x = 0.2
    y = -0.08
    z = 1.12
    trajectory.append([x, y, z])

    for i in range(num_points):
        if cir % 2 != 0:
            y += delta_p
        else:
            if up == 1:
                x += delta_p
            else:
                x -= delta_p

        if i in [4*segment, 9*segment, 14*segment, 19*segment]:
            cir += 1
            up = 1 - up
        elif i in [5*segment, 10*segment, 15*segment, 20*segment]:
            cir += 1

        trajectory.append([x, y, z])
    return trajectory

def write_to_csv(file_path, data_list):
    """将数据写入 CSV 文件"""
    with open(file_path, mode='a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_list)

def save_position_data(state_object, target_position, trajectory_flag, form):
    """保存胶囊位置和目标位置的数据"""
    capsule_pos_data = np.array(state_object)
    target_pos_data = np.array(target_position)
    
    if capsule_pos_data.ndim == 1:
        capsule_pos_data = capsule_pos_data.reshape(-1, 1)
    if target_pos_data.ndim == 1:
        target_pos_data = target_pos_data.reshape(-1, 1)
    
    capsule_pos_list = capsule_pos_data.tolist()
    target_pos_list = target_pos_data.tolist()

    # 根据轨迹类型设置文件路径
    if trajectory_flag:
        file_capsule_path = f'data/traject/{get_trajectory_folder(form)}/capsule_pos.csv'
        file_target_path = f'data/traject/{get_trajectory_folder(form)}/target_pos.csv'
    else:
        file_capsule_path = 'data/capsule_pos.csv'
        file_target_path = 'data/target_pos.csv'

    # 写入位置数据
    write_to_csv(file_capsule_path, capsule_pos_list)
    write_to_csv(file_target_path, target_pos_list)

    return capsule_pos_data, target_pos_data

def get_trajectory_folder(form):
    """返回对应轨迹的文件夹名称"""
    if form == 1:
        return 'screw'
    elif form == 2:
        return 'square'
    elif form == 3:
        return 'square/xy'
    else:
        raise ValueError("Invalid form value")

def save_error_data(capsule_pos_data, target_pos_data):
    """计算并保存位置误差"""
    error_data = np.linalg.norm(capsule_pos_data - target_pos_data)
    write_to_csv('data/error_200.csv', [[error_data]])

def save_velocity_data(obj, p):
    """保存物体速度数据"""
    current_velocity, _ = p.getBaseVelocity(obj)
    velocity_data = np.linalg.norm(current_velocity)
    write_to_csv('data/test_velocity.csv', [[velocity_data]])



class MagnetEnv_OSC(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,gui=1,mode ="P",record=False,T_sens = 200, V_sens=1, P_sens = 1, P_max_force=300):

        self._p = utils.connect(gui)
        #self._p.setRealTimeSimulation(0)
        self.timeStep=1./ 240.
        self._p.setTimeStep(self.timeStep)

        self._p.setPhysicsEngineParameter(numSubSteps=2)
        # self._p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        # self._p.setPhysicsEngineParameter(contactSlop=0.001)
        # self._p.setPhysicsEngineParameter(erp=0.2)
        # self._p.setPhysicsEngineParameter(frictionERP=0.2)
        # # self._p.setPhysicsEngineParameter(enableFrictionAnchor=True)
        # self._p.setPhysicsEngineParameter(restitutionVelocityThreshold=0.2)

        # solver_residual_threshold = 0.01
        # num_solver_iterations = 200
        # contact_breaking_threshold = 0.01
        # self._p.setPhysicsEngineParameter(solverResidualThreshold=solver_residual_threshold, 
        #                     numSolverIterations=num_solver_iterations, 
        #                     contactBreakingThreshold=contact_breaking_threshold)
    
        self.step_counter = 0
        self._control_eu_or_quat = 1
        self._include_vel_obs =1

        self.mode = mode
        self.T_sens = T_sens
        self.V_sens = V_sens
        self.P_sens = P_sens
        self.P_max_force = P_max_force
        self.render_mode = None

        # self.target_position = [0.16,0.06,1.15]
        # self.target_position = [0.1,-0.1,1.1]
        self.target_position = [0.2,0.0,1.2]
        self.target_vel = [0.0,0.0,0.0]
        self.target_point = [0.0,0.0,1.0]
        self.lineardamp = 80
        self.angulatdamp = 2
        self.dv = 0.2
        self.action_scale = 1
        

        self.pre_action = np.array([0,0,0,0,0,0])
        self.action_fre = 2
        self.last_command_time = None
        self.alpha = 0.6

        """数据记录"""
        self.record_flag = False

        """是否生成轨迹"""
        self.trajectory_flag = False
        self.form = 1
        self.point_num = 0
        self.point_number = 4500
        self.target_pos_fre = 4
        if self.trajectory_flag:
            self.trajectory = generate_trajectory_numpy(self.form , self.point_number)
            # print(self.trajectory)
        # self._render_sleep = 1
        # self._last_frame_time = 0.0
        self.disturb_flag = False
        

        self.setup_action_space(mode)
        self.setup_observation_space()

        # obs = [ "q_pos 6", "q_vel 6", "eef_pos 3", "eef_quat 4",  "eef_vel 6", "capsule_pos 3", "capsule_quat 4", 
        #        "capsule_linear_vel 3", "capsule_pos_relative 3","mc_hat 3","target_pos 3","target_vel 3","target_point 3"]

        # self.observation_space = gym.spaces.box.Box(    #change later
        #     # low=np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
        #     # high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        #     low = np.array([-2,-2,-2,-3.14 ,-3.14, -3.14,-0.5,-0.5,-0.5,-1,-1,-1,-1,-1,-1,-3.14 ,-3.14, -3.14, -3.14,-3.14,-3.14,-2,-2,-2,-2,-2,-2,-2,-2,-2,-1]),
        #     high = np.array([2, 2, 2, 3.14, 3.14, 3.14, 0.5,0.5,0.5,1, 1, 1, 1, 1, 1,3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 2, 2, 2, 2, 2, 2, 2, 2, 2,1])
        # )

        #self.reset()
    def setup_action_space(self, mode):
        if mode == 'T':
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -1, -1, -1, -1, -1]),
                high=np.array([1, 1, 1, 1, 1, 1])
            )
        elif mode == 'V':
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -1, -1, -1, -1, -1]),
                high=np.array([1, 1, 1, 1, 1, 1])
            )
        elif mode == 'P':
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -1, -1, -1, -1, -1]),
                high=np.array([1, 1, 1, 1, 1, 1])
            )
        else:
            self.action_space = gym.spaces.box.Box(
                low=np.array([-5, -5, -5, -5, -5, -5]),
                high=np.array([5, 5, 5, 5, 5, 5])
            )
        
        # obs = [ "q_pos 6", "q_vel 6", "eef_pos 3", "eef_quat 4",  "eef_vel 6", "capsule_pos 3", "capsule_quat 4", 
        #        "capsule_linear_vel 3", "capsule_pos_relative 3","mc_hat 3","target_pos 3","target_vel 3","target_point 3"]
    def setup_observation_space(self):
        q_pos_low, q_pos_high = np.array([-3.141,-3.141,-3.141,-3.141,-3.141,-3.141]), np.array([3.141,3.141,3.141,3.141,3.141,3.141])
        q_vel_low, q_vel_high = np.array([-3.141,-3.141,-3.141,-6.282,-6.282,-6.282]), np.array([3.141,3.141,3.141,6.282,6.282,6.282])
        eef_pos_low, eef_pos_high = np.array([-2,-2,-2]), np.array([2, 2, 2])
        eef_quat_low, eef_quat_high = np.array([-1,-1,-1,-1]), np.array([1, 1, 1, 1])
        eef_vel_low, eef_vel_high = np.array([-1,-1,-1,-1,-1,-1]), np.array([1, 1, 1, 1, 1, 1])
        capsule_pos_low, capsule_pos_high = np.array([-2,-2,-2]), np.array([2, 2, 2])
        capsule_quat_low, capsule_quat_high = np.array([-1,-1,-1,-1]), np.array([1, 1, 1, 1])
        capsule_linear_vel_low, capsule_linear_vel_high = np.array([-1,-1,-1]), np.array([1, 1, 1])
        capsule_pos_relative_low, capsule_pos_relative_high = np.array([-2,-2,-2]), np.array([2, 2, 2])
        mc_hat_low, mc_hat_high = np.array([-1,-1,-1]), np.array([1, 1, 1])
        target_pos_low, target_pos_high = np.array([-2,-2,-2]), np.array([2, 2, 2])
        target_vel_low, target_vel_high = np.array([-1,-1,-1]), np.array([1, 1, 1])
        target_point_low, target_point_high = np.array([-1,-1,-1]), np.array([1, 1, 1])

        low = np.concatenate([q_pos_low, q_vel_low, eef_pos_low, eef_vel_low, mc_hat_low, capsule_pos_low, capsule_linear_vel_low, mc_hat_low, target_pos_low, target_point_low])
        high = np.concatenate([q_pos_high, q_vel_high, eef_pos_high, eef_vel_high, mc_hat_high, capsule_pos_high, capsule_linear_vel_high, mc_hat_high, target_pos_high, target_point_high])

        self.observation_space = gym.spaces.box.Box(low=low, high=high)

    def reset(self, seed=None, **options):


        self._p.resetSimulation()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.807)
        self.planeID = self._p.loadURDF("plane.urdf")
        self.external_force = np.array([0,0,0])
   

        self.env_dict = utils.create_tabletop(self._p)
        


        self.agent = manipulators.Manipulator(self._p, path="Mag_Env/ur10e/ur10e.urdf", position=[-0.5, 0., 1.125], ik_idx=6)
        base_constraint = self._p.createConstraint(parentBodyUniqueId=self.env_dict["base"], parentLinkIndex=0,
                                                   childBodyUniqueId=self.agent.arm, childLinkIndex=-1,
                                                   jointType=self._p.JOINT_FIXED, jointAxis=(0, 0, 0),
                                                   parentFramePosition=(0, 0, 0),
                                                   childFramePosition=(0.0, 0.0, -0.2),
                                                   childFrameOrientation=(0, 0, 0, 1))
        self._p.changeConstraint(base_constraint, maxForce=10000)

        #创建工具和约束
        # self.tool_id = utils.create_object(self._p, self._p.GEOM_CYLINDER, size=[0.025, 0.05], position=[0, 0, 0], mass=1.0, color=[0.5, 0.2, 0.5, 1.0])

        # tool_constraint = self._p.createConstraint(parentBodyUniqueId=self.agent.arm, parentLinkIndex=6,
        #                                            childBodyUniqueId=self.tool_id, childLinkIndex=-1,
        #                                            jointType=self._p.JOINT_FIXED, jointAxis=(0, 0, 0),
        #                                            parentFramePosition=(0, 0, 0), childFramePosition=(0, 0, 0.03),
        #                                            childFrameOrientation=self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]))
        # self._p.changeConstraint(tool_constraint, maxForce=10000)

        # num_joints = 6
        # idx = 0
        # for i in range(num_joints):
        #     joint_info = self._p.getJointInfo(self.agent.id, i)
        #     joint_name = joint_info[1].decode("UTF-8")
        #     joint_type = joint_info[2]
        #
        #     if joint_type is self._p.JOINT_REVOLUTE or joint_type is self._p.JOINT_PRISMATIC:
        #         assert joint_name in self.agent.initial_positions.keys()
        #
        #         self.agent._joint_name_to_ids[joint_name] = i
        #
        #         # self._p.resetJointState(self.agent.id, i, self.agent.initial_positions[joint_name])
        #         # self._p.setJointMotorControl2(self.agent.id, i, self._p.POSITION_CONTROL,
        #         #                         targetPosition=self.agent.initial_positions[joint_name],
        #         #                         positionGain=0.2, velocityGain=1.0
        #         #                         )
        #         #
        #         # idx += 1
        #
        # self.ll, self.ul, self.jr, self.rs = self.agent.get_joint_ranges()

        self.step_counter = 0
        self.obj = None
        self.init_agent_pose(t=1)
        self.init_tip_pos , self.init_tip_ori = self.agent.get_tip_pose()
        # self.init_glass()
        self.init_capsule()
        
        self.done = False
        self.terminated = False
        self.truncated = False

        # 画世界坐标系
        # self._p.addUserDebugLine([0, 0, 0], [2, 0, 0], [1, 0, 0], lineWidth=2,lifeTime=0)
        # self._p.addUserDebugLine([0, 0, 0], [0, 2, 0], [0, 1, 0], lineWidth=2,lifeTime=0)
        # self._p.addUserDebugLine([0, 0, 0], [0, 0, 2], [0, 0, 1], lineWidth=2,lifeTime=0)
        visual_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_SPHERE, radius=0.005, rgbaColor=[1, 0, 0, 1])
        self._p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=self.target_position)
        self.lower_limits, self.upper_limits, self.joint_ranges, self.rest_poses = self.agent.get_joint_ranges()
        # 0.8052,0.1664
        self.magnet_init(82.5, 0.8052)

        self.obs = self.get_observation()


        #print(self.observation_space.shape, len(obs))
        #scaled_obs = utils.scale_gym_data(self.observation_space, obs)

        
        

        self.info = {}


        return np.array(self.obs).astype(np.float32) , self.info

    def init_agent_pose(self, t=None, sleep=False, traj=False):
        angles = [-0.294, -1.650, 2.141, -2.062, -1.572, 1.277]
        #self.agent.set_joint_position(angles, t=t, sleep=sleep, traj=traj)
        self._p.setJointMotorControlArray(
            bodyUniqueId=self.agent.arm,
            jointIndices=self.agent.joints,
            controlMode=self._p.POSITION_CONTROL,
            targetPositions=angles,
            forces=self.agent.forces,
        # positionGain=[400.0,400.0,400.0,400.0,400.0,400.0],
        # velocityGain=[40.0,40.0,40.0,40.0,40.0,40.0]
        )
        self.agent._waitsleep(t, sleep)
    # def create_gym_spaces(self):
    #     # Configure observation limits
    #     obs, obs_lim = self.get_extended_observation()
    #     observation_dim = len(obs)
    #     #print(observation_dim)
    #     observation_low = []
    #     observation_high = []
    #     for el in obs_lim:
    #         observation_low.extend([el[0]])
    #         observation_high.extend([el[1]])
    #
    #     # Configure the observation space
    #     observation_space = gym.spaces.Box(np.array(observation_low), np.array(observation_high), dtype='float32')
    #
    #     # Configure action space
    #     action_dim = DIM_ACT
    #     action_bound = 1
    #     action_high = np.array([action_bound] * action_dim)
    #     action_space = gym.spaces.Box(-action_high, action_high, dtype='float32')
    #
    #     return observation_space, action_space

    def get_observation(self):
        # Create observation state
        observation = []
        #observation_lim = []

        # Get state of the end-effector link
        
        """q_pos  6 机械臂反馈"""
        q_pos = self.agent.get_joint_position()
        observation.extend(list(q_pos))
        """q_vel  6 机械臂反馈"""
        q_vel = self.agent.get_joint_velocity()
        observation.extend(list(q_vel))
        # print(q_vel)
        """eef_pos  3 机械臂反馈"""
        state = self.agent.get_tip_pose()
        tip_pos = state[0]
        
        observation.extend(list(tip_pos))
        # print(tip_pos)
        """eef_quat  4 """
        # tip_orn = state[1]
        # if self._control_eu_or_quat == 0:
        #     euler = self._p.getEulerFromQuaternion(tip_orn)
        #     observation.extend(list(euler))  # roll, pitch, yaw
        #     #observation_lim.extend(self._eu_lim)
        # else:
        #     observation.extend(list(tip_orn))
            
        """eef_vel  6末端执行器的速度,last - now"""
        tip_vel ,tip_angle_vel= self.agent.get_tip_vel()
        observation.extend(list(tip_vel))
        observation.extend(list(tip_angle_vel))

        """ma_hat 3 机械臂反馈"""
        observation.extend(list(self.ma_hat.squeeze()))
        """capsule_pos  3 视觉决定位置"""
        obj_pos, obj_ori = self._p.getBasePositionAndOrientation(self.obj)
        #obj_ori_euler = self._p.getEulerFromQuaternion(obj_ori)
        noise = np.random.normal(0, 0.01, size=3)  # 高斯噪声，均值为0，标准差为0.01
        noisy_obj_pos = np.array(obj_pos) + noise + np.array([0.005,0.005,0.005])
        observation.extend(list(noisy_obj_pos))

        """capsule_quat  4"""
        #observation.extend(list(obj_ori))

        """capsule_linear_vel  3 视觉计算"""
        obj_vel, _ = self._p.getBaseVelocity(self.obj)
        observation.extend(list(obj_vel))

        """capsule_pos_relative  3 计算末端执行器和胶囊的位置"""
        # observation.extend(list(self.p_delta.squeeze()))

        """mc_hat  3 开环计算"""
        observation.extend(list(self.mc_hat.squeeze()))

        """target_pos  3 目标位置"""
        observation.extend(list(self.target_position))

        """target_vel  3 目标速度"""
        # observation.extend(list(self.target_vel))

        """target_point  3 目标朝向"""
        observation.extend(list(self.target_point))

        """pre_action 6 上一次动作"""
        # observation.extend(list(self.pre_action))
        # observation_clip = np.clip(observation,self.clip_ob_min,self.clip_ob_max)

        return observation

    # def get_extended_observation(self):
    #     self._observation = []
    #     observation_lim = []
    #
    #     # ----------------------------------- #
    #     # --- Robot and world observation 15--- #
    #     # ----------------------------------- #
    #     #15
    #     robot_observation, robot_obs_lim = self.get_observation()
    #
    #     self._observation.extend(list(robot_observation))
    #     observation_lim.extend(robot_obs_lim)
    #
    #
    #     # ----------------------------------------- #
    #     # --- Object pose 3--- #
    #     # ----------------------------------------- #
    #     state_object, _ = self._p.getBasePositionAndOrientation(self.obj)
    #
    #     self._observation.extend(list(state_object))
    #
    #     observation_lim.extend(self._workspace_lim)
    #     # ------------------- #
    #     # --- Target pose 6--- #
    #     # ------------------- #
    #     self._observation.extend(list(target_position))
    #     observation_lim.extend(self._workspace_lim)
    #
    #     return np.array(self._observation), observation_lim






    # def init_objects(self):
    #     obj_type = self._p.GEOM_CYLINDER
    #     position = [0.8, 0.0, 0.5]
    #     rotation = [0, 0, 0]
    #     if obj_type == self._p.GEOM_CYLINDER:
    #         r = 0.0125
    #         h = 0.025
    #         size = [r, h]
    #         #rotation = [np.pi/2, 0, np.pi]
    #         rotation = [0, np.pi, 0]
    #     else:
    #         r = np.random.uniform(0.025, 0.05)
    #         size = [r, r, r]

    #     self.obj = utils.create_object(self._p, obj_type=obj_type, size=size, position=position,
    #                                    rotation=rotation, color=[0.5,0,0.5,1], mass=0.00156)

    #     self._p.changeDynamics(self.obj, -1, lateralFriction=0.99) #0.05
    #     self._p.changeDynamics(self.obj, -1, spinningFriction=0.05) #0.05
    #     self._p.changeDynamics(self.obj, -1, rollingFriction=0.04)
    #     self._p.changeDynamics(self.obj, -1, restitution=0.00)
    #     self._p.changeDynamics(self.obj, -1, linearDamping=130) #50可以做到悬浮,last_value 100
    #     self._p.changeDynamics(self.obj, -1, angularDamping=0.04) #初始0.04
    #     self._p.changeDynamics(self.obj, -1, frictionAnchor=1)
    #     self._p.changeDynamics(self.obj, -1, ccdSweptSphereRadius=0.02, contactProcessingThreshold=0.0)


    def init_capsule(self):
        path="Mag_Env/mc_capsule/urdf/mc_capsule.urdf"
        #position = [0.1, 0.0, 1.0316]
        position = [0.2, 0.0, 1.0316 + 0.005]
        rotation = [0.0, 0.0, 0.0, 1.0]
        # | self._p.URDF_USE_INERTIA_FROM_FILE 
        self.obj = self._p.loadURDF(
        # fileName=path,basePosition=position,baseOrientation=rotation,useMaximalCoordinates = True,
        #     flags=self._p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | self._p.URDF_MERGE_FIXED_LINKS )
        fileName=path,basePosition=position,baseOrientation=rotation , flags= self._p.URDF_USE_INERTIA_FROM_FILE)
        self._p.changeDynamics(self.obj, -1, lateralFriction=0.01) #0.05
        self._p.changeDynamics(self.obj, -1, spinningFriction=0.0) #0.05
        self._p.changeDynamics(self.obj, -1, rollingFriction=0.0)
        self._p.changeDynamics(self.obj, -1, restitution=0.0)
        self._p.changeDynamics(self.obj, -1, linearDamping = self.lineardamp) #80
        self._p.changeDynamics(self.obj, -1, angularDamping = self.angulatdamp ) #初始0.04


        # self._p.changeDynamics(self.obj, -1, contactStiffness=1e3, contactDamping=1e2) 


        # self._p.changeDynamics(self.env_dict["table"], -1, lateralFriction=0.05, spinningFriction=0.03, rollingFriction=0.02, restitution=0.0)
        # self._p.changeDynamics(self.obj, -1, frictionAnchor=1)

        info = self._p.getDynamicsInfo(self.obj, -1)
        # print(info)

    def init_glass(self):
        path  =   "Mag_Env/glass_box/urdf/glass_box.urdf"
        position = [0.2,0.0,1.025]
        rotation = [0.0, 0.0, 0.0, 1.0]
        self.glass_box = self._p.loadURDF(
            fileName = path, basePosition = position, baseOrientation = rotation
        )
        self._p.changeDynamics(self.glass_box, -1, lateralFriction=1.0) #0.05
        self._p.changeDynamics(self.glass_box, -1, spinningFriction=0.0) #0.05
        self._p.changeDynamics(self.glass_box, -1, rollingFriction=0.0)
        self._p.changeDynamics(self.glass_box, -1, restitution=0.0)

    def magnet_init(self, ma_norm,mc_norm):
        self.ma_norm = ma_norm
        self.mc_norm = mc_norm
        self.magnet_update()

    def Z(self):
        I = np.eye(3)

        p_delta_hat_outer = np.dot(self.p_delta_hat, self.p_delta_hat.T)
        return I - 5 * p_delta_hat_outer

    def D(self):
        I = np.eye(3)
        p_delta_hat_outer = np.dot(self.p_delta_hat, self.p_delta_hat.T)
        # print(p_delta_hat_outer)
        return 3 * p_delta_hat_outer - I
    

    def magnet_update(self):
        self.ma_position, self.ma_orientation = self.agent.get_tip_pose()
        self.mc_position, self.mc_orientation = self._p.getBasePositionAndOrientation(self.obj)
        
        self.ma_o_Matrix = np.array(self._p.getMatrixFromQuaternion(self.ma_orientation)).reshape(3, 3)
        # 0 ,2
        self.ma_hat = np.array(self.ma_o_Matrix[:, 0]).reshape(3,1)

        self.mc_o_Matrix = np.array(self._p.getMatrixFromQuaternion(self.mc_orientation)).reshape(3, 3)
        self.mc_hat = np.array(self.mc_o_Matrix[:, 1]).reshape(3,1)
        
        # print("ma_hat",self.ma_hat)
        # print("mc_hat", self.mc_hat)
        self.p_delta = np.array(self.mc_position).reshape(3,1) - np.array(self.ma_position).reshape(3,1)
        self.p_delta_norm = np.linalg.norm(self.p_delta)
        self.p_delta_hat = self.p_delta / self.p_delta_norm 
        #print("mc_o_Matrix:", self.mc_o_Matrix)
        # self.target_mc_o_Matrix = self.mc_o_Matrix
        # # 得到mc的目标朝向
        # self.target_mc_o_Matrix[:, 2] = self.calculate_mc_hat().flatten()
        # r = R.from_matrix(self.target_mc_o_Matrix)
        # quaternion = r.as_quat()
        # # 将四元数转换为列表形式
        # self.target_mc_Orientation_quaternion = quaternion.tolist()


    # def calculate_mc_hat(self):

    #     temp = np.dot(self.D(), self.ma_hat)
    #     temp_norm = np.linalg.norm(temp)
    #     target_mc_hat = temp / temp_norm if temp_norm != 0 else np.zeros_like(temp)
    #     return target_mc_hat


    def get_magnetic_force(self):
        I = np.eye(3)

        """consider ma_hat and mc_hat"""
        magnetic_force = 3 * u0 * self.ma_norm * self.mc_norm/(4 * np.pi * (self.p_delta_norm**4)) * np.dot(
            (np.dot(self.ma_hat,self.mc_hat.T)+np.dot(self.mc_hat, self.ma_hat.T) + (np.dot(np.dot(self.mc_hat.T, self.Z()), self.ma_hat))*I), self.p_delta_hat)
        # magnetic_force = (3 * u0 * self.ma_norm * self.mc_norm / (4 * np.pi * (self.p_delta_norm ** 4)) *
        #      np.dot((np.outer(self.ma_hat, self.mc_hat) + np.outer(self.mc_hat, self.ma_hat) +
        #              ((self.mc_hat.T.dot(self.Z())).dot(self.ma_hat)) * I), self.p_delta_hat))

        #print("magnetic_force:", magnetic_force)
        """just consider ma_hat"""
        # D_ma_norm = np.linalg.norm(self.D().dot(self.ma_hat))
        # magnetic_force = (3 * u0 * self.ma_norm * self.mc_norm)/(4 * np.pi * (self.p_delta_norm ** 4) * D_ma_norm)*np.dot((np.outer(self.ma_hat,self.ma_hat)-(1+4*pow(self.ma_hat.T.dot(self.p_delta_hat),2))*I),self.p_delta_hat)

        # magnetic_force_norm = np.linalg.norm(magnetic_force)
        # magnetic_force_hat = magnetic_force / magnetic_force_norm

        # #print("magnetic_force_hat:", magnetic_force_hat)
        # if magnetic_force_norm > 1:
        #     #magnetic_force = 0.9 * self.ma_hat
        #     magnetic_force = 1 * magnetic_force_hat

        return magnetic_force

    def get_buoyancy_force(self):

        buoyancy_force = np.array([0, 0, 0.036764]).reshape(3,1) #
        return buoyancy_force
        #

    def total_force(self):

        
        _total_force = self.get_magnetic_force() + self.get_buoyancy_force()

        return _total_force

    def get_magnetic_torque(self):
        
        #magnetic_torque = u0 * self.ma_norm * self.mc_norm / (4 * np.pi * (self.p_delta_norm ** 3)) * np.dot(np.cross(self.mc_hat, self.D()), self.ma_hat)
        # print()
        # print(np.dot(self.D(), self.ma_hat))
        
        a = np.squeeze(self.mc_hat)
        b = np.squeeze(np.dot(self.D(), self.ma_hat))
        c = np.cross(a,b).reshape(3,1)
        magnetic_torque = u0 * self.ma_norm * self.mc_norm / (4 * np.pi * (self.p_delta_norm ** 3)) * c 

        # print(c)

        return magnetic_torque

    def get_jacobian(self, robot_id, end_effector_index, joint_indices):
        # Get joint states (position, velocity, and torque)
        joint_states = self._p.getJointStates(robot_id, joint_indices)
        joint_positions = [state[0] for state in joint_states]
        
        # Get link state (position and orientation)
        link_state = self._p.getLinkState(robot_id, end_effector_index, computeForwardKinematics=True)
        link_trn = link_state[0]
        link_rot = link_state[1]

        # Compute Jacobian
        zero_vec = [0.0] * len(joint_indices)
        local_position = [0.0, 0.0, 0.0]
        jac_t, jac_r = self._p.calculateJacobian(robot_id, end_effector_index, local_position, joint_positions, zero_vec, zero_vec)
        
        # Convert to numpy arrays
        jac_t = np.array(jac_t)
        jac_r = np.array(jac_r)
        return jac_t, jac_r

    def damped_least_squares_ik(self, robot_id, end_effector_index, joint_indices, dpose, damping=0.05):
        jac_t, jac_r = self.get_jacobian(robot_id, end_effector_index, joint_indices)
        
        # Combine translational and rotational Jacobians
        jac = np.vstack((jac_t, jac_r))

        # Compute DLS solution
        jac_T = jac.T
        lmbda = np.eye(6) * (damping ** 2)
        dpose = np.array(dpose).reshape(-1, 1)
        
        dq = np.dot(np.dot(jac_T , np.linalg.inv(np.dot(jac, jac_T) + lmbda) ), dpose)
        dq = dq.flatten()
        
        return dq

    def get_reward(self):
        pass

    def apply_velocity_change(self):
        velocity_change = np.random.uniform(-8, 8, size=3)
        current_velocity, _ = self._p.getBaseVelocity(self.obj)
        current_velocity = np.array(current_velocity)
        new_velocity = current_velocity + velocity_change
        self._p.resetBaseVelocity(self.obj, linearVelocity=new_velocity.tolist())
        
    def apply_lateral_force_once(self):
        if not hasattr(self, 'lateral_force_applied'):
            # 在xy方向采样横向力，只采样一次
            self.lateral_force = np.random.uniform(-0.1, 0.1, size=2)  # 只采样xy方向的力
            self.lateral_force_applied = True  # 标记力已经采样

        decay_factor = 0.99 ** (self.step_counter - 400)
        # 将力的z方向设为0，因为只需要在xy方向上施加力
        force = np.array([self.lateral_force[0], self.lateral_force[1], 0]) * decay_factor
        # 获取物体当前位置
        obj_position, _ = self._p.getBasePositionAndOrientation(self.obj)

        # 施加外力到物体的中心位置
        return force
        

    def step(self,action):
        
        # if self._render_sleep:
        #     time_spent = time.time() - self._last_frame_time
        #     self._last_frame_time = time.time()
        #     time_to_sleep = self._action_repeat * self._time_step - time_spent
        #     if time_to_sleep > 0:
        #         time.sleep(time_to_sleep)
        if ((self.step_counter % 600 == 0) & (self.step_counter >= 1) & (self.disturb_flag)) :
        # 抽样新的速度增量
            self.apply_velocity_change()
    
        # if (self.step_counter >= 400) & self.disturb_flag:
        #     self.external_force = self.apply_lateral_force_once()

        if self.trajectory_flag:
            if (self.step_counter % self.target_pos_fre) == 0:
                self.point_num += 1
                print(self.point_num)
                if self.point_num >= self.point_number:
                    self.point_num = self.point_number
                    print("over")
                    self.terminated = True
            self.target_position = self.trajectory[self.point_num-1,:]


        reward = 0
        joint_indices = [0,1,2,3,4,5]


        #=========================================================================#
        #  Execute Actions                                                        #
        #=========================================================================#

        # METHOD 1: action = dx,dy,dz in cartesian space

        #default: 0.2 /3, how big are the actions
        # dx = action[0] * dv / 3
        # dy = action[1] * dv / 3
        # dz = action[2] * dv / 3
    
    
        # currentPosition , currentPose = self.agent.get_tip_pose()
    
        # newPosition = [currentPosition[0] + dx,
        #                 currentPosition[1] + dy,
        #                 currentPosition[2] + dz]
    
        # currentPose_Euler = self._p.getEulerFromQuaternion(currentPose)
    
        # dyaw = action[3] * dv / 3
        # dpitch = action[4] * dv / 3
        # droll = action[5] * dv / 3
    
        # newPose_Euler = [currentPose_Euler[0] + dyaw,
        #             currentPose_Euler[1] + dpitch,
        #             currentPose_Euler[2] + droll]
    
        # self.agent.set_cartesian_position(position=newPosition, orientation=newPose_Euler)


        # METHOD 2: action = delta_q
        # Get the current joint angles

        # joint_angles = self.agent.get_joint_position()

        # Apply the delta_q values from the action vector
        # dv = 0.02  # how big are the actions
        # joint_angles = [a + action[i] * dv for i, a in enumerate(joint_angles)]
        # self.agent.set_joint_position(joint_angles)
        # for i in range(1):  # ADD THIS AS AN ARGUMENT TO ENV CONSTRUCTOR
        #     self.agent.apply_action(action, self.mode, torque_sens=self.T_sens, vel_sens=self.V_sens,
        #                           pos_sens=self.P_sens, P_mx_fr=self.P_max_force)

        
        # METHOD 3: 阻尼最小二乘
        if self.pre_action is None:
            osc_dq = action
        else:
            osc_dq = self.alpha * action + (1 - self.alpha) * self.pre_action
        osc_dq = osc_dq * self.dv / self.action_scale

        dq = self.damped_least_squares_ik(self.agent.arm, 6, joint_indices, osc_dq)
        
        joint_states = self._p.getJointStates(self.agent.arm, joint_indices)
        joint_positions = [state[0] for state in joint_states]
        
        joint_target_pos = joint_positions + dq 

        joint_target_pos = np.clip(joint_target_pos,self.lower_limits,self.upper_limits)

        self.agent.set_joint_position(joint_target_pos)

        # print("目标关节位置:", joint_target_pos)


        # METHOD 4: 速度控制
        # cmd_limit = [1.56, 1.56, 1.56, 3.14, 3.14, 3.14]
        # dq = np.array(action) * cmd_limit
        # self.agent.apply_action(dq, self.mode,torque_sens=None,pos_sens=None, vel_sens=1,P_mx_fr=None)
        for i in range(1):
            self.magnet_update()
            f = self.total_force()+self.external_force.reshape(3,1)
            tau = self.get_magnetic_torque()
            ma_pos, ma_orn= self.ma_position , self.ma_o_Matrix
            mc_pos, mc_orn= self.mc_position , self.mc_o_Matrix
            # print("mc_pos:",mc_pos)
            # tau = np.around(tau, decimals=4)
            # tau = np.array([0, 0, 0])

            self._p.applyExternalForce(self.obj, -1, f.flatten(), mc_pos, self._p.WORLD_FRAME)
            self._p.applyExternalTorque(self.obj, -1, tau.flatten(), self._p.WORLD_FRAME)
            self._p.stepSimulation()
            time.sleep(self.timeStep)


        # contact_points = self._p.getContactPoints(self.obj)
        # if contact_points:
        #     for point in contact_points:
        #         print(f"接触点位置: {point[5]}, 法向力: {point[9]}, 法向力: {point[10]}")
        # else:
        #     print("当前没有接触点")

        
        self.pre_action = action


    # =========================================================================#
    #  Reward Function and Episode End States                                 #
    # =========================================================================#

        state_object, _ = self._p.getBasePositionAndOrientation(self.obj)
        state_position_tip, _ = self.agent.get_tip_pose()

        obj_position = []
        result1 = []
        # di+1物体目标位置与当前位置的欧式距离
        for i in range(len(self.target_position)):
            result1.append(pow(self.target_position[i] - state_object[i],2))
            
            obj_position.append(state_object[i])

        dx = np.abs(self.target_position[0] - state_object[0])
        dy = np.abs(self.target_position[1] - state_object[1])
        dz = np.abs(self.target_position[2] - state_object[2])
        d = np.sqrt(np.sum(result1))

        # print(d)
        # 是否ma，mc吸在一起
        contacts = self._p.getContactPoints(self.obj, self.agent.arm)
        if contacts:
            self.truncated = True


        if self.record_flag:
            # 记录当前位置和目标位置
            capsule_pos_data, target_pos_data = save_position_data(state_object, self.target_position, self.trajectory_flag, self.form)

            # 记录误差
            save_error_data(capsule_pos_data, target_pos_data)

            # 记录速度
            save_velocity_data(self.obj, self._p)



        #print(state_position_tip)
       
        # End episode
        self.step_counter += 1
        if self.step_counter > MAX_EPISODE_LEN :

            self.terminated = True
        # print("REWARD: ",reward)
        self.obs = self.get_observation()
        
        #scaled_obs = utils.scale_gym_data(self.observation_space, obs)
        self.info = {"obj_position":np.array(obj_position) ,"step_counter":self.step_counter, "max_length":MAX_EPISODE_LEN}
        # print("obs:", self.obs)
        return np.array(self.obs).astype(np.float32), reward, self.terminated, self.truncated, self.info
    

    
    def test_step(self):
        

        for i in range(1):
            self.magnet_update()
            f = self.total_force()
            tau = self.get_magnetic_torque()
            ma_pos, ma_orn= self.ma_position , self.ma_o_Matrix
            
            mc_pos, mc_orn= self.mc_position , self.mc_o_Matrix
            print(ma_pos)

            # print("p",self.p_delta[2])
            # print("f",f[2])
            # print("t",tau)
            # tau = np.around(tau, decimals=4)
            # tau = np.array([0, 0, 0])
            self._p.applyExternalForce(self.obj, -1, f.flatten(), mc_pos, self._p.WORLD_FRAME)
            self._p.applyExternalTorque(self.obj, -1,tau.flatten(),self._p.WORLD_FRAME)

            self._p.stepSimulation()
            time.sleep(self.timeStep)
       

        
        # print(f)
        axis_length = 0.2
        # """ma画坐标轴"""
        ma_pos_x_end = ma_pos + axis_length*ma_orn[:,0].flatten()
        ma_pos_y_end = ma_pos + axis_length*ma_orn[:,1].flatten()
        ma_pos_z_end = ma_pos + axis_length*ma_orn[:,2].flatten()
        self._p.addUserDebugLine(ma_pos, ma_pos_x_end, [1, 0, 0], lineWidth=2,lifeTime=0.09)
        self._p.addUserDebugLine(ma_pos, ma_pos_y_end, [0, 1, 0], lineWidth=2,lifeTime=0.09)
        self._p.addUserDebugLine(ma_pos, ma_pos_z_end, [0, 0, 1], lineWidth=2,lifeTime=0.09)
        # """mc画坐标轴"""
        # mc_pos, mc_orn= self.mc_position , self.mc_o_Matrix
        # mc_pos_x_end = mc_pos + axis_length*mc_orn[:,0].flatten()
        # mc_pos_y_end = mc_pos + axis_length*mc_orn[:,1].flatten()
        # mc_pos_z_end = mc_pos + axis_length*mc_orn[:,2].flatten()
        # self._p.addUserDebugLine(mc_pos, mc_pos_x_end, [1, 0, 0], lineWidth=2,lifeTime=0.09)
        # self._p.addUserDebugLine(mc_pos, mc_pos_y_end, [0, 1, 0], lineWidth=2,lifeTime=0.09)
        # self._p.addUserDebugLine(mc_pos, mc_pos_z_end, [0, 0, 1], lineWidth=2,lifeTime=0.09)

        """pybullet 自带api"""
        # dv = 0.005
        # action = [0,0,1,0,0,0]
        # dx = action[0] * dv
        # dy = action[1] * dv
        # dz = action[2] * dv
    
    
        # currentPosition , currentPose = self.agent.get_tip_pose()
    
        # newPosition = [currentPosition[0] + dx,
        #                 currentPosition[1] + dy,
        #                 currentPosition[2] + dz]
    
        # currentPose_Euler = self._p.getEulerFromQuaternion(currentPose)
    
        # dyaw = action[3] * dv
        # dpitch = action[4] * dv
        # droll = action[5] * dv
    
        # newPose_Euler = [currentPose_Euler[0] + dyaw,
        #             currentPose_Euler[1] + dpitch,
        #             currentPose_Euler[2] + droll]
        
        # newOrientation = self._p.getQuaternionFromEuler(newPose_Euler)
        # # if (self.step_counter % 2 == 0):ma_height
        # #     current_time = time.time()
        # #     if self.last_command_time is not None:
        # #         interval = current_time - self.last_command_time
        # #         print(f"间隔时间: {interval:.6f} 秒")
        # #     self.last_command_time = current_time

        # self.agent.set_cartesian_position(position=newPosition, orientation=newOrientation)
            
        """手写阻尼IK"""
        jointindices = [0,1,2,3,4,5]

        

        d_action = [0,0,-0.001,0,0,0]
        dq = self.damped_least_squares_ik(self.agent.arm, 6, jointindices, d_action)
        
        joint_states = self._p.getJointStates(self.agent.arm, jointindices)
        joint_positions = [state[0] for state in joint_states]
        
        joint_target_pos = joint_positions + dq
        # self.agent.set_joint_position(joint_target_pos)

        # print("目标关节位置:", joint_target_pos)

        """速度控制"""

        # cmd_limit = [1.56, 1.56, 1.56, 3.14, 3.14, 3.14]
        # dq = np.array(action) * cmd_limit
        # self.agent.apply_action(dq, self.mode,vel_sens=1)

        # 30Hz

        self.step_counter += 1
        

        




    def render(self, mode='human'):
        pass


    def seed(self,seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        self._p.disconnect()



