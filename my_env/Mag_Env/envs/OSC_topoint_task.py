import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R
from Mag_Env.envs import utils
from . import manipulators


from gymnasium.utils import seeding
import gymnasium as gym

u0 = 4 * np.pi * 1e-7
MAX_EPISODE_LEN = 4000

gui = 1
direct = 0


DIM_OBS = 24 # no. of dimensions in observation space
DIM_ACT = 1 # no. of dimensions in action space
#target_position = (1.1,0.0,0.55)
#target_position = (0.5,0.0,1.0)


class Env_topoint_OSC(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,gui=1,mode ="P",record=False,T_sens = 200, V_sens=1, P_sens = 1, P_max_force=300):

        self._p = utils.connect(gui)
        self._p.setRealTimeSimulation(0)
        self.step_counter = 0
        self._control_eu_or_quat = 0
        self._include_vel_obs =1

        self.mode = mode
        self.T_sens = T_sens
        self.V_sens = V_sens
        self.P_sens = P_sens
        self.P_max_force = P_max_force
        self.render_mode = None
        self.seed(2)

        if(mode == 'T'):
            print('TORQUE CONTROL')
            self.action_space = gym.spaces.box.Box(
                low = np.array([-1, -1, -1, -1, -1, -1]),
                high = np.array([1,  1,  1,  1,  1,  1])
            )
        elif(mode == 'V'):
            print("VELOCITY CONTROL")
            self.action_space = gym.spaces.box.Box(
                low = np.array([-1, -1, -1, -1, -1, -1]),
                high = np.array([1,  1,  1,  1,  1,  1])
            )
        elif (mode == 'P'):
            print("POSITION CONTROL")
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -1, -1]),
                high=np.array([1, 1, 1])
            )
        else:
            self.action_space = gym.spaces.box.Box(
                low = np.array([-5, -5, -5, -5, -5, -5]),
                high = np.array([5,  5,  5,  5,  5,  5])
            )

        #tip pos(3),tip ori(3),tip vel(3),arm vel(6),arm pos(6),target tip pos(3) ,|| obj_pos(3), obj_vel(3),distance between obj and ma(1)
        self.observation_space = gym.spaces.box.Box(    #change later
            # low=np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
            # high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            low = np.array([-2,-2,-2,-3.14 ,-3.14, -3.14,-0.5,-0.5,-0.5,-1,-1,-1,-1,-1,-1,-3.14 ,-3.14, -3.14, -3.14,-3.14,-3.14,-2,-2,-2]),
            high = np.array([2, 2, 2, 3.14, 3.14, 3.14, 0.5,0.5,0.5,1, 1, 1, 1, 1, 1,3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 2, 2, 2])
        )

        #self.reset()

    def reset(self, seed=None, **options):


        self._p.resetSimulation()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.807)
        self.planeID = self._p.loadURDF("plane.urdf")

        self.env_dict = utils.create_tabletop(self._p)
        self.agent = manipulators.Manipulator(self._p, path="Mag_Env/ur10e/ur10e.urdf", position=[0., 0., 0.4], ik_idx=6)
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
        self.init_objects()

        self.done = False
        self.terminated = False
        self.truncated = False



        self.magnet_init(150, 10)
        self.target_position = self.generate_random_position()



        self.obs = self.get_observation()


        #print(self.observation_space.shape, len(obs))
        #scaled_obs = utils.scale_gym_data(self.observation_space, obs)

        
        

        self.info = {}


        return np.array(self.obs).astype(np.float32) , self.info

    def init_agent_pose(self, t=None, sleep=False, traj=False):
        angles = [-0.294, -1.650, 2.141, -2.062, -1.572, 1.277]
        self.agent.set_joint_position(angles, t=t, sleep=sleep, traj=traj)

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
        state = self.agent.get_tip_pose()

        # ------------------------- #
        # --- Cartesian 6D pose 6--- #
        # ------------------------- #
        tip_pos = state[0]
        tip_orn = state[1]

        observation.extend(list(tip_pos))
        #observation_lim.extend(list(self._workspace_lim))

        # cartesian orientation
        if self._control_eu_or_quat == 0:
            euler = self._p.getEulerFromQuaternion(tip_orn)
            observation.extend(list(euler))  # roll, pitch, yaw
            #observation_lim.extend(self._eu_lim)
        else:
            observation.extend(list(tip_orn))
            #observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])

        # ------------------------- #
        # --- tip vel 3D--- #
        # ------------------------- #

        tip_vel = self.agent.get_tip_vel()
        observation.extend(list(tip_vel))
        # --------------------------------- #
        # --- joint velocity 6--- #
        # --------------------------------- #
        if self._include_vel_obs:
            # standardize by subtracting the mean and dividing by the std

            # vel_std = [0.04, 0.07, 0.03]
            # vel_mean = [0.0, 0.01, 0.0]
            #
            # vel_l = np.subtract(state[0], vel_mean)
            # vel_l = np.divide(vel_l, vel_std)
            vel_joint = self.agent.get_joint_velocity()
            #print("vel_joint:",vel_joint)

            observation.extend(list(vel_joint))
            #observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, -1], [-1, -1], [-1, -1]])

        # ------------------- #
        # --- Joint poses 6--- #
        # ------------------- #

        jointPoses = self.agent.get_joint_position()


        observation.extend(list(jointPoses))
        #observation_lim.extend([[self.ll[i], self.ul[i]] for i in range(0, len(self.agent._joint_name_to_ids.values()))])


        # ------------------- #
        # --- obj target poses--- #
        # ------------------- #

        observation.extend(list(self.target_position))
        #15
        #return observation, observation_lim
        #print(observation)


        # # ------------------- #
        # # --- obj position 3-- #
        # # ------------------- #


        # obj_pos, obj_ori = self._p.getBasePositionAndOrientation(self.obj)
        # obj_ori_euler = self._p.getEulerFromQuaternion(obj_ori)

        # observation.extend(list(obj_pos))
        # #observation.extend(list(obj_ori_euler))
        
        # # ------------------- #
        # # --- obj vel 3-- #
        # # ------------------- #
        
        # obj_vel, _ = self._p.getBaseVelocity(self.obj)
        # observation.extend(list(obj_vel))

        # # --------------------------------- #
        # # --- distance between obj and ma 1 #
        # # --------------------------------- #

        # distance = []
        # # di+1物体目标位置与当前位置的欧式距离
        # for i in range(len(target_position)):
        #     distance.append(pow(obj_pos[i] - tip_pos[i],2))
        # d1 = np.sqrt(np.sum(distance))

        # observation.append(d1)


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






    def init_objects(self):
        obj_type = self._p.GEOM_CYLINDER
        position = [0.8, 0.0, 0.5]
        rotation = [0, 0, 0]
        if obj_type == self._p.GEOM_CYLINDER:
            r = 0.0125
            h = 0.025
            size = [r, h]
            #rotation = [np.pi/2, 0, np.pi]
            rotation = [0, np.pi, 0]
        else:
            r = np.random.uniform(0.025, 0.05)
            size = [r, r, r]

        self.obj = utils.create_object(self._p, obj_type=obj_type, size=size, position=position,
                                       rotation=rotation, color=[0.5,0,0.5,1], mass=0.00156)

        self._p.changeDynamics(self.obj, -1, lateralFriction=0.2) #0.05
        self._p.changeDynamics(self.obj, -1, spinningFriction=0.2) #0.05
        self._p.changeDynamics(self.obj, -1, rollingFriction=0.04)
        self._p.changeDynamics(self.obj, -1, restitution=0.00)
        self._p.changeDynamics(self.obj, -1, linearDamping=130) #50可以做到悬浮,last_value 100
        self._p.changeDynamics(self.obj, -1, angularDamping=30) #初始0.04
        self._p.changeDynamics(self.obj, -1, frictionAnchor=1)


    def magnet_init(self, ma_norm,mc_norm):
        self.ma_norm = ma_norm
        self.mc_norm = mc_norm
        self.magnet_update()

    def Z(self):
        I = np.eye(3)

        p_delta_hat_outer = np.outer(self.p_delta_hat, self.p_delta_hat)
        return I - 5 * p_delta_hat_outer

    def D(self):
        I = np.eye(3)
        p_delta_hat_outer = np.outer(self.p_delta_hat, self.p_delta_hat)
        return 3 * p_delta_hat_outer - I

    def magnet_update(self):
        self.ma_position, self.ma_orientation = self.agent.get_tip_pose()
        self.mc_position, self.mc_orientation = self._p.getBasePositionAndOrientation(self.obj)

        ma_o_Matrix = np.array(self._p.getMatrixFromQuaternion(self.ma_orientation)).reshape(3, 3)
        self.ma_hat = np.array(ma_o_Matrix[:, 2]).reshape(3,1)
        mc_o_Matrix = np.array(self._p.getMatrixFromQuaternion(self.mc_orientation)).reshape(3, 3)
        self.mc_hat = np.array(mc_o_Matrix[:, 2]).reshape(3,1)
        # print("ma_hat",self.ma_hat)
        # print("mc_hat", self.mc_hat)
        self.p_delta = np.array(self.mc_position).reshape(3,1) - np.array(self.ma_position).reshape(3,1)
        self.p_delta_norm = np.linalg.norm(self.p_delta)
        self.p_delta_hat = self.p_delta / self.p_delta_norm if self.p_delta_norm != 0 else np.zeros_like(self.p_delta)
        #print("mc_o_Matrix:", mc_o_Matrix)
        self.target_mc_o_Matrix = mc_o_Matrix
        # 得到mc的目标朝向
        self.target_mc_o_Matrix[:, 2] = self.calculate_mc_hat().flatten()
        r = R.from_matrix(self.target_mc_o_Matrix)
        quaternion = r.as_quat()
        # 将四元数转换为列表形式
        self.target_mc_Orientation_quaternion = quaternion.tolist()


    def calculate_mc_hat(self):

        temp = np.dot(self.D(), self.ma_hat)
        temp_norm = np.linalg.norm(temp)
        target_mc_hat = temp / temp_norm if temp_norm != 0 else np.zeros_like(temp)
        return target_mc_hat


    def get_magnetic_force(self):
        I = np.eye(3)

        """consider ma_hat and mc_hat"""
        magnetic_force = 3 * u0 * self.ma_norm * self.mc_norm/(4 * np.pi * (self.p_delta_norm**4)) * np.dot((np.outer(self.ma_hat,self.mc_hat)+np.outer(self.mc_hat, self.ma_hat) + (np.dot(np.dot(self.mc_hat.T, self.Z()), self.ma_hat))*I), self.p_delta_hat)
        # magnetic_force = (3 * u0 * self.ma_norm * self.mc_norm / (4 * np.pi * (self.p_delta_norm ** 4)) *
        #      np.dot((np.outer(self.ma_hat, self.mc_hat) + np.outer(self.mc_hat, self.ma_hat) +
        #              ((self.mc_hat.T.dot(self.Z())).dot(self.ma_hat)) * I), self.p_delta_hat))

        #print("magnetic_force:", magnetic_force)
        """just consider ma_hat"""
        # D_ma_norm = np.linalg.norm(self.D().dot(self.ma_hat))
        # magnetic_force = (3 * u0 * self.ma_norm * self.mc_norm)/(4 * np.pi * (self.p_delta_norm ** 4) * D_ma_norm)*np.dot((np.outer(self.ma_hat,self.ma_hat)-(1+4*pow(self.ma_hat.T.dot(self.p_delta_hat),2))*I),self.p_delta_hat)

        magnetic_force_norm = np.linalg.norm(magnetic_force)
        magnetic_force_hat = magnetic_force / magnetic_force_norm

        #print("magnetic_force_hat:", magnetic_force_hat)
        if magnetic_force_norm > 0.5:
            #magnetic_force = 0.9 * self.ma_hat
            magnetic_force = 0.5* magnetic_force_hat

        return magnetic_force

    def get_buoyancy_force(self):

        buoyancy_force = np.array([0, 0, 0.0148]).reshape(3,1) #
        return buoyancy_force
        #

    def total_force(self):


        total_force = self.get_magnetic_force() + self.get_buoyancy_force()
        return total_force

    def get_magnetic_torque(self):
        magnetic_torque = u0 * self.ma_norm * self.mc_norm / (4 * np.pi * (self.p_delta_norm ** 3)) * np.dot(np.multiply(self.mc_hat, self.D()), self.ma_hat)
        #magnetic_torque = u0 * self.ma_norm * self.mc_norm / (4 * np.pi * (self.p_delta_norm ** 3)) * np.multiply(self.mc_hat, np.dot(self.D(), self.ma_hat))
        return magnetic_torque


    def get_reward(self):
        pass
    
    def step(self,action, count=13):


        f = self.total_force()
        self._p.applyExternalForce(self.obj, -1, f.flatten(), [0, 0, 0], self._p.LINK_FRAME)
        self._p.resetBasePositionAndOrientation(self.obj, self.mc_position, self.target_mc_Orientation_quaternion)

        reward = 0


        state_object, _ = self._p.getBasePositionAndOrientation(self.obj)
        state_position_tip, _ = self.agent.get_tip_pose()


        # #di物体目标位置与当前位置的欧式距离
        # result0 = []
        # for i in range(len(target_position)):
        #     result0.append(pow(target_position[i] - state_object[i],2))
        # d0 = np.sqrt(np.sum(result0))


        """test code"""
        #t = self.get_magnetic_torque()
        #self._p.applyExternalTorque(self.obj, -1, t.flatten(),  self._p.WORLD_FRAME)
        # _, mco = self._p.getBasePositionAndOrientation(self.obj)

        # _, mao = self._p.getBasePositionAndOrientation(self.tool_id)
        # mco = self._p.getMatrixFromQuaternion(mco)
        #print("f",f)
        # print("t:",t)
        # print("mco:",mco)
        #=========================================================================#
        #  Execute Actions                                                        #
        #=========================================================================#

        # METHOD 1: action = dx,dy,dz in cartesian space

        dv = 0.005 # default: 0.005, how big are the actions
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
    
    
        currentPosition , currentPose = self.agent.get_tip_pose()
    
        newPosition = [currentPosition[0] + dx,
                        currentPosition[1] + dy,
                        currentPosition[2] + dz]
    
        currentPose_Euler = self._p.getEulerFromQuaternion(currentPose)
    
        # dyaw = action[3] * dv
        # dpitch = action[4] * dv
        # droll = action[5] * dv
    
        # newPose_Euler = [currentPose_Euler[0] + dyaw,
        #             currentPose_Euler[1] + dpitch,
        #             currentPose_Euler[2] + droll]
    

        target_tip_ori = self.init_tip_ori
        self.agent.set_cartesian_position(position=newPosition, orientation=target_tip_ori)


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

        for _ in range(count):
            self._p.stepSimulation()

        state_object, _ = self._p.getBasePositionAndOrientation(self.obj)
        state_position_tip, _ = self.agent.get_tip_pose()

        self.magnet_update()
    # =========================================================================#
    #  Reward Function and Episode End States                                 #
    # =========================================================================#

    # Dense Reward:

        time_penalty = -0.1

        result1 = []
        # di+1物体目标位置与当前位置的欧式距离
        for i in range(len(self.target_position)):
            result1.append(pow(self.target_position[i] - state_position_tip[i],2))
        d1 = np.sqrt(np.sum(result1))

        r1 = -np.exp(d1)

        
        result2 = []
        tip_vel = self.agent.get_tip_vel()
        for i in range(len(tip_vel)):
            result2.append(pow(tip_vel[i],2))
        v1 = np.sqrt(np.sum(result1))

        
        # if d1 <= d0:
        #     r1 = 0
        # els
        #     r1 = -5

        # 限制机械臂移动
        # result2 = [abs(action[i]) for in range(len(action))]
        # r2 = -np.sum(result2)


        

        # obj_vel, _ = self._p.getBaseVelocity(self.obj)

        # # 是否ma，mc吸在一起
        # contacts = self._p.getContactPoints(self.obj, self.agent.arm)
        # if contacts:
        #     #self.truncated = True
        #     #print("Collision detected!")
        #     #done = True
        #     r4 = -4

        # else:
        #     if 0.62 > state_object[2] > 0.58:
                
        #         r4 = 10 - np.sum(obj_vel)

        #     else:
        #         r4 = -2

        # # 是否和桌面接触
        
        # contact1 = self._p.getContactPoints(self.env_dict["table"], self.obj)
        # if contact1:
        #     r5 = -4
        # else:
        #     r5 = 0

        # #是否和地面碰撞
        # contact2 = self._p.getContactPoints(self.planeID,self.obj)
        # if contact2:
        #     self.truncated = True
        # #reward = Q1*r1 + Q3*r3 + Q4*r4


        reward = r1 
        reward += time_penalty
        


        # End episode
        self.step_counter += 1
        if self.step_counter > MAX_EPISODE_LEN or (d1 < 0.05 and v1 < 0.01):
            # reward = 0
            #self.done = True
            self.terminated = True
        # print("REWARD: ",reward)
        self.obs= self.get_observation()
        #scaled_obs = utils.scale_gym_data(self.observation_space, obs)
        self.info = {"reward":reward}
        #print("obs:", obs)
        return np.array(self.obs).astype(np.float32), reward, self.terminated, self.truncated, self.info



    def test_step(self,count=25):
        f = self.total_force()
        self._p.applyExternalForce(self.obj, -1, f.flatten(), [0, 0, 0], self._p.LINK_FRAME)
        self._p.resetBasePositionAndOrientation(self.obj, self.mc_position, self.target_mc_Orientation_quaternion)

        for _ in range(count):
            self._p.stepSimulation()
        self.magnet_update()




    def render(self, mode='human'):
        pass


    def seed(self,seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed = seed)
        return seed

    def close(self):
        self._p.disconnect()


    def generate_random_position(self):
        # 设置随机种子
        seed = self.seed()

        np_random = np.random.RandomState(43)

        # 生成随机位置，每个分量的取值范围为 [0, 1)
        x = np_random.rand()
        y = np_random.rand()
        z = 0.5 + np_random.rand() * 0.5

        # 返回随机位置的数组，形状为 (3,)
        return np.array([x, y, z])
