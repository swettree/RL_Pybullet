import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R
from Mag_Env.envs import utils
from . import manipulators

import time
from gymnasium.utils import seeding
import gymnasium as gym

u0 = 4 * np.pi * 1e-7
MAX_EPISODE_LEN = 600

gui = 1
direct = 0


DIM_OBS = 24 # no. of dimensions in observation space
DIM_ACT = 1 # no. of dimensions in action space
#target_position = (1.1,0.0,0.55)



class MagnetEnv_OSC(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,gui=1,mode ="P",record=False,T_sens = 200, V_sens=1, P_sens = 1, P_max_force=300):

        self._p = utils.connect(gui)
        #self._p.setRealTimeSimulation(0)
        self.timeStep=1./ 240.
        self._p.setTimeStep(self.timeStep)

        self._p.setPhysicsEngineParameter(numSubSteps=2)
        self._p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        self._p.setPhysicsEngineParameter(contactSlop=0.001)
        self._p.setPhysicsEngineParameter(erp=0.2)
        self._p.setPhysicsEngineParameter(frictionERP=0.2)
        # self._p.setPhysicsEngineParameter(enableFrictionAnchor=True)
        self._p.setPhysicsEngineParameter(restitutionVelocityThreshold=0.2)

        solver_residual_threshold = 0.01
        num_solver_iterations = 200
        contact_breaking_threshold = 0.01
        self._p.setPhysicsEngineParameter(solverResidualThreshold=solver_residual_threshold, 
                            numSolverIterations=num_solver_iterations, 
                            contactBreakingThreshold=contact_breaking_threshold)
    
        self.step_counter = 0
        self._control_eu_or_quat = 1
        self._include_vel_obs =1

        self.mode = mode
        self.T_sens = T_sens
        self.V_sens = V_sens
        self.P_sens = P_sens
        self.P_max_force = P_max_force
        self.render_mode = None
        
        self.target_position = [0.5,0.0,1.6]
        self.target_vel = [0.0,0.0,0.0]
        self.target_point = [0.0,0.0,-1.0]

        self.action_fre = 2
        self.last_command_time = None


 
        # self._render_sleep = 1
        # self._last_frame_time = 0.0
        
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
                low=np.array([-1, -1, -1, -1, -1, -1]),
                high=np.array([1, 1, 1, 1, 1, 1])
            )
        else:
            self.action_space = gym.spaces.box.Box(
                low = np.array([-5, -5, -5, -5, -5, -5]),
                high = np.array([5,  5,  5,  5,  5,  5])
            )

        # obs = [ "q_pos 6", "q_vel 6", "eef_pos 3", "eef_quat 4",  "eef_vel 6", "capsule_pos 3", "capsule_quat 4", 
        #        "capsule_linear_vel 3", "capsule_pos_relative 3","mc_hat 3","target_pos 3","target_vel 3","target_point 3"]

        # self.observation_space = gym.spaces.box.Box(    #change later
        #     # low=np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
        #     # high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        #     low = np.array([-2,-2,-2,-3.14 ,-3.14, -3.14,-0.5,-0.5,-0.5,-1,-1,-1,-1,-1,-1,-3.14 ,-3.14, -3.14, -3.14,-3.14,-3.14,-2,-2,-2,-2,-2,-2,-2,-2,-2,-1]),
        #     high = np.array([2, 2, 2, 3.14, 3.14, 3.14, 0.5,0.5,0.5,1, 1, 1, 1, 1, 1,3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 2, 2, 2, 2, 2, 2, 2, 2, 2,1])
        # )
        q_pos_low, q_pos_high = np.array([-3.14,-3.14,-3.14,-3.14,-3.14,-3.14]),np.array([3.14,3.14,3.14,3.14,3.14,3.14]) # 6
        q_vel_low, q_vel_high = np.array([-3.14,-3.14,-3.14,-6.28,-6.28,-6.28]),np.array([3.14,3.14,3.14,6.28,6.28,6.28]) # 6
        eef_pos_low, eef_pos_high = np.array([-2,-2,-2]),np.array([2, 2, 2]) # 3
        eef_quat_low, eef_quat_high = np.array([-1,-1,-1,-1]),np.array([1, 1, 1, 1]) # 4
        eef_vel_low, eef_vel_high = np.array([-1,-1,-1,-1,-1,-1]),np.array([1, 1, 1, 1, 1, 1]) # 6
        capsule_pos_low, capsule_pos_high = np.array([-2,-2,-2]),np.array([2, 2, 2]) # 3
        capsule_quat_low, capsule_quat_high = np.array([-1,-1,-1,-1]),np.array([1, 1, 1, 1]) # 4
        capsule_linear_vel_low, capsule_linear_vel_high = np.array([-1,-1,-1]),np.array([1, 1, 1]) # 3
        capsule_pos_relative_low, capsule_pos_relative_high = np.array([-2,-2,-2]),np.array([2, 2, 2]) # 3
        mc_hat_low, mc_hat_high = np.array([-1,-1,-1]),np.array([1, 1, 1]) # 3
        target_pos_low, target_pos_high = np.array([-2,-2,-2]),np.array([2, 2, 2]) # 3
        target_vel_low, target_vel_high = np.array([-1,-1,-1]),np.array([1, 1, 1]) # 3
        target_point_low, target_point_high = np.array([-1,-1,-1]),np.array([1, 1, 1]) # 3

        #low = np.concatenate([q_pos_low, q_vel_low, eef_pos_low, eef_quat_low, eef_vel_low, capsule_pos_low, capsule_quat_low, capsule_linear_vel_low, capsule_pos_relative_low, mc_hat_low, target_pos_low, target_vel_low, target_point_low])
        #high = np.concatenate([q_pos_high, q_vel_high, eef_pos_high, eef_quat_high, eef_vel_high, capsule_pos_high, capsule_quat_high, capsule_linear_vel_high, capsule_pos_relative_high, mc_hat_high, target_pos_high, target_vel_high, target_point_high])
        low = np.concatenate([q_pos_low,  eef_pos_low,  target_pos_low])
        high = np.concatenate([q_pos_high,  eef_pos_high,  target_pos_high])

        self.observation_space = gym.spaces.box.Box(    #change later
            low = low,
            high = high
        )
        #self.reset()

    def reset(self, seed=None, **options):


        self._p.resetSimulation()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.807)
        self.planeID = self._p.loadURDF("plane.urdf")

   

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
        self.init_capsule()

        self.done = False
        self.terminated = False
        self.truncated = False

        self._p.addUserDebugLine([0, 0, 0], [2, 0, 0], [1, 0, 0], lineWidth=2,lifeTime=0)
        self._p.addUserDebugLine([0, 0, 0], [0, 2, 0], [0, 1, 0], lineWidth=2,lifeTime=0)
        self._p.addUserDebugLine([0, 0, 0], [0, 0, 2], [0, 0, 1], lineWidth=2,lifeTime=0)

        self.magnet_init(82.5, 0.1664)

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
        
        """q_pos  6"""
        q_pos = self.agent.get_joint_position()
        observation.extend(list(q_pos))
        """q_vel  6"""
        # q_vel = self.agent.get_joint_velocity()
        # observation.extend(list(q_vel))
        # print(q_vel)
        """eef_pos  3"""
        state = self.agent.get_tip_pose()
        tip_pos = state[0]
        observation.extend(list(tip_pos))
        # print(tip_pos)
        """eef_quat  4"""
        # tip_orn = state[1]
        # if self._control_eu_or_quat == 0:
        #     euler = self._p.getEulerFromQuaternion(tip_orn)
        #     observation.extend(list(euler))  # roll, pitch, yaw
        #     #observation_lim.extend(self._eu_lim)
        # else:
        #     observation.extend(list(tip_orn))
            #observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])
        """eef_vel  6"""
        # tip_vel ,tip_angle_vel= self.agent.get_tip_vel()
        # observation.extend(list(tip_vel))
        # observation.extend(list(tip_angle_vel))
        """capsule_pos  3"""
        # obj_pos, obj_ori = self._p.getBasePositionAndOrientation(self.obj)
        # #obj_ori_euler = self._p.getEulerFromQuaternion(obj_ori)
        # observation.extend(list(obj_pos))

        """capsule_quat  4"""
        # observation.extend(list(obj_ori))

        """capsule_linear_vel  3"""
        # obj_vel, _ = self._p.getBaseVelocity(self.obj)
        # observation.extend(list(obj_vel))

        """capsule_pos_relative  3"""
        # observation.extend(list(self.p_delta.squeeze()))

        """mc_hat  3"""
        # observation.extend(list(self.mc_hat.squeeze()))

        """target_pos  3"""
        observation.extend(list(self.target_position))

        # """target_vel  3"""
        # observation.extend(list(self.target_vel))

        # """target_point  3"""
        # observation.extend(list(self.target_point))
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

        self._p.changeDynamics(self.obj, -1, lateralFriction=0.99) #0.05
        self._p.changeDynamics(self.obj, -1, spinningFriction=0.05) #0.05
        self._p.changeDynamics(self.obj, -1, rollingFriction=0.04)
        self._p.changeDynamics(self.obj, -1, restitution=0.00)
        self._p.changeDynamics(self.obj, -1, linearDamping=130) #50可以做到悬浮,last_value 100
        self._p.changeDynamics(self.obj, -1, angularDamping=0.04) #初始0.04
        self._p.changeDynamics(self.obj, -1, frictionAnchor=1)
        self._p.changeDynamics(self.obj, -1, ccdSweptSphereRadius=0.02, contactProcessingThreshold=0.0)


    def init_capsule(self):
        path="Mag_Env/mc_capsule/urdf/mc_capsule.urdf"
        #position = [0.1, 0.0, 1.0316]
        position = [0.1, 0.0, 1.0316]
        rotation = [0.0, 0.0, 0.0, 1.0]
        # | self._p.URDF_USE_INERTIA_FROM_FILE 
        self.obj = self._p.loadURDF(
        # fileName=path,basePosition=position,baseOrientation=rotation,useMaximalCoordinates = True,
        #     flags=self._p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | self._p.URDF_MERGE_FIXED_LINKS )
        fileName=path,basePosition=position,baseOrientation=rotation)
        self._p.changeDynamics(self.obj, -1, lateralFriction=1) #0.05
        self._p.changeDynamics(self.obj, -1, spinningFriction=0.02) #0.05
        self._p.changeDynamics(self.obj, -1, rollingFriction=0.01)
        self._p.changeDynamics(self.obj, -1, restitution=0.0)
        self._p.changeDynamics(self.obj, -1, linearDamping=15) #50可以做到悬浮,last_value 100
        self._p.changeDynamics(self.obj, -1, angularDamping=5) #初始0.04
        # self._p.changeDynamics(self.obj, -1, contactStiffness=1e3, contactDamping=1e2) 


        self._p.changeDynamics(self.env_dict["table"], -1, lateralFriction=1, spinningFriction=0.02, rollingFriction=0.01, restitution=0.0)
        # self._p.changeDynamics(self.obj, -1, frictionAnchor=1)

        info = self._p.getDynamicsInfo(self.obj, -1)
        #print(info)

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
        return 3 * p_delta_hat_outer - I
    

    def magnet_update(self):
        self.ma_position, self.ma_orientation = self.agent.get_tip_pose()
        self.mc_position, self.mc_orientation = self._p.getBasePositionAndOrientation(self.obj)
        
        self.ma_o_Matrix = np.array(self._p.getMatrixFromQuaternion(self.ma_orientation)).reshape(3, 3)
        self.ma_hat = np.array(self.ma_o_Matrix[:, 2]).reshape(3,1)

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


        total_force = self.get_magnetic_force() + self.get_buoyancy_force()
        return total_force

    def get_magnetic_torque(self):
        
        #magnetic_torque = u0 * self.ma_norm * self.mc_norm / (4 * np.pi * (self.p_delta_norm ** 3)) * np.dot(np.cross(self.mc_hat, self.D()), self.ma_hat)
        # print()
        # print(np.dot(self.D(), self.ma_hat))
        a = np.squeeze(self.mc_hat)
        b = np.squeeze(np.dot(self.D(), self.ma_hat))
        c = np.cross(a,b).reshape(3,1)
        magnetic_torque = u0 * self.ma_norm * self.mc_norm / (4 * np.pi * (self.p_delta_norm ** 3)) * c
        #print(c)
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
    
    def step(self,action):
        
        # if self._render_sleep:
        #     time_spent = time.time() - self._last_frame_time
        #     self._last_frame_time = time.time()
        #     time_to_sleep = self._action_repeat * self._time_step - time_spent
        #     if time_to_sleep > 0:
        #         time.sleep(time_to_sleep)

        self.magnet_update()
        f = self.total_force()
        tau = self.get_magnetic_torque()
        ma_pos, ma_orn= self.ma_position , self.ma_o_Matrix
        mc_pos, mc_orn= self.mc_position , self.mc_o_Matrix
        # tau = np.around(tau, decimals=4)
        # tau = np.array([0, 0, 0])
        self._p.applyExternalForce(self.obj, -1, f.flatten(), mc_pos, self._p.WORLD_FRAME)
        self._p.applyExternalTorque(self.obj, -1,tau.flatten(),self._p.WORLD_FRAME)

        reward = 0


        state_object, _ = self._p.getBasePositionAndOrientation(self.obj)
        state_position_tip, _ = self.agent.get_tip_pose()

        joint_indices = [0,1,2,3,4,5]

        dv = [0.2,0.2,0.2,0.2,0.2,0.2]
        #=========================================================================#
        #  Execute Actions                                                        #
        #=========================================================================#

        # METHOD 1: action = dx,dy,dz in cartesian space

         # default: 0.005, how big are the actions
        # dx = action[0] * dv[0]
        # dy = action[1] * dv[1]
        # dz = action[2] * dv[2]
    
    
        # currentPosition , currentPose = self.agent.get_tip_pose()
    
        # newPosition = [currentPosition[0] + dx,
        #                 currentPosition[1] + dy,
        #                 currentPosition[2] + dz]
    
        # currentPose_Euler = self._p.getEulerFromQuaternion(currentPose)
    
        # dyaw = action[3] * dv[3]
        # dpitch = action[4] * dv[4]
        # droll = action[5] * dv[5]
    
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
            
        dq = self.damped_least_squares_ik(self.agent.arm, 6, joint_indices, action)
        
        joint_states = self._p.getJointStates(self.agent.arm, joint_indices)
        joint_positions = [state[0] for state in joint_states]
        
        joint_target_pos = joint_positions + dq * dv
        

        self.agent.set_joint_position(joint_target_pos)

        # print("目标关节位置:", joint_target_pos)


        # METHOD 4: 速度控制
        # cmd_limit = [1.56, 1.56, 1.56, 3.14, 3.14, 3.14]
        # dq = np.array(action) * cmd_limit
        # self.agent.apply_action(dq, self.mode,torque_sens=None,pos_sens=None, vel_sens=1,P_mx_fr=None)

        for i in range(4):
            self._p.stepSimulation()
            time.sleep(self.timeStep)
        
        


    # =========================================================================#
    #  Reward Function and Episode End States                                 #
    # =========================================================================#

        state_object, _ = self._p.getBasePositionAndOrientation(self.obj)
        state_position_tip, _ = self.agent.get_tip_pose()
    # Dense Reward:
        d_scale = 1.5
        contact_scale = 0.5
        ori_scale = 0.1

        result1 = []
        # di+1物体目标位置与当前位置的欧式距离
        for i in range(len(self.target_position)):
            result1.append(pow(self.target_position[i] - state_object[i],2))
        d = np.sqrt(np.sum(result1))
        d_reward = 1 - np.tanh(10.0 * (d) / 3.0)

        

        
        v, _ = self._p.getBaseVelocity(self.obj)
        v_reward = 1 - np.tanh(10.0 * (np.array(v)) / 3.0)

        # 是否ma，mc吸在一起
        contacts = self._p.getContactPoints(self.obj, self.agent.arm)
        if contacts:
            self.truncated = True

        # 是否和桌面接触
        contact1 = self._p.getContactPoints(self.env_dict["table"], self.obj)
        if contact1:
            contact1_reward = -0.5
        else:
            contact1_reward = 0

        point_delta = np.linalg.norm(self.mc_hat - self.target_point)
        p_reward = 1 - np.tanh(10.0 * (point_delta) / 3)


        reach_reward = d_scale * d_reward  + contact_scale * contact1_reward + ori_scale * p_reward
        # 限制ma和mc的距离
        # d2 = abs(state_position_tip[2] - state_object[2])

        # if d2 < 0.1:
        #     r4 = -1
        # else:
        #     r4 = 0




        # 稀疏奖励
        if d < 0.05 and contact1 == 0:
            keep_reward = v_reward
        else:
            keep_reward = 0

        reward = reach_reward + keep_reward

        #print(state_position_tip)
       
        # End episode
        self.step_counter += 1
        if self.step_counter > MAX_EPISODE_LEN :

            self.terminated = True
        # print("REWARD: ",reward)
        self.obs = self.get_observation()
        
        #scaled_obs = utils.scale_gym_data(self.observation_space, obs)
        self.info = {"reward":reward}
        # print("obs:", self.obs)
        return np.array(self.obs).astype(np.float32), reward, self.terminated, self.truncated, self.info


    def test_step(self):
        
        self.starttime = time.time()
        self.magnet_update()
        f = self.total_force()
        tau = self.get_magnetic_torque()
        ma_pos, ma_orn= self.ma_position , self.ma_o_Matrix
        mc_pos, mc_orn= self.mc_position , self.mc_o_Matrix
        # tau = np.around(tau, decimals=4)
        # tau = np.array([0, 0, 0])
        self._p.applyExternalForce(self.obj, -1, f.flatten(), mc_pos, self._p.WORLD_FRAME)
        self._p.applyExternalTorque(self.obj, -1,tau.flatten(),self._p.WORLD_FRAME)
        
        # print(f)
        axis_length = 0.2
        # """ma画坐标轴"""
        # ma_pos_x_end = ma_pos + axis_length*ma_orn[:,0].flatten()
        # ma_pos_y_end = ma_pos + axis_length*ma_orn[:,1].flatten()
        # ma_pos_z_end = ma_pos + axis_length*ma_orn[:,2].flatten()
        # self._p.addUserDebugLine(ma_pos, ma_pos_x_end, [1, 0, 0], lineWidth=2,lifeTime=0.09)
        # self._p.addUserDebugLine(ma_pos, ma_pos_y_end, [0, 1, 0], lineWidth=2,lifeTime=0.09)
        # self._p.addUserDebugLine(ma_pos, ma_pos_z_end, [0, 0, 1], lineWidth=2,lifeTime=0.09)
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
        # # if (self.step_counter % 2 == 0):
        # #     current_time = time.time()
        # #     if self.last_command_time is not None:
        # #         interval = current_time - self.last_command_time
        # #         print(f"间隔时间: {interval:.6f} 秒")
        # #     self.last_command_time = current_time

        # self.agent.set_cartesian_position(position=newPosition, orientation=newOrientation)
            
        """手写阻尼IK"""
        jointindices = [0,1,2,3,4,5]

        
        d_action = [0,0,-0.003,0,0,0]
        dq = self.damped_least_squares_ik(self.agent.arm, 6, jointindices, d_action)
        
        joint_states = self._p.getJointStates(self.agent.arm, jointindices)
        joint_positions = [state[0] for state in joint_states]
        
        joint_target_pos = joint_positions + dq
        

        self.agent.set_joint_position(joint_target_pos)

        # print("目标关节位置:", joint_target_pos)

        """速度控制"""

        # cmd_limit = [1.56, 1.56, 1.56, 3.14, 3.14, 3.14]
        # dq = np.array(action) * cmd_limit
        # self.agent.apply_action(dq, self.mode,vel_sens=1)

        # 30Hz
        for i in range(8):
            self._p.stepSimulation()
            
        self.step_counter += 1
        

        




    def render(self, mode='human'):
        pass


    def seed(self,seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        self._p.disconnect()



