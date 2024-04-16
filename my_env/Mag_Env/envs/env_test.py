import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R
from . import utils
from . import manipulators

from gymnasium.utils import seeding
import gymnasium as gym

u0 = 4 * np.pi * 1e-7
MAX_EPISODE_LEN = 1000

# gui = 1,direct = 0


DIM_OBS = 6  # no. of dimensions in observation space
DIM_ACT = 6  # no. of dimensions in action space
target_position = (1.1, 0.0, 0.55)


class MagnetEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gui=0):

        self._p = utils.connect(gui)
        self._p.setRealTimeSimulation(0)
        self.step_counter = 0
        self._control_eu_or_quat = 0
        self._include_vel_obs = 1

        # limit x_min x_max,y_min y_max,z_min z_max
        self._workspace_lim = [[0.5, 1.3], [-0.4, 0.4], [0.4, 1.2]]
        self._eu_lim = [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]
        self.action_space = gym.spaces.Box(np.array([-1]*DIM_ACT), np.array([1]*DIM_ACT))
        self.observation_space = gym.spaces.Box(np.array([-1]*DIM_OBS), np.array([1]*DIM_OBS))

        #self.reset()

    def reset(self, seed=None):

        self._p.resetSimulation()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.807)
        self._p.loadURDF("plane.urdf")

        self.env_dict = utils.create_tabletop(self._p)
        self.agent = manipulators.Manipulator(self._p, path="../ur10e/ur10e.urdf", position=[0., 0., 0.4], ik_idx=6)
        base_constraint = self._p.createConstraint(parentBodyUniqueId=self.env_dict["base"], parentLinkIndex=0,
                                                   childBodyUniqueId=self.agent.id, childLinkIndex=-1,
                                                   jointType=self._p.JOINT_FIXED, jointAxis=(0, 0, 0),
                                                   parentFramePosition=(0, 0, 0),
                                                   childFramePosition=(0.0, 0.0, -0.2),
                                                   childFrameOrientation=(0, 0, 0, 1))
        self._p.changeConstraint(base_constraint, maxForce=10000)

        # 创建工具和约束
        self.tool_id = utils.create_object(self._p, self._p.GEOM_CYLINDER, size=[0.025, 0.05], position=[0, 0, 0],
                                           mass=1.0, color=[0.5, 0.2, 0.5, 1.0])

        tool_constraint = self._p.createConstraint(parentBodyUniqueId=self.agent.id, parentLinkIndex=6,
                                                   childBodyUniqueId=self.tool_id, childLinkIndex=-1,
                                                   jointType=self._p.JOINT_FIXED, jointAxis=(0, 0, 0),
                                                   parentFramePosition=(0, 0, 0), childFramePosition=(0, 0, 0.03),
                                                   childFrameOrientation=self._p.getQuaternionFromEuler(
                                                       [-np.pi / 2, 0, 0]))
        self._p.changeConstraint(tool_constraint, maxForce=10000)

        num_joints = 6


        self.step_counter = 0
        self.obj = None

        self.init_agent_pose(t=1)

        self.init_objects()



        self.magnet_init(150, 10)
        # self._step(40)

        state_object, _ = self._p.getBasePositionAndOrientation(self.obj)
        self.observation = state_object + target_position
        #print(self.observation)

        info = {self.observation}
        return np.array(self.observation).astype(np.float32) ,info

    def init_agent_pose(self, t=None, sleep=False, traj=False):
        angles = [-0.294, -1.650, 2.141, -2.062, -1.572, 1.277]
        self.agent.set_joint_position(angles, t=t, sleep=sleep, traj=traj)


    def init_objects(self):
        obj_type = self._p.GEOM_CYLINDER
        position = [0.8, 0.0, 0.5]
        rotation = [0, 0, 0]
        if obj_type == self._p.GEOM_CYLINDER:
            r = 0.0125
            h = 0.025
            size = [r, h]
            # rotation = [np.pi/2, 0, np.pi]
            rotation = [0, np.pi, 0]
        else:
            r = np.random.uniform(0.025, 0.05)
            size = [r, r, r]

        self.obj = utils.create_object(self._p, obj_type=obj_type, size=size, position=position,
                                       rotation=rotation, color=[0.5, 0, 0.5, 1], mass=0.00156)

        self._p.changeDynamics(self.obj, -1, lateralFriction=0.05)
        self._p.changeDynamics(self.obj, -1, spinningFriction=0.05)
        self._p.changeDynamics(self.obj, -1, rollingFriction=0.04)
        self._p.changeDynamics(self.obj, -1, restitution=0.00)
        self._p.changeDynamics(self.obj, -1, linearDamping=130)  # 50可以做到悬浮,last_value 100
        self._p.changeDynamics(self.obj, -1, angularDamping=30)  # 初始0.04
        self._p.changeDynamics(self.obj, -1, frictionAnchor=1)

    def magnet_init(self, ma_norm, mc_norm):
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
        self.ma_position, self.ma_orientation = self._p.getBasePositionAndOrientation(self.tool_id)
        self.mc_position, self.mc_orientation = self._p.getBasePositionAndOrientation(self.obj)

        ma_o_Matrix = np.array(self._p.getMatrixFromQuaternion(self.ma_orientation)).reshape(3, 3)
        self.ma_hat = np.array(ma_o_Matrix[:, 2]).reshape(3, 1)
        mc_o_Matrix = np.array(self._p.getMatrixFromQuaternion(self.mc_orientation)).reshape(3, 3)
        self.mc_hat = np.array(mc_o_Matrix[:, 2]).reshape(3, 1)
        # print("ma_hat",self.ma_hat)
        # print("mc_hat", self.mc_hat)
        self.p_delta = np.array(self.mc_position).reshape(3, 1) - np.array(self.ma_position).reshape(3, 1)
        self.p_delta_norm = np.linalg.norm(self.p_delta)
        self.p_delta_hat = self.p_delta / self.p_delta_norm if self.p_delta_norm != 0 else np.zeros_like(self.p_delta)
        # print("mc_o_Matrix:", mc_o_Matrix)
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

        magnetic_force = 3 * u0 * self.ma_norm * self.mc_norm / (4 * np.pi * (self.p_delta_norm ** 4)) * np.dot((np.outer(self.ma_hat,self.mc_hat) + np.outer(self.mc_hat,self.ma_hat) + (np.dot(np.dot(self.mc_hat.T,self.Z()),self.ma_hat)) * I),self.p_delta_hat)
        # magnetic_force = (3 * u0 * self.ma_norm * self.mc_norm / (4 * np.pi * (self.p_delta_norm ** 4)) *
        #      np.dot((np.outer(self.ma_hat, self.mc_hat) + np.outer(self.mc_hat, self.ma_hat) +
        #              ((self.mc_hat.T.dot(self.Z())).dot(self.ma_hat)) * I), self.p_delta_hat))

        magnetic_force_norm = np.linalg.norm(magnetic_force)
        magnetic_force_hat = magnetic_force / magnetic_force_norm
        if magnetic_force_norm > 1:
            magnetic_force = 1 * self.ma_hat

        return magnetic_force

    def get_buoyancy_force(self):

        buoyancy_force = np.array([0, 0, 0.0148]).reshape(3, 1)  # N
        return buoyancy_force
        #

    def total_force(self):

        total_force = self.get_magnetic_force() + self.get_buoyancy_force()
        return total_force

    def get_magnetic_torque(self):
        magnetic_torque = u0 * self.ma_norm * self.mc_norm / (4 * np.pi * (self.p_delta_norm ** 3)) * np.dot(
            np.multiply(self.mc_hat, self.D()), self.ma_hat)
        # magnetic_torque = u0 * self.ma_norm * self.mc_norm / (4 * np.pi * (self.p_delta_norm ** 3)) * np.multiply(self.mc_hat, np.dot(self.D(), self.ma_hat))
        return magnetic_torque

    def step(self, count=1, action=None, sleep=False, test=False):

        f = self.total_force()
        self._p.applyExternalForce(self.obj, -1, f.flatten(), [0, 0, 0], self._p.LINK_FRAME)
        self._p.resetBasePositionAndOrientation(self.obj, self.mc_position, self.target_mc_Orientation_quaternion)

        """test code"""
        # t = self.get_magnetic_torque()
        # self._p.applyExternalTorque(self.obj, -1, t.flatten(),  self._p.WORLD_FRAME)
        # _, mco = self._p.getBasePositionAndOrientation(self.obj)

        # _, mao = self._p.getBasePositionAndOrientation(self.tool_id)
        # mco = self._p.getMatrixFromQuaternion(mco)
        # print("f",f)
        # print("t:",t)
        # print("mco:",mco)
        # =========================================================================#
        #  Execute Actions                                                        #
        # =========================================================================#

        # METHOD 1: action = dx,dy,dz in cartesian space
        # if action is not None:
        #     dv = 0.005 # default: 0.005, how big are the actions
        #     dx = action[0] * dv
        #     dy = action[1] * dv
        #     dz = action[2] * dv
        #
        #
        #     currentPosition , currentPose = self.agent.get_tip_pose()
        #
        #     newPosition = [currentPosition[0] + dx,
        #                    currentPosition[1] + dy,
        #                    currentPosition[2] + dz]
        #
        #     currentPose_Euler = self._p.getEulerFromQuaternion(currentPose)
        #
        #     dyaw = action[3] * dv
        #     dpitch = action[4] * dv
        #     droll = action[5] * dv
        #
        #     newPose_Euler = [currentPose_Euler[0] + dyaw,
        #                currentPose_Euler[1] + dpitch,
        #                currentPose_Euler[2] + droll]
        #
        #     newPose = self._p.getQuaternionFromEuler(newPose_Euler)
        #     self.agent.set_cartesian_position(position=newPosition, orientation=newPose)

        # METHOD 2: action = delta_q
        # Get the current joint angles
        if action is not None:
            joint_angles = self.agent.get_joint_position()

            # Apply the delta_q values from the action vector
            dv = 0.02  # how big are the actions
            joint_angles = [a + action[i] * dv for i, a in enumerate(joint_angles)]

            self._p.setJointMotorControlArray(self.agent.id, list(range(6)), self._p.POSITION_CONTROL, joint_angles)

            for _ in range(count):
                self._p.stepSimulation()

            state_object, _ = self._p.getBasePositionAndOrientation(self.obj)
            state_position_tip, _ = self._p.getBasePositionAndOrientation(self.tool_id)

        if test is True:
            for _ in range(count):
                self._p.stepSimulation()

        self.magnet_update()
        # =========================================================================#
        #  Reward Function and Episode End States                                 #
        # =========================================================================#

        # Dense Reward:
        #
        Q1 = 0.8
        Q2 = 0.3
        Q3 = 0.3
        Q4 = 0.2
        Q5 = 0.6
        reward = 0
        if action is not None:

            done = False

            tip = state_position_tip
            obj = state_object

            # 使物体尽量处于中间位置
            result = [abs(target_position[i] - obj[i]) for i in range(len(target_position))]
            r1 = -sum(result)

            # 限制机械臂移动
            result1 = [abs(action[i]) for i in range(len(action))]
            r2 = -sum(result1)

            # 是否悬浮在固定位置
            if 0.65 > state_object[2] > 0.55:
                r3 = 50
            else:
                r3 = 0

            # End episode
            self.step_counter += 1

            # 是否ma，mc吸在一起
            contacts = self._p.getContactPoints(self.obj, self.tool_id)
            if contacts:
                r4 = -1
                # print("Collision detected!")
                done = True
            else:
                r4 = 0

            # 是否和地面碰撞
            contact1 = self._p.getContactPoints(self.env_dict["table"], self.obj)
            if contact1:
                r5 = -1
            else:
                r5 = 2

            # reward = Q1*r1 + Q3*r3 + Q4*r4
            reward = Q1 * r1 + Q2 * r2 + Q3 * r3 + Q4 * r4 + Q5 * r5

            if self.step_counter > MAX_EPISODE_LEN:
                # reward = 0
                done = True
            # print("REWARD: ",reward)

            self.observation = state_object + target_position

            return np.array(self.observation).astype(np.float32), reward, done, {}

    def render(self, mode='human'):
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.05],
                                                                distance=.7,
                                                                yaw=90,
                                                                pitch=-70,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(960) / 720,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(width=960,
                                                  height=720,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=self._p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def close(self):
        self._p.disconnect()
