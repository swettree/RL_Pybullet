import time

import numpy as np
import pybullet as p
# 设置机械臂的相关参数，定义相关函数
class Manipulator:
    initial_positions = {
        'shoulder_pan_joint': -0.294, 'shoulder_lift_joint': -1.650, 'elbow_joint': 2.141,
        'wrist_1_joint': -2.062, 'wrist_2_joint': -1.572, 'wrist_3_joint': 1.277,

    }


    def __init__(self, p, path, position=(0, 0, 0), orientation=(0, 0, 0, 1), ik_idx=-1):
        self._p = p
        # 获取物理引擎的固定时间步长
        #self._timestep = 1/1920
        self._timestep = self._p.getPhysicsEngineParameters()["fixedTimeStep"]
        self._freq = int(1. / self._timestep)
        self.arm = self._p.loadURDF(
            fileName=path,
            basePosition=position,
            baseOrientation=orientation,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES )

        self.ik_idx = ik_idx
        self._joint_name_to_ids = {}
        self.joints = []
        self.names = []
        self.forces = []
        self.max_velocity = []
        self.fixed_joints = []
        self.fixed_names = []
        for i in range(self._p.getNumJoints(self.arm)):
            info = self._p.getJointInfo(self.arm, i)
            if info[2] != self._p.JOINT_FIXED:
                # 关节序号(0 - )
                self.joints.append(i)
                # 关节名称
                self.names.append(info[1])
                # 关节最大力
                self.forces.append(info[10])
                self.max_velocity.append(info[11])
            else:
                self.fixed_joints.append(i)
                self.fixed_names.append(info[1])
        # 转换成元组
        self.joints = tuple(self.joints)
        self.names = tuple(self.names)

        self.num_joints = len(self.joints)
        self.debug_params = []
        self.child = None
        self.constraints = []
        for j in self.joints:
            self._p.enableJointForceTorqueSensor(self.arm, j, 1)

    def apply_action(self, action, mode, torque_sens, vel_sens, pos_sens, P_mx_fr):
        ## CHANGE
        ## Make mode changeable

        if (mode == 'T'):
            mode = p.TORQUE_CONTROL
            action = action * torque_sens
            p.setJointMotorControlArray(self.arm, self.joints, mode, forces=action)
        elif (mode == 'V'):
            mode = p.VELOCITY_CONTROL
            action = action * vel_sens
            p.setJointMotorControlArray(self.arm, self.joints, mode, targetVelocities=action,
                                        )
        elif (mode == 'P'):
            max_force = P_mx_fr * np.ones(6)
            mode = p.POSITION_CONTROL
            action = action * pos_sens
            p.setJointMotorControlArray(self.arm, self.joints, mode, targetPositions=action, forces=max_force,
                                        )

    # 连接工具
    def attach(self, child_body, parent_link, position=(0, 0, 0), orientation=(0, 0, 0, 1), max_force=None):
        self.child = child_body
        constraint_id = self._p.createConstraint(
            parentBodyUniqueId=self.arm,
            parentLinkIndex=parent_link,
            childBodyUniqueId=self.child.id,
            childLinkIndex=-1,
            jointType=self._p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=position,
            childFrameOrientation=orientation)
        if max_force is not None:
            self._p.changeConstraint(constraint_id, maxForce=max_force)
        self.constraints.append(constraint_id)

    # 设置笛卡尔坐标系坐标，末端执行器
    def set_cartesian_position(self, position, orientation=None, t=None, sleep=False, traj=False):
        target_joints = self._p.calculateInverseKinematics(
            bodyUniqueId=self.arm,
            endEffectorLinkIndex=self.ik_idx,
            targetPosition=position,
            targetOrientation=orientation)
        self.set_joint_position(target_joints, t=t, sleep=sleep, traj=traj)

    # 设置笛卡尔坐标系运动轨迹
    def move_in_cartesian(self, position, quaternion=None, t=1.0, sleep=False):
        N = int(t * 240)

        current_position, current_orientation = self.get_tip_pose()

        position_traj = np.linspace(current_position, position, N+1)[1:]

        for p_i in position_traj:
            target_joints = self._p.calculateInverseKinematics(
                bodyUniqueId=self.arm,
                endEffectorLinkIndex=self.ik_idx,
                targetPosition=p_i,
                targetOrientation=quaternion)
            self.set_joint_position(target_joints, t=1/240, sleep=sleep)

    # 设置关节位置

    def get_joint_ranges(self):
        lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []

        for joint_name in self._joint_name_to_ids.keys():
            jointInfo = self._p.getJointInfo(self.arm, self._joint_name_to_ids[joint_name])

            ll, ul = jointInfo[8:10]
            jr = ul - ll
            # For simplicity, assume resting state == initial state
            rp = self.initial_positions[joint_name]
            lower_limits.append(ll)
            upper_limits.append(ul)
            joint_ranges.append(jr)
            rest_poses.append(rp)

        return lower_limits, upper_limits, joint_ranges, rest_poses
    
    def set_joint_position(self, position, velocity=None, t=None, sleep=False, traj=False):
        assert len(self.joints) > 0
        if traj:
            assert (t is not None)
            N = int(t * 240)
            current_position = self.get_joint_position()
            trajectory = np.linspace(current_position, position, N)
            for t_i in trajectory:
                self._p.setJointMotorControlArray(
                    bodyUniqueId=self.arm,
                    jointIndices=self.joints,
                    controlMode=self._p.POSITION_CONTROL,
                    targetPositions=t_i,
                    forces=self.forces)
                self._p.stepSimulation()
                if sleep:
                    time.sleep(self._timestep)

        else:
            if velocity is not None:
                self._p.setJointMotorControlArray(
                    bodyUniqueId=self.arm,
                    jointIndices=self.joints,
                    controlMode=self._p.POSITION_CONTROL,
                    targetPositions=position,
                    targetVelocities=velocity,
                    forces=self.forces)
            else:
                self._p.setJointMotorControlArray(
                    bodyUniqueId=self.arm,
                    jointIndices=self.joints,
                    controlMode=self._p.POSITION_CONTROL,
                    targetPositions=position,
                    forces=self.forces)
            self._waitsleep(t, sleep)

    # 设置关节速度
    def set_joint_velocity(self, velocity, t=None, sleep=False):
        assert len(self.joints) > 0
        self._p.setJointMotorControlArray(
            bodyUniqueId=self.arm,
            jointIndices=self.joints,
            controlMode=self._p.VELOCITY_CONTROL,
            targetVelocities=velocity,
            forces=self.forces)
        self._waitsleep(t, sleep)

    # 设置关节力矩
    def set_joint_torque(self, torque, t=None, sleep=False):
        assert len(self.joints) > 0
        self._p.setJointMotorControlArray(
            bodyUniqueId=self.arm,
            jointIndices=self.joints,
            controlMode=self._p.TORQUE_CONTROL,
            forces=torque)
        self._waitsleep(t, sleep)

    # TODO: make this only joint position, joint velocity etc.
    def get_joint_states(self):
        #返回四个state[位置，速度，关节力，关节力矩]
        return self._p.getJointStates(self.arm, self.joints)

    #得到关节位置和速度
    def get_joint_position(self):
        joint_states = self.get_joint_states()
        return [joint[0] for joint in joint_states]

    def get_joint_velocity(self):
        joint_states = self.get_joint_states()
        return [joint[1] for joint in joint_states]
    # of IK link

    def get_tip_pose(self):
        result = self._p.getLinkState(self.arm, self.ik_idx)
        return result[0], result[1]

    # 设置debug时的按键
    def add_debug_param(self):
        current_angle = [j[0] for j in self.get_joint_states()]
        for i in range(self.num_joints):
            joint_info = self._p.getJointInfo(self.arm, self.joints[i])
            low, high = joint_info[8:10]
            self.debug_params.append(self._p.addUserDebugParameter(self.names[i].decode("utf-8"), low, high, current_angle[i]))

    def update_debug(self):
        target_angles = []
        for param in self.debug_params:
            try:
                angle = self._p.readUserDebugParameter(param)
                target_angles.append(angle)
            except Exception:
                break
        if len(target_angles) == len(self.joints):
            self.set_joint_position(target_angles)

    def _waitsleep(self, t, sleep=False):
        if t is not None:
            iters = int(t*self._freq)
            for _ in range(iters):
                self._p.stepSimulation()
                if sleep:
                    time.sleep(self._timestep)
