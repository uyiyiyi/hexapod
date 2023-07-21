# 本脚本实现了六足机器人前进和转向的功能
import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import math
from kinematic_model import Kinematic
import  sophus as sp
import threading

NUM_LEGS = 6
CYCLE_LENGTH = 100 # 50 time steps
HALF_CYCLE_LENGTH = int(CYCLE_LENGTH/2)
LEG_LIFT_HEIGHT = 0.1
forward_length = 0
turn_radius = 0

def skew(x):
    return np.matrix([[0, -x[2], x[1]],[x[2], 0, -x[0]],[-x[1], x[0], 0]])

def rot_x(theta):
    R = np.matrix([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    return R

def rot_y(theta):
    R = np.matrix([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return R

def rot_z(theta):
    R = np.matrix([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return R

class Hexapod(object):
    def __init__(self) -> None:
        self.physicsClient = p.connect(p.GUI)
        # self.physicsClient = p.connect(p.DIRECT) #p.DIRECT for non-graphical version
        p.setGravity(0,0,-9.8)
        self.timestep = 0.01
        p.setTimeStep(self.timestep, self.physicsClient)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        self.planeId = p.loadURDF("urdf/plane.urdf", baseOrientation=p.getQuaternionFromEuler([0,0,0]))
        # self.blockId = p.loadURDF("urdf/block.urdf", [1,0,0], baseOrientation=p.getQuaternionFromEuler([0,0,0]), useFixedBase=True)
        # self.planeId = p.loadURDF("terrain.urdf", baseOrientation=p.getQuaternionFromEuler([0,0,0]))
        self.hexapod = p.loadURDF("robot_liuzu/urdf/robot_liuzu.urdf",[0,0,0.37], p.getQuaternionFromEuler([0,0,np.pi/2]))
        p.setJointMotorControlArray(self.hexapod, [i for i in range(18)], p.POSITION_CONTROL, [0.0] * 18)
        self.v_debugId = p.addUserDebugParameter("linear_velocity", -0.075, 0.075, 0.0, self.physicsClient)
        self.w_debugId = p.addUserDebugParameter("angular_velocity", -0.262, 0.262, 0.0, self.physicsClient)
        self.rot_x_id = p.addUserDebugParameter('rot_x', -np.pi / 12, np.pi / 12, 0, self.physicsClient)
        self.rot_y_id = p.addUserDebugParameter('rot_y', -np.pi / 12, np.pi / 12, 0, self.physicsClient)
        self.rot_z_id = p.addUserDebugParameter('rot_z', -np.pi / 18, np.pi / 18, 0, self.physicsClient)
        self.body_height_id = p.addUserDebugParameter("height", 0.25, 0.45, 0.3754, self.physicsClient)
        self.cycle_leg_number_ =[0,1,1,0,0,1]
        self.sequenceID = 0
        self.stanceLegID = [i for i, x in enumerate(self.cycle_leg_number_) if x == 1]
        self.swingLegID = [i for i, x in enumerate(self.cycle_leg_number_) if x == 0]
        self.k = Kinematic()
        self.joint_indices = [i for i in range(18)]
        self.joint_positions = [0] * 18
        self.joint_positions_ = [0] * 18
        self.vx = 0
        self.wz = 0
        self.pcd = np.array([[0,0,0.3754]]).transpose()
        # self.pcd_dot = np.array([[0,0,0]]).transpose()
        self.Rd =  np.identity(3)
        self.stride = 0
        self.alpha = 0
        self.height_d = 0.3754
        self.T = 1
        self.Kpp = np.diag([200, 300, 100]) # [[1., 0., 0.], [0., 10., 0.], [0., 0., 10.]]
        self.Kpd = np.diag([0, 0, 0])
        self.Kwp = np.diag([300, 300, 300])
        # self.kwd = np.diag([60, 60, 60])

    def gait(self):
        # p.stepSimulation()
        self.vx = p.readUserDebugParameter(self.v_debugId, self.physicsClient)
        self.wz = p.readUserDebugParameter(self.w_debugId, self.physicsClient)
        Rx = rot_x(p.readUserDebugParameter(self.rot_x_id, self.physicsClient))
        Ry = rot_y(p.readUserDebugParameter(self.rot_y_id, self.physicsClient))
        self.height_d = p.readUserDebugParameter(self.body_height_id, self.physicsClient)
        # 定义步长stride为速度乘以时间，速度vx人工设定，时间T暂定2秒
        self.stride = self.vx * self.T
        # 定义机身绕z轴旋转角度alpha为角速度乘以时间
        self.alpha = self.wz * self.T
        self.pcd[0,0] = self.stride * np.cos(self.alpha)
        self.pcd[1,0] = self.stride * np.sin(self.alpha)
        self.pcd[2,0] = self.height_d
        # print(self.pcd)
        qc = p.getBasePositionAndOrientation(self.hexapod, self.physicsClient)[1]
        yaw = p.getEulerFromQuaternion(qc)[2] - np.pi / 2
        Rz = rot_z(self.alpha + yaw)
        self.Rd = Rz * Ry * Rx
        # 创建两个线程，分别关联两个函数
        thread1 = threading.Thread(target=self.stance_controller)
        thread2 = threading.Thread(target=self.swing_controller)
        # 启动线程
        thread1.start()
        thread2.start()
        # 等待线程执行完毕
        thread1.join()
        thread2.join()
        # Lowering leg
        self.lowering_foot()

    

    def stance_controller(self):
        # 计算支撑腿序号
        self.stanceLegID = [i for i, x in enumerate(self.cycle_leg_number_) if x == 1]
        front_stance_legID = self.stanceLegID[0]
        mid_stance_legID = self.stanceLegID[1]
        back_stance_legID = self.stanceLegID[2]
        # 获取所有关节位置
        for i in range(18):
            self.joint_positions[i] = p.getJointState(self.hexapod, self.joint_indices[i], self.physicsClient)[0]            
        # 计算支撑腿的关节角位置
        front_joint_positions = self.joint_positions[(3 * front_stance_legID): (3 * front_stance_legID + 3)]
        mid_joint_positions = self.joint_positions[(3 * mid_stance_legID): (3 * mid_stance_legID + 3)]
        back_joint_positions = self.joint_positions[(3 * back_stance_legID): (3 * back_stance_legID + 3)]
        # 计算支撑相足端位置
        front_tip = self.k.fk(front_joint_positions, front_stance_legID)
        mid_tip = self.k.fk(mid_joint_positions, mid_stance_legID)
        back_tip = self.k.fk(back_joint_positions, back_stance_legID)
        # 记初始位置为pc0，即[0 ,0 , 机身高度]
        pc0 = [0, 0, -(front_tip[0][2] + mid_tip[0][2] + back_tip[0][2]) / 3]
        pc0 = np.matrix(pc0).transpose()
        # 计算机器人当前位置坐标，O为该步初始时机器人坐标原点位置，O_为当前机器人坐标原点位置，P1为一号腿足端位置（固定于大地坐标系，但相对机器人坐标系运动）
        # 机器人运动的量为OO_ = OP1 - O_P1，为了计算机器人的速度需要知道两个时刻间OO_的差值，因此记上一时刻的OO_为OO__
        # 速度记为vO__O_ = (OO_ - OO__) / timestep
        OP1 = np.matrix(self.k.fk(mid_joint_positions, mid_stance_legID)) # OP1固定不变，作为参考
        O_P1 = np.matrix(mid_tip)
        OO_ = OP1.transpose() - O_P1.transpose()
        # OO__ 初始值为[0, 0, 0]，后面每次执行完电机操作后将OO_赋给OO__
        OO__ = np.matrix([0, 0, 0])
        # print("OP1", OP1)
        # print("O_P1", O_P1)
        # print("OO_", OO_)
        vO__O_ = (OO_ - OO__) / self.timestep
        # 计算机器人姿态仿真中可直接读得，实际中需要从IMU获取
        qc = p.getBasePositionAndOrientation(self.hexapod, self.physicsClient)[1]
        Rot = np.matrix(p.getMatrixFromQuaternion(qc)).reshape((3,3)) * rot_z(-np.pi / 2)
        # 计算机器人位置误差
        pos_error = self.pcd - np.array([[OO_[0, 0], OO_[1, 0], -(front_tip[0][2] + mid_tip[0][2] + back_tip[0][2]) / 3]]).transpose()
        error = np.array([pos_error[0, 0], pos_error[2, 0]])
        pos_error_norm = np.linalg.norm(error)
        # pos_error_norm = np.linalg.norm(pos_error)
        # 计算机器人姿态误差
        R_so3 = np.matrix(sp.SO3(self.Rd * Rot.transpose()).log()).transpose()
        R_so3_error_norm = np.linalg.norm(R_so3)
        while pos_error_norm > 0.02 or R_so3_error_norm > 0.1:
            print("pos_error", pos_error)
            print("R_so3", R_so3)
            # print("pos_error_norm", pos_error_norm)
            # print("R_so3_norm", R_so3_error_norm)
            # 位置PD控制，姿态P控制
            v_base_desired = np.dot(self.Kpp, pos_error) + np.dot(self.Kpd, vO__O_)
            w_base_desired = np.dot(self.Kwp, R_so3)
            # print("v_base_desired", v_base_desired)
            # print("w_base_desired", w_base_desired)
            r1 = np.matrix(front_tip)
            r2 = np.matrix(mid_tip)
            r3 = np.matrix(back_tip)
            v_front_tip = -v_base_desired.transpose() + np.cross(w_base_desired.transpose(), -r1)
            w_front_tip = -w_base_desired.transpose()
            V_front = np.concatenate((v_front_tip[0], w_front_tip[0]), axis=1)
            v_mid_tip = -v_base_desired.transpose() + np.cross(w_base_desired.transpose(), -r2)
            w_mid_tip = -w_base_desired.transpose()
            V_mid = np.concatenate((v_mid_tip[0], w_mid_tip[0]), axis=1)
            v_back_tip = -v_base_desired.transpose() + np.cross(w_base_desired.transpose(), -r3)
            w_back_tip = -w_base_desired.transpose()
            V_back = np.concatenate((v_back_tip[0], w_back_tip[0]), axis=1)
            # 雅可比矩阵
            front_jacobian = self.k.jacobian(front_joint_positions, front_stance_legID)
            mid_jacobian = self.k.jacobian(mid_joint_positions, mid_stance_legID)
            back_jacobian = self.k.jacobian(back_joint_positions, back_stance_legID)
            # 计算雅可比矩阵的逆
            front_jacobian_inverse = np.linalg.pinv(front_jacobian)
            mid_jacobian_inverse = np.linalg.pinv(mid_jacobian)
            back_jacobian_inverse = np.linalg.pinv(back_jacobian)
            # 计算关节速度
            front_v_theta = (front_jacobian_inverse[0:3, 0:3] * v_front_tip.transpose()).transpose()[0, 0:3]
            mid_v_theta = (mid_jacobian_inverse[0:3, 0:3] * v_mid_tip.transpose()).transpose()[0, 0:3]
            back_v_theta = (back_jacobian_inverse[0:3, 0:3] * v_back_tip.transpose()).transpose()[0, 0:3]
            # front_v_theta = (front_jacobian_inverse * V_front.transpose()).transpose()[0, 0:3]
            # mid_v_theta = (mid_jacobian_inverse * V_mid.transpose()).transpose()[0, 0:3]
            # back_v_theta = (back_jacobian_inverse * V_back.transpose()).transpose()[0, 0:3]
            # 限幅范围
            theta_lower_bound = np.array([-3, -10, -10])
            theta_upper_bound = np.array([3, 10, 10])
            # 对矩阵进行限幅
            # front_v_theta = np.clip(front_v_theta, theta_lower_bound, theta_upper_bound)
            # mid_v_theta = np.clip(mid_v_theta, theta_lower_bound, theta_upper_bound)
            # back_v_theta = np.clip(back_v_theta, theta_lower_bound, theta_upper_bound)
            # print("front_v_theta", front_v_theta)
            # print("mid_v_theta", mid_v_theta)
            # print("back_v_theta", back_v_theta)
            front_delta = front_v_theta * self.timestep
            mid_delta = mid_v_theta * self.timestep
            back_delta = back_v_theta * self.timestep
            # print("front_delta:", front_delta)
            # print("mid_delta:", mid_delta)
            # print("back_delta:", back_delta)
            front_joint_positions_ = (np.asarray(front_joint_positions) + np.asarray(front_delta)).tolist()[0]
            mid_joint_positions_ = (np.asarray(mid_joint_positions) + np.asarray(mid_delta)).tolist()[0]
            back_joint_positions_ = (np.asarray(back_joint_positions) + np.asarray(back_delta)).tolist()[0]

            self.front_tip_estimate = self.k.fk(front_joint_positions_, front_stance_legID)
            self.mid_tip_estimate = self.k.fk(mid_joint_positions_, mid_stance_legID)
            self.back_tip_estimate = self.k.fk(back_joint_positions_, back_stance_legID)
            valid_front_joint_positions, valid_mid_joint_positions, valid_back_joint_positions = self.checkvalidity()
            
            stance_joint_positions_ = [0] * 9
            stance_joint_positions_[0:3] = valid_front_joint_positions
            stance_joint_positions_[3:6] = valid_mid_joint_positions
            stance_joint_positions_[6:9] = valid_back_joint_positions
            stance_jointIDs = [0] * 9
            stance_jointIDs[0:3] = [3 * front_stance_legID + i for i in range(3)]
            stance_jointIDs[3:6] = [3 * mid_stance_legID + i for i in range(3)]
            stance_jointIDs[6:9] = [3 * back_stance_legID + i for i in range(3)]
            p.setJointMotorControlArray(robot.hexapod, stance_jointIDs, p.POSITION_CONTROL, stance_joint_positions_)
            p.stepSimulation()
            time.sleep(0.01)
            # 更新状态和误差
            self.wz = p.readUserDebugParameter(self.w_debugId, self.physicsClient)
            Rx = rot_x(p.readUserDebugParameter(self.rot_x_id, self.physicsClient))
            Ry = rot_y(p.readUserDebugParameter(self.rot_y_id, self.physicsClient))
            Rz = rot_z(self.wz)
            self.height_d = p.readUserDebugParameter(self.body_height_id, self.physicsClient)
            qc = p.getBasePositionAndOrientation(self.hexapod, self.physicsClient)[1]
            # yaw = p.getEulerFromQuaternion(qc)[2] - np.pi / 2
            # Rz = rot_z(self.alpha + yaw)
            # self.Rd = Rz * Ry * Rx
            self.pcd[2, 0] = self.height_d
            for i in range(18):
                self.joint_positions[i] = p.getJointState(self.hexapod, self.joint_indices[i], self.physicsClient)[0]
            front_joint_positions = self.joint_positions[(3 * front_stance_legID): (3 * front_stance_legID + 3)]
            mid_joint_positions = self.joint_positions[(3 * mid_stance_legID): (3 * mid_stance_legID + 3)]
            back_joint_positions = self.joint_positions[(3 * back_stance_legID): (3 * back_stance_legID + 3)]
            front_tip = self.k.fk(front_joint_positions, front_stance_legID)
            mid_tip = self.k.fk(mid_joint_positions, mid_stance_legID)
            back_tip = self.k.fk(back_joint_positions, back_stance_legID)

            # 将旧的OO_赋给OO__
            OO__ = OO_
            # 计算新的OO_
            O_P1 = np.matrix(mid_tip)
            OO_ = OP1.transpose() - O_P1.transpose()
            # 计算VO__O_
            vO__O_ = (OO_ - OO__) / self.timestep

            qc = p.getBasePositionAndOrientation(self.hexapod, self.physicsClient)[1]
            Rot = np.matrix(p.getMatrixFromQuaternion(qc)).reshape((3,3)) * rot_z(-np.pi / 2)
            # 计算机器人位置误差
            pos_error = self.pcd - np.array([[OO_[0, 0], OO_[1, 0], -(front_tip[0][2] + mid_tip[0][2] + back_tip[0][2]) / 3]]).transpose()
            error = np.array([pos_error[0, 0], pos_error[2, 0]])
            pos_error_norm = np.linalg.norm(error)
            # pos_error_norm = np.linalg.norm(pos_error)
            # 计算机器人姿态误差
            R_so3 = np.matrix(sp.SO3(self.Rd * Rot.transpose()).log()).transpose()
            R_so3_error_norm = np.linalg.norm(R_so3)

            # print("yaw desired = ", p.getEulerFrom)

    def swing_controller(self):
        self.swingLegID = [i for i, x in enumerate(self.cycle_leg_number_) if x == 0]
        body_target = self.k.ik([0.34 + self.stride / 2, -0.3907771798601381, -0.3753941065956188], 0)[0]
        thigh_target = 0.4
        shank_target = 0.6
        front_swing_legID = self.swingLegID[0]
        mid_swing_legID = self.swingLegID[1]
        back_swing_legID = self.swingLegID[2]
        swing_jointIDs = [0] * 9
        swing_jointIDs[0:3] = [3 * front_swing_legID + i for i in range(3)]
        swing_jointIDs[3:6] = [3 * mid_swing_legID + i for i in range(3)]
        swing_jointIDs[6:9] = [3 * back_swing_legID + i for i in range(3)]
        for i in range(18):
            self.joint_positions_[i] = p.getJointState(self.hexapod, self.joint_indices[i], self.physicsClient)[0]
        front_joint_positions = self.joint_positions_[(3 * front_swing_legID): (3 * front_swing_legID + 3)]
        mid_joint_positions = self.joint_positions_[(3 * mid_swing_legID): (3 * mid_swing_legID + 3)]
        back_joint_positions = self.joint_positions_[(3 * back_swing_legID): (3 * back_swing_legID + 3)]

        if front_swing_legID % 2 == 0:
            front_joint_target = [body_target, -thigh_target, shank_target]
        else:
            front_joint_target = [-body_target, thigh_target, -shank_target]
        if mid_swing_legID % 2 == 0:
            mid_joint_target = [body_target, -thigh_target, shank_target]
        else:
            mid_joint_target = [-body_target, thigh_target, -shank_target]
        if back_swing_legID % 2 == 0:
            back_joint_target = [body_target, -thigh_target, shank_target]
        else:
            back_joint_target = [-body_target, thigh_target, -shank_target]
        # Lifting leg
        lift_steps = 20
        for i in range(lift_steps):
            front_body_joint_pos = front_joint_positions[0]
            front_thigh_joint_pos = front_joint_positions[1] + (front_joint_target[1] - front_joint_positions[1]) / lift_steps * (i + 1)
            front_shank_joint_pos = front_joint_positions[2] + (front_joint_target[2] - front_joint_positions[2]) / lift_steps * (i + 1)
            mid_body_joint_pos = mid_joint_positions[0]
            mid_thigh_joint_pos = mid_joint_positions[1] + (mid_joint_target[1] - mid_joint_positions[1]) / lift_steps * (i + 1)
            mid_shank_joint_pos = mid_joint_positions[2] + (mid_joint_target[2] - mid_joint_positions[2]) / lift_steps * (i + 1)
            back_body_joint_pos = back_joint_positions[0]
            back_thigh_joint_pos = back_joint_positions[1] + (back_joint_target[1] - back_joint_positions[1]) / lift_steps * (i + 1)
            back_shank_joint_pos = back_joint_positions[2] + (back_joint_target[2] - back_joint_positions[2]) / lift_steps * (i + 1)
            swing_joint_positions_ = [front_body_joint_pos, front_thigh_joint_pos, front_shank_joint_pos, mid_body_joint_pos, mid_thigh_joint_pos, mid_shank_joint_pos, back_body_joint_pos, back_thigh_joint_pos, back_shank_joint_pos]
            p.setJointMotorControlArray(self.hexapod, swing_jointIDs, p.POSITION_CONTROL, swing_joint_positions_)
            p.stepSimulation()
            time.sleep(0.01)
        # Swinging leg
        swing_steps = 80
        for i in range(swing_steps):
            front_body_joint_pos = front_joint_positions[0] + (front_joint_target[0] - front_joint_positions[0]) / swing_steps * (i + 1)
            front_thigh_joint_pos = front_joint_target[1]
            front_shank_joint_pos = front_joint_target[2]
            mid_body_joint_pos = mid_joint_positions[0] + (mid_joint_target[0] - mid_joint_positions[0]) / swing_steps * (i + 1)
            mid_thigh_joint_pos = mid_joint_target[1]
            mid_shank_joint_pos = mid_joint_target[2]
            back_body_joint_pos = back_joint_positions[0] + (back_joint_target[0] - back_joint_positions[0]) / swing_steps * (i + 1)
            back_thigh_joint_pos = back_joint_target[1]
            back_shank_joint_pos = back_joint_target[2]
            swing_joint_positions_ = [front_body_joint_pos, front_thigh_joint_pos, front_shank_joint_pos, mid_body_joint_pos, mid_thigh_joint_pos, mid_shank_joint_pos, back_body_joint_pos, back_thigh_joint_pos, back_shank_joint_pos]
            p.setJointMotorControlArray(self.hexapod, swing_jointIDs, p.POSITION_CONTROL, swing_joint_positions_)
            p.stepSimulation()
            time.sleep(0.01)
    
    def lowering_foot(self):
        self.swingLegID = [i for i, x in enumerate(self.cycle_leg_number_) if x == 0]
        body_target = self.k.ik([0.34 + self.stride / 2, -0.3907771798601381, -0.3753941065956188], 0)[0]
        thigh_target = 0.4
        shank_target = 0.6
        front_swing_legID = self.swingLegID[0]
        mid_swing_legID = self.swingLegID[1]
        back_swing_legID = self.swingLegID[2]
        swing_jointIDs = [0] * 9
        swing_jointIDs[0:3] = [3 * front_swing_legID + i for i in range(3)]
        swing_jointIDs[3:6] = [3 * mid_swing_legID + i for i in range(3)]
        swing_jointIDs[6:9] = [3 * back_swing_legID + i for i in range(3)]
        for i in range(18):
            self.joint_positions_[i] = p.getJointState(self.hexapod, self.joint_indices[i], self.physicsClient)[0]
        front_joint_positions = self.joint_positions_[(3 * front_swing_legID): (3 * front_swing_legID + 3)]
        mid_joint_positions = self.joint_positions_[(3 * mid_swing_legID): (3 * mid_swing_legID + 3)]
        back_joint_positions = self.joint_positions_[(3 * back_swing_legID): (3 * back_swing_legID + 3)]

        if front_swing_legID % 2 == 0:
            front_joint_target = [body_target, -thigh_target, shank_target]
        else:
            front_joint_target = [-body_target, thigh_target, -shank_target]
        if mid_swing_legID % 2 == 0:
            mid_joint_target = [body_target, -thigh_target, shank_target]
        else:
            mid_joint_target = [-body_target, thigh_target, -shank_target]
        if back_swing_legID % 2 == 0:
            back_joint_target = [body_target, -thigh_target, shank_target]
        else:
            back_joint_target = [-body_target, thigh_target, -shank_target]
        lower_steps = 100
        for i in range(lower_steps):
            front_body_joint_pos = front_joint_target[0]
            front_thigh_joint_pos = front_joint_positions[1] - front_joint_positions[1] / lower_steps * (i + 1)
            front_shank_joint_pos = front_joint_positions[2] - front_joint_positions[2] / lower_steps * (i + 1)
            mid_body_joint_pos = mid_joint_target[0]
            mid_thigh_joint_pos = mid_joint_positions[1] - mid_joint_positions[1] / lower_steps * (i + 1)
            mid_shank_joint_pos = mid_joint_positions[2] - mid_joint_positions[2] / lower_steps * (i + 1)
            back_body_joint_pos = back_joint_target[0]
            back_thigh_joint_pos = back_joint_positions[1] - back_joint_positions[1] / lower_steps * (i + 1)
            back_shank_joint_pos = back_joint_positions[2] - back_joint_positions[2] / lower_steps * (i + 1)
            swing_joint_positions_ = [front_body_joint_pos, front_thigh_joint_pos, front_shank_joint_pos, mid_body_joint_pos, mid_thigh_joint_pos, mid_shank_joint_pos, back_body_joint_pos, back_thigh_joint_pos, back_shank_joint_pos]
            p.setJointMotorControlArray(self.hexapod, swing_jointIDs, p.POSITION_CONTROL, swing_joint_positions_)
            p.stepSimulation()
            time.sleep(0.01)

    def checkvalidity(self):
        valid_front_tip = self.front_tip_estimate
        valid_mid_tip = self.mid_tip_estimate
        valid_back_tip = self.back_tip_estimate
        for id in self.stanceLegID:
            if id == 0:
                valid_front_tip[0][0] = np.clip(self.front_tip_estimate[0][0], 0.24, 0.44)
                valid_front_tip[0][1] = np.clip(self.front_tip_estimate[0][1], -0.41, -0.34)
                valid_front_tip[0][2] = np.clip(self.front_tip_estimate[0][2], -0.45, -0.30)
                valid_front_joint_position = self.k.ik(valid_front_tip[0], id)
            if id == 1:
                valid_front_tip[0][0] = np.clip(self.front_tip_estimate[0][0], 0.24, 0.44)
                valid_front_tip[0][1] = np.clip(self.front_tip_estimate[0][1], 0.34, 0.41)
                valid_front_tip[0][2] = np.clip(self.front_tip_estimate[0][2], -0.45, -0.30)
                valid_front_joint_position = self.k.ik(valid_front_tip[0], id)
            if id == 2:
                valid_mid_tip[0][0] = np.clip(self.mid_tip_estimate[0][0], -0.1, 0.1)
                valid_mid_tip[0][1] = np.clip(self.mid_tip_estimate[0][1], -0.41, -0.34)
                valid_mid_tip[0][2] = np.clip(self.mid_tip_estimate[0][2], -0.45, -0.30)
                valid_mid_joint_position = self.k.ik(valid_mid_tip[0], id)
            if id == 3:
                valid_mid_tip[0][0] = np.clip(self.mid_tip_estimate[0][0], -0.1, 0.1)
                valid_mid_tip[0][1] = np.clip(self.mid_tip_estimate[0][1], 0.34, 0.41)
                valid_mid_tip[0][2] = np.clip(self.mid_tip_estimate[0][2], -0.45, -0.30)
                valid_mid_joint_position = self.k.ik(valid_mid_tip[0], id)
            if id == 4:
                valid_back_tip[0][0] = np.clip(self.back_tip_estimate[0][0], -0.44, -0.24)
                valid_back_tip[0][1] = np.clip(self.back_tip_estimate[0][1], -0.41, -0.34)
                valid_back_tip[0][2] = np.clip(self.back_tip_estimate[0][2], -0.45, -0.30)
                valid_back_joint_position = self.k.ik(valid_back_tip[0], id)
            if id == 5:
                valid_back_tip[0][0] = np.clip(self.back_tip_estimate[0][0], -0.44, -0.24)
                valid_back_tip[0][1] = np.clip(self.back_tip_estimate[0][1], 0.34, 0.41)
                valid_back_tip[0][2] = np.clip(self.back_tip_estimate[0][2], -0.45, -0.30)
                valid_back_joint_position = self.k.ik(valid_back_tip[0], id)
        return valid_front_joint_position, valid_mid_joint_position, valid_back_joint_position


def sequence_change(list):
    for i in range(len(list)):
        if list[i] == 0:
            list[i] = 1
        else:
            list[i] = 0

if __name__ == "__main__":
    robot = Hexapod()
    p.stepSimulation()
    robot.vx = p.readUserDebugParameter(robot.v_debugId, robot.physicsClient)
    # sequence_change(robot.cycle_leg_number_)
    # print(robot.cycle_leg_number_)
    # front_swing_legID = robot.swingLegID[0]
    # mid_swing_legID = robot.swingLegID[1]
    # back_swing_legID = robot.swingLegID[2]
    # front_joint_target = [0, -0.4, 0.6]
    # mid_joint_target = [0, 0.4, -0.6]
    # back_joint_target = [0, -0.4, 0.6]
    # swing_joint_positions_ = [0] * 9
    # swing_joint_positions_[0:3] = front_joint_target
    # swing_joint_positions_[3:6] = mid_joint_target
    # swing_joint_positions_[6:9] = back_joint_target
    # swing_jointIDs = [0] * 9
    # swing_jointIDs[0:3] = [3 * front_swing_legID + i for i in range(3)]
    # swing_jointIDs[3:6] = [3 * mid_swing_legID + i for i in range(3)]
    # swing_jointIDs[6:9] = [3 * back_swing_legID + i for i in range(3)]
    # p.setJointMotorControlArray(robot.hexapod, swing_jointIDs, p.POSITION_CONTROL, swing_joint_positions_, )
    # for i in range(40):
    #     p.stepSimulation()
    while True:
        # if robot.vx < 0.01:
        #     p.setJointMotorControlArray(robot.hexapod, [i for i in range(18)], p.POSITION_CONTROL, [0.0] * 18, )
        #     p.stepSimulation()
        # else:
        #     robot.gait()
        #     sequence_change(robot.cycle_leg_number_)
        robot.gait()
        print("change gait")
        sequence_change(robot.cycle_leg_number_)