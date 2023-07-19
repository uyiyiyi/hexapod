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
        self.hexapod = p.loadURDF("robot_liuzu/urdf/robot_liuzu.urdf",[0,0,0.3755], p.getQuaternionFromEuler([0,0,np.pi/2]))
        p.setJointMotorControlArray(self.hexapod, [i for i in range(18)], p.POSITION_CONTROL, [0.0] * 18)
        self.v_debugId = p.addUserDebugParameter("linear_velocity", 0.001, 0.15, 0.05, self.physicsClient)
        self.w_debugId = p.addUserDebugParameter("angular_velocity", -0.4, 0.4, -0.0001, self.physicsClient)
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
        self.Rd =  np.identity(3)
        self.stride = 0.05
        self.alpha = 0
        self.height_d = 0.3754
        self.kpp = np.diag([100, 200, 150]) # [[1., 0., 0.], [0., 10., 0.], [0., 0., 10.]]
        self.Kpd = np.diag([80, 50, 100])
        self.kwp = np.diag([50, 50, 40])
        self.kwd = np.diag([60, 60, 60])

    def gait(self):
        # p.stepSimulation()
        self.vx = p.readUserDebugParameter(self.v_debugId, self.physicsClient)
        self.wz = p.readUserDebugParameter(self.w_debugId, self.physicsClient)
        Rx = rot_x(p.readUserDebugParameter(self.rot_x_id, self.physicsClient))
        Ry = rot_y(p.readUserDebugParameter(self.rot_y_id, self.physicsClient))
        Rz = rot_z(self.wz)
        self.height_d = p.readUserDebugParameter(self.body_height_id, self.physicsClient)
        self.T = self.stride / self.vx
        self.alpha = self.wz * self.T
        self.pcd[0,0] = self.stride * np.cos(self.wz * self.T)
        self.pcd[1,0] = self.stride * np.sin(self.wz * self.T)
        self.pcd[2,0] = self.height_d
        # print(self.pcd)
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

    def lowering_foot(self):
        self.swingLegID = [i for i, x in enumerate(self.cycle_leg_number_) if x == 0]
        body_target = np.pi / 18
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
            p.setJointMotorControlArray(self.hexapod, swing_jointIDs, p.POSITION_CONTROL, swing_joint_positions_, forces=[200]*9)
            p.stepSimulation()
            time.sleep(0.01)

    def stance_controller(self):
        pos_error_norm = 1
        R_so3_error_norm = 1
        while pos_error_norm > 0.04 or R_so3_error_norm > 0.1:
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
            # 计算机器人当前位置坐标，可由

            
            pc0 = np.matrix(pc0).transpose()
            qc = p.getBasePositionAndOrientation(self.hexapod, self.physicsClient)[1]
            Rot = np.matrix(p.getMatrixFromQuaternion(qc)).reshape((3,3)) * rot_z(-np.pi / 2)
            pos_error = self.pcd - pc0
            pos_error_norm = np.linalg.norm(pos_error)
            pcd_dot = self.kpp * (pos_error)
            R_so3 = np.matrix(sp.SO3(self.Rd * Rot.transpose()).log()).transpose()
            R_so3_error_norm = np.linalg.norm(R_so3)
            _, w = p.getBaseVelocity(self.hexapod, self.physicsClient)
            wbd = self.kwp * R_so3 + self.kwd * (np.matrix([[0], [0], [0]] - np.matrix(w).transpose()))
            print("wbd", wbd)
            # print("pcd", self.pcd)
            # print("pc", pc)
            # print("pos_error", pos_error)
            # print("pos_error_norm", pos_error_norm)
            # print("R_so3", R_so3)
            # print("R_so3_norm", R_so3_error_norm)
            r1 = np.matrix(front_tip)
            r1_0 = r1
            r2 = np.matrix(mid_tip)
            r3 = np.matrix(back_tip)

            v_front_tip = -pcd_dot.transpose() + np.cross(wbd.transpose(), -r1)
            w_front_tip = -wbd.transpose()
            # w_front_tip = wbd.transpose() + np.cross(pcd_dot.transpose(), r1) / np.linalg.norm(r1)**2
            V_front = np.concatenate((v_front_tip[0], w_front_tip[0]), axis=1)
            # v_mid_tip = -pcd_dot.transpose()
            v_mid_tip = -pcd_dot.transpose() + np.cross(wbd.transpose(), -r2)
            w_mid_tip = -wbd.transpose()
            # w_mid_tip = wbd.transpose() + np.cross(pcd_dot.transpose(), r2) / np.linalg.norm(r2)**2
            V_mid = np.concatenate((v_mid_tip[0], w_mid_tip[0]), axis=1)
            # v_back_tip = pcd_dot.transpose()
            v_back_tip = -pcd_dot.transpose() + np.cross(wbd.transpose(), -r3)
            w_back_tip = -wbd.transpose()
            # w_back_tip = wbd.transpose() + np.cross(pcd_dot.transpose(), r3) / np.linalg.norm(r3)**2
            V_back = np.concatenate((v_back_tip[0], w_back_tip[0]), axis=1)
            # print("v_front_tip", v_front_tip)
            # print("w_front_tip", w_front_tip)
            # print("V_front", V_front)
            # print("v_mid_tip", v_mid_tip)
            # print("w_mid_tip", w_mid_tip)
            # print("V_mid", V_mid)
            # print("v_back_tip", v_back_tip)
            # print("w_back_tip", w_back_tip)
            # print("V_back", V_back)
            # 雅可比矩阵
            front_jacobian = self.k.jacobian(front_joint_positions, front_stance_legID)
            mid_jacobian = self.k.jacobian(mid_joint_positions, mid_stance_legID)
            back_jacobian = self.k.jacobian(back_joint_positions, back_stance_legID)
            # 计算雅可比矩阵的逆
            front_jacobian_inverse = np.linalg.pinv(front_jacobian)
            mid_jacobian_inverse = np.linalg.pinv(mid_jacobian)
            back_jacobian_inverse = np.linalg.pinv(back_jacobian)
            # 计算关节速度
            # front_v_theta = (front_jacobian_inverse[0:3, 0:3] * v_front_tip.transpose()).transpose()[0, 0:3]
            # mid_v_theta = (mid_jacobian_inverse[0:3, 0:3] * v_mid_tip.transpose()).transpose()[0, 0:3]
            # back_v_theta = (back_jacobian_inverse[0:3, 0:3] * v_back_tip.transpose()).transpose()[0, 0:3]
            front_v_theta = (front_jacobian_inverse * V_front.transpose()).transpose()[0, 0:3]
            mid_v_theta = (mid_jacobian_inverse * V_mid.transpose()).transpose()[0, 0:3]
            back_v_theta = (back_jacobian_inverse * V_back.transpose()).transpose()[0, 0:3]

            # print("front_v_theta", front_v_theta)
            # print("mid_v_theta", mid_v_theta)
            # print("back_v_theta", back_v_theta)
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
            # print("front_joint_positions_", front_joint_positions_)
            # print("mid_joint_positions_", mid_joint_positions_)
            # print("back_joint_positions_", back_joint_positions_)
            
            stance_joint_positions_ = [0] * 9
            stance_joint_positions_[0:3] = front_joint_positions_
            stance_joint_positions_[3:6] = mid_joint_positions_
            stance_joint_positions_[6:9] = back_joint_positions_
            stance_jointIDs = [0] * 9
            stance_jointIDs[0:3] = [3 * front_stance_legID + i for i in range(3)]
            stance_jointIDs[3:6] = [3 * mid_stance_legID + i for i in range(3)]
            stance_jointIDs[6:9] = [3 * back_stance_legID + i for i in range(3)]
            p.setJointMotorControlArray(robot.hexapod, stance_jointIDs, p.POSITION_CONTROL, stance_joint_positions_, forces=[100]*9)
            p.stepSimulation()
            time.sleep(0.01)
            # 更新状态和误差
            self.wz = p.readUserDebugParameter(self.w_debugId, self.physicsClient)
            Rx = rot_x(p.readUserDebugParameter(self.rot_x_id, self.physicsClient))
            Ry = rot_y(p.readUserDebugParameter(self.rot_y_id, self.physicsClient))
            Rz = rot_z(self.wz)
            self.height_d = p.readUserDebugParameter(self.body_height_id, self.physicsClient)
            self.Rd = Rz * Ry * Rx
            self.pcd[2, 0] = self.height_d
            for i in range(18):
                self.joint_positions[i] = p.getJointState(self.hexapod, self.joint_indices[i], self.physicsClient)[0]
            front_joint_positions = self.joint_positions[(3 * front_stance_legID): (3 * front_stance_legID + 3)]
            mid_joint_positions = self.joint_positions[(3 * mid_stance_legID): (3 * mid_stance_legID + 3)]
            back_joint_positions = self.joint_positions[(3 * back_stance_legID): (3 * back_stance_legID + 3)]
            front_tip = self.k.fk(front_joint_positions, front_stance_legID)
            mid_tip = self.k.fk(mid_joint_positions, mid_stance_legID)
            back_tip = self.k.fk(back_joint_positions, back_stance_legID)
            r1_ = np.matrix(front_tip)
            r2_ = np.matrix(mid_tip)
            r3_ = np.matrix(back_tip)
            pc = [r1_0[0,0] - r1_[0,0], r1_0[0,1] - r1_[0,1], -(front_tip[0][2] + mid_tip[0][2] + back_tip[0][2]) / 3]
            pc_dot = []
            pc = np.matrix(pc).transpose()
            qc = p.getBasePositionAndOrientation(self.hexapod, self.physicsClient)[1]
            R = np.matrix(p.getMatrixFromQuaternion(qc)).reshape((3,3)) * rot_z(-np.pi / 2)
            pos_error = self.pcd - pc
            pos_error_norm = np.linalg.norm(pos_error)
            pcd_dot = self.kpp * (pos_error)
            # print("pos_error", pos_error.transpose())
            # print("pcd", self.pcd.transpose())
            # print("pc", pc.transpose())
            R_so3 = np.matrix(sp.SO3(self.Rd * R.transpose()).log()).transpose()
            R_so3_error_norm = np.linalg.norm(R_so3)
            wbd = self.kwp * R_so3
            r1 = r1_
            r2 = r2_
            r3 = r3_
        # print("pos_error_norm", pos_error_norm)
        # print("R_so3_norm", R_so3_error_norm)
        print("pcd", self.pcd.transpose())
        print("pc", pc.transpose())
        print("pos_error", pos_error.transpose())
        print("Stance Leg ")

    def swing_controller(self):
        self.swingLegID = [i for i, x in enumerate(self.cycle_leg_number_) if x == 0]
        body_target = np.pi / 18
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
            p.setJointMotorControlArray(self.hexapod, swing_jointIDs, p.POSITION_CONTROL, swing_joint_positions_, forces=[100]*9)
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
            p.setJointMotorControlArray(self.hexapod, swing_jointIDs, p.POSITION_CONTROL, swing_joint_positions_, forces=[100]*9)
            p.stepSimulation()
            time.sleep(0.01)
        

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
    # p.setJointMotorControlArray(robot.hexapod, swing_jointIDs, p.POSITION_CONTROL, swing_joint_positions_, forces=[100]*9)
    # for i in range(40):
    #     p.stepSimulation()
    while True:
        # if robot.vx < 0.01:
        #     p.setJointMotorControlArray(robot.hexapod, [i for i in range(18)], p.POSITION_CONTROL, [0.0] * 18, forces = [50] * 18)
        #     p.stepSimulation()
        # else:
        #     robot.gait()
        #     sequence_change(robot.cycle_leg_number_)
        robot.gait()
        print("change gait")
        sequence_change(robot.cycle_leg_number_)