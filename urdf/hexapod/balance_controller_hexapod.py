from cvxopt import matrix, solvers
import pybullet as p
import numpy as np
from sympy import *
import  sophus as sp



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

class BodyController(object):
    def __init__(self) -> None:
        self.cid = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        p.setTimeStep(0.001)
        p.setGravity(0,0,-9.8)
        self.planeId = p.loadURDF("urdf/plane.urdf", baseOrientation=p.getQuaternionFromEuler([0,0,0]))
        self.hexapod = p.loadURDF("robot_liuzu/urdf/robot_liuzu.urdf",[0,0,0.455056], p.getQuaternionFromEuler([0,0,0]))
        self.pcd = np.array([[0,0,0.455056]]).transpose()
        self.pcd_x_id = p.addUserDebugParameter('pc_desire_x', -0.1, 0.1, .0, self.cid)
        self.pcd_y_id = p.addUserDebugParameter('pc_desire_y', -0.1, 0.1, .0, self.cid)
        self.pcd_z_id = p.addUserDebugParameter('pc_desire_z', 0.2, 0.6, 0.455056, self.cid)
        self.pcd_dot = np.array([[0,0,0]]).transpose()
        self.rot_x_id = p.addUserDebugParameter('rot_x', -np.pi / 12, np.pi / 12, 0, self.cid)
        self.rot_y_id = p.addUserDebugParameter('rot_y', -np.pi / 12, np.pi / 12, 0, self.cid)
        self.rot_z_id = p.addUserDebugParameter('rot_z', -np.pi / 18, np.pi / 18, 0, self.cid)
        self.Rd =  np.identity(3) # rot_z(np.pi / 6)
        self.wbd = np.array([[.0,.0,.0]]).transpose()
        self.Kp = np.diag([100, 100, 100]) # [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        self.Kw = np.diag([100, 100, 100])
        # self.S = np.identity(6)
        p.stepSimulation()

    def controller(self, kinematic):
        self.pcd[0,0] = p.readUserDebugParameter(self.pcd_x_id, self.cid)
        self.pcd[1,0] = p.readUserDebugParameter(self.pcd_y_id, self.cid)
        self.pcd[2,0] = p.readUserDebugParameter(self.pcd_z_id, self.cid)
        Rx = rot_x(p.readUserDebugParameter(self.rot_x_id, self.cid))
        Ry = rot_y(p.readUserDebugParameter(self.rot_y_id, self.cid))
        Rz = rot_z(p.readUserDebugParameter(self.rot_z_id, self.cid))
        self.Rd = Rz * Ry * Rx
        pc, qc = p.getBasePositionAndOrientation(self.body, self.cid)
        pc = np.matrix(pc).transpose()
        R = np.matrix(p.getMatrixFromQuaternion(qc)).reshape((3,3))
        self.pcd_dot = self.Kp * (self.pcd - pc)
        R_so3 = np.matrix(sp.SO3(self.Rd * R.transpose()).log()).transpose()
        self.wbd = self.Kw * R_so3



    def MapContactForceToJointTorques(self, kinematic, F, stance_legs):
        jps = []
        for i in range(18):
            jp = p.getJointState(self.hexapod, i, self.cid)[0]
            jps += [jp]
        leg_name = ['lb', 'lm', 'lf', 'rf', 'rm', 'rb']
        torques = []
        for index, i in enumerate(stance_legs):
            Jv = kinematic.jacobian(jps[3*i:3*i+3], leg_name[i])[0:3,:]
            torque = (Jv.transpose() * -F[3*index:(3*index+3)])[0:3].transpose().tolist()[0]
            torques += torque
        # torques = [-torques[i] for i in len(torques)]
        return torques

    def actuate_motors(self, torques, stance_legs):
        joints = []
        for i in range(len(torques)):
            if torques[i] > 0.60:
                torques[i] = 60
            if torques[i] < -60:
                torques[i] = -60
        for index, i in enumerate(stance_legs):
            joints += [3*i, 3*i+1, 3*i+2]
        print(torques)
        maxForce = [0] * 18
        mode = p.VELOCITY_CONTROL
        p.setJointMotorControlArray(self.hexapod, [i for i in range(18)],controlMode=mode, forces=maxForce)
        p.setJointMotorControlArray(self.hexapod, joints, p.TORQUE_CONTROL, torques)
        p.stepSimulation()

    def apply_force(self, F):
        f1 = F[0:3,0].transpose().tolist()[0]
        f2 = F[3:6].transpose().tolist()[0]
        f3 = F[6:9].transpose().tolist()[0]
        f4 = F[9:12].transpose().tolist()[0]
        f5 = F[12:15].transpose().tolist()[0]
        f6 = F[15:18].transpose().tolist()[0]
        p.applyExternalForce(self.hexapod,-1,f1, [-0.34, 0.40942, 0],p.WORLD_FRAME)
        p.applyExternalForce(self.hexapod,-1,f2, [0, 0.40942, 0],p.WORLD_FRAME)
        p.applyExternalForce(self.hexapod,-1,f3, [0.34, 0.40942, 0],p.WORLD_FRAME)
        p.applyExternalForce(self.hexapod,-1,f4, [0.34, -0.40942, 0],p.WORLD_FRAME)
        p.applyExternalForce(self.hexapod,-1,f5, [0, -0.40942, 0],p.WORLD_FRAME)
        p.applyExternalForce(self.hexapod,-1,f6, [-0.34, -0.40942, 0],p.WORLD_FRAME)
        p.stepSimulation()

    def legState(self):
        stance_leg_id = list()
        contact_points = p.getContactPoints(self.hexapod, self.planeId)
        stance_leg_num = len(contact_points)
        for i in range(stance_leg_num):
            stance_leg_id += [int((contact_points[i][3] - 2) / 3)]
        res1 = list(set(stance_leg_id))
        res1.sort(key=stance_leg_id.index)
        result = []
        for i in range(len(res1)):
            if res1[i] >= 0 and res1[i] <= 5:
                result += [res1[i]]
        return result

# render = False
sc = StanceController()
while True:
    F = sc.controller()
    if render == True:
        renderForce(F)
    sc.apply_force(F)
    # input()

