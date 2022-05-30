from cvxopt import matrix, solvers
import pybullet as p
import numpy as np
import cvxopt
from sympy import *
import sophus as sp
from qpsolvers import solve_qp


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    # P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x'])[:,0]

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

class BalanceController(object):
    def __init__(self, Robot) -> None:
        self.cid = Robot.cid
        self.Robot = Robot
        self.g = np.array([[0,0,-9.8]]).transpose()
        self.m = self.Robot.body_mass   # mass of the robot
        # Body Inertia
        self.I_g = self.Robot.I_g
        self.pcd = np.array([[0,0,0.455056]]).transpose()
        self.pcd_dot = np.array([[0,0,0]]).transpose()
        self.Rd =  np.identity(3) # rot_z(np.pi / 6)
        self.wbd = np.array([[.0,.0,.0]]).transpose()
        self.pcd_x_id = p.addUserDebugParameter('pc_desire_x', -0.1, 0.1, .0, self.cid)
        self.pcd_y_id = p.addUserDebugParameter('pc_desire_y', -0.1, 0.1, .0, self.cid)
        self.pcd_z_id = p.addUserDebugParameter('pc_desire_z', 0.3, 0.5, 0.455056, self.cid)
        self.rot_x_id = p.addUserDebugParameter('rot_x', -np.pi / 9, np.pi / 9, 0, self.cid)
        self.rot_y_id = p.addUserDebugParameter('rot_y', -np.pi / 9, np.pi / 9, 0, self.cid)
        self.rot_z_id = p.addUserDebugParameter('rot_z', -np.pi / 4, np.pi / 4, 0, self.cid)
        kp_xy = 0
        kp_z = 100
        kd_xy = 10
        kd_z = 10
        kp_wxy = 100
        kp_wz = 0
        kd_wxyz = 10
        self.Kpp = np.diag([0., 0., 100.]) # [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        self.Kdp = np.diag([40., 30., 10.])
        self.Kpw = np.diag([100., 100., 0.])
        self.Kdw = np.diag([10., 10., 30.])
        # self.S = np.identity(6)

    def controller(self):
        self.pcd[0,0] = p.readUserDebugParameter(self.pcd_x_id, self.cid)
        self.pcd[1,0] = p.readUserDebugParameter(self.pcd_y_id, self.cid)
        self.pcd[2,0] = p.readUserDebugParameter(self.pcd_z_id, self.cid)
        Rx = rot_x(p.readUserDebugParameter(self.rot_x_id, self.cid))
        Ry = rot_y(p.readUserDebugParameter(self.rot_y_id, self.cid))
        Rz = rot_z(p.readUserDebugParameter(self.rot_z_id, self.cid))
        self.Rd = Rz * Ry * Rx
        pc = np.matrix(self.Robot.position).transpose()
        R = np.matrix(self.Robot.rotation_matrix).reshape((3,3))
        pc_dot = np.matrix(self.Robot.linear_velocity).transpose()
        w = np.matrix(self.Robot.angular_velocity).transpose()
        ad = self.Kpp * (self.pcd - pc) + self.Kdp * (self.pcd_dot - pc_dot)
        R_so3 = np.matrix(sp.SO3(self.Rd * R.transpose()).log()).transpose()
        wbd_dot = self.Kpw * R_so3 + self.Kdw * (self.wbd - w)
        bd = np.matrix(np.vstack((self.m * (ad + self.g), self.I_g * wbd_dot)))

        num_leg = len(self.Robot.support_foot_dict.keys())
        A_upper = np.zeros((3, 3*num_leg))
        A_lower = np.zeros((3, 3*num_leg))
        diag_g = []
        elem_h = []
        lb = []
        ub = []
        for index, i in enumerate(self.Robot.support_foot):
            position = self.Robot.support_foot_position[index]
            A_upper[0:3, 3*index:(3*index+3)] = np.identity(3)
            A_lower[0:3, 3*index:(3*index+3)] = skew(position)
            lb += [-1.0, -1.0, 0]
            ub += [1.0, 1.0, 100.0]
            diag_g += [1.0,1.0,1.0]
            elem_h += [1.0,1.0,100.0]

        A = np.matrix(np.vstack((A_upper, A_lower)))
        S = np.diag([1,1,1,1,1,1])
        P = np.matrix(0.5 * A.transpose() * S *  A)
        P = np.array(P.transpose() + P)
        q_ = np.array((-bd.transpose() * S *  A).transpose())
        n = shape(q_)[0]
        q = q_.reshape((n,))
        G = np.array(np.diag(diag_g))
        h = np.array(elem_h)
        # F = np.matrix(cvxopt_solve_qp(P, q, G, h)).transpose()
        F = solve_qp(P, q, lb=np.array(lb), ub=np.array(ub), solver='cvxopt')
        print("QP solution: x = {}".format(F))
        return F

    def MapContactForceToJointTorques(self, F):
        J = {}
        if F == []:
            torques = [.0] * 18
            self.Robot.support_foot = [3, 7, 11, 15, 19, 23]
            return torques
        F = np.matrix(F).transpose()
        torques = []
        zero_vec = [0] * len(self.Robot.joint_angles)
        legs = self.Robot.support_foot_dict.keys()
        tips = self.Robot.support_foot
        leg_id = self.Robot.support_foot_id
        self.Robot.get_joint_angle()
        for i, leg in enumerate(legs):
            Jv, _ = p.calculateJacobian(self.Robot.hexapod, tips[i], (0, 0, 0), self.Robot.joint_angles, zero_vec, zero_vec)
            J[leg] = np.matrix(Jv)[0:3, (6+leg_id[i]*3):(6+leg_id[i]*3+3)]
            torque = (J[leg].transpose() * -F[3*i:(3*i+3)]).transpose().tolist()[0]
            # print(torque)
            torques += torque
        # torques = [-torque for torque in torques]
        return torques