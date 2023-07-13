import numpy as np
from sympy import *


class Kinematic(object):
    def __init__(self) -> None:
        pass

    def fk(self, theta, leg_name):
        q1 = theta[0]
        q2 = theta[1]
        q3 = theta[2]
        if leg_name == 'lf':
            p1x,p1y,p1z = 0.34,0.05,-0.021
            p2x,p2y,p2z = 0,0.076,-0.05
            p3x,p3y,p3z = 0,0.25842,-0.094056
            p4 = np.matrix([0, 0.025, -0.29, 1]).transpose()
        if leg_name == 'lm':
            p1x,p1y,p1z = 0,0.05,-0.021
            p2x,p2y,p2z = 0,0.076,-0.05
            p3x,p3y,p3z = 0,0.25842,-0.094056
            p4 = np.matrix([0, 0.025, -0.29, 1]).transpose()
        if leg_name == 'lb':
            p1x,p1y,p1z = -0.34,0.05,-0.021
            p2x,p2y,p2z = 0,0.076,-0.05
            p3x,p3y,p3z = 0,0.25842,-0.094056
            p4 = np.matrix([0, 0.025, -0.29, 1]).transpose()            
        if leg_name == 'rf':
            p1x,p1y,p1z = 0.34,-0.05,-0.021
            p2x,p2y,p2z = 0,-0.076,-0.05
            p3x,p3y,p3z = 0,-0.25842,-0.094056
            p4 = np.matrix([0, -0.025, -0.29, 1]).transpose()
        if leg_name == 'rm':
            p1x,p1y,p1z = 0,-0.05,-0.021
            p2x,p2y,p2z = 0,-0.076,-0.05
            p3x,p3y,p3z = 0,-0.25842,-0.094056
            p4 = np.matrix([0, -0.025, -0.29, 1]).transpose()
        if leg_name == 'rb':
            p1x,p1y,p1z = -0.34,-0.05,-0.021
            p2x,p2y,p2z = 0,-0.076,-0.05
            p3x,p3y,p3z = 0,-0.25842,-0.094056
            p4 = np.matrix([0, -0.025, -0.29, 1]).transpose()

        T = np.matrix([[np.cos(q1), np.sin(q1)*np.sin(q2)*np.sin(q3) - np.sin(q1)*np.cos(q2)*np.cos(q3), np.sin(q1)*np.sin(q2)*np.cos(q3) + np.sin(q1)*np.sin(q3)*np.cos(q2), p1x + p2x*np.cos(q1) - p2y*np.sin(q1) + p3x*np.cos(q1) - p3y*np.sin(q1)*np.cos(q2) + p3z*np.sin(q1)*np.sin(q2)], 
                                    [np.sin(q1), -np.sin(q2)*np.sin(q3)*np.cos(q1) + np.cos(q1)*np.cos(q2)*np.cos(q3), -np.sin(q2)*np.cos(q1)*np.cos(q3) - np.sin(q3)*np.cos(q1)*np.cos(q2), p1y + p2x*np.sin(q1) + p2y*np.cos(q1) + p3x*np.sin(q1) + p3y*np.cos(q1)*np.cos(q2) - p3z*np.sin(q2)*np.cos(q1)], 
                                    [0, np.sin(q2)*np.cos(q3) + np.sin(q3)*np.cos(q2), -np.sin(q2)*np.sin(q3) + np.cos(q2)*np.cos(q3), p1z + p2z + p3y*np.sin(q2) + p3z*np.cos(q2)], 
                                    [0, 0, 0, 1]])
        position = (T * p4).transpose()[0,0:3].tolist()
        return position

    def jacobian(self, theta, leg_name):
        q1 = theta[0]
        q2 = theta[1]
        q3 = theta[2]
        if leg_name == 'lf':
            p1x,p1y,p1z = 0.34,0.05,-0.021
            p2x,p2y,p2z = 0,0.076,-0.05
            p3x,p3y,p3z = 0,0.25842,-0.094056
            p4x,p4y,p4z = 0,0.025,-0.29
        if leg_name == 'lm':
            p1x,p1y,p1z = 0,0.05,-0.021
            p2x,p2y,p2z = 0,0.076,-0.05
            p3x,p3y,p3z = 0,0.25842,-0.094056
            p4x,p4y,p4z = 0,0.025,-0.29
        if leg_name == 'lb':
            p1x,p1y,p1z = -0.34,0.05,-0.021
            p2x,p2y,p2z = 0,0.076,-0.05
            p3x,p3y,p3z = 0,0.25842,-0.094056
            p4x,p4y,p4z = 0,0.025,-0.29            
        if leg_name == 'rf':
            p1x,p1y,p1z = 0.34,0.05,-0.021
            p2x,p2y,p2z = 0,-0.076,-0.05
            p3x,p3y,p3z = 0,-0.25842,-0.094056
            p4x,p4y,p4z = 0,-0.025,-0.29
        if leg_name == 'rm':
            p1x,p1y,p1z = 0,0.05,-0.021
            p2x,p2y,p2z = 0,-0.076,-0.05
            p3x,p3y,p3z = 0,-0.25842,-0.094056
            p4x,p4y,p4z = 0,-0.025,-0.29
        if leg_name == 'rb':
            p1x,p1y,p1z = -0.34,0.05,-0.021
            p2x,p2y,p2z = 0,-0.076,-0.05
            p3x,p3y,p3z = 0,-0.25842,-0.094056
            p4x,p4y,p4z = 0,-0.025,-0.29   
        J = np.matrix([[-p2x*np.sin(q1) - p2y*np.cos(q1) - p3x*np.sin(q1) - p3y*np.cos(q1)*np.cos(q2) + p3z*np.sin(q2)*np.cos(q1) - p4x*np.sin(q1) + p4y*(np.sin(q2)*np.sin(q3)*np.cos(q1) - np.cos(q1)*np.cos(q2)*np.cos(q3)) + p4z*(np.sin(q2)*np.cos(q1)*np.cos(q3) + np.sin(q3)*np.cos(q1)*np.cos(q2)), p3y*np.sin(q1)*np.sin(q2) + p3z*np.sin(q1)*np.cos(q2) + p4y*(np.sin(q1)*np.sin(q2)*np.cos(q3) + np.sin(q1)*np.sin(q3)*np.cos(q2)) + p4z*(-np.sin(q1)*np.sin(q2)*np.sin(q3) + np.sin(q1)*np.cos(q2)*np.cos(q3)), p4y*(np.sin(q1)*np.sin(q2)*np.cos(q3) + np.sin(q1)*np.sin(q3)*np.cos(q2)) + p4z*(-np.sin(q1)*np.sin(q2)*np.sin(q3) + np.sin(q1)*np.cos(q2)*np.cos(q3)), 0], 
                                    [p2x*np.cos(q1) - p2y*np.sin(q1) + p3x*np.cos(q1) - p3y*np.sin(q1)*np.cos(q2) + p3z*np.sin(q1)*np.sin(q2) + p4x*np.cos(q1) + p4y*(np.sin(q1)*np.sin(q2)*np.sin(q3) - np.sin(q1)*np.cos(q2)*np.cos(q3)) + p4z*(np.sin(q1)*np.sin(q2)*np.cos(q3) + np.sin(q1)*np.sin(q3)*np.cos(q2)), -p3y*np.sin(q2)*np.cos(q1) - p3z*np.cos(q1)*np.cos(q2) + p4y*(-np.sin(q2)*np.cos(q1)*np.cos(q3) - np.sin(q3)*np.cos(q1)*np.cos(q2)) + p4z*(np.sin(q2)*np.sin(q3)*np.cos(q1) - np.cos(q1)*np.cos(q2)*np.cos(q3)), p4y*(-np.sin(q2)*np.cos(q1)*np.cos(q3) - np.sin(q3)*np.cos(q1)*np.cos(q2)) + p4z*(np.sin(q2)*np.sin(q3)*np.cos(q1) - np.cos(q1)*np.cos(q2)*np.cos(q3)), 0], 
                                    [0, p3y*np.cos(q2) - p3z*np.sin(q2) + p4y*(-np.sin(q2)*np.sin(q3) + np.cos(q2)*np.cos(q3)) + p4z*(-np.sin(q2)*np.cos(q3) - np.sin(q3)*np.cos(q2)), p4y*(-np.sin(q2)*np.sin(q3) + np.cos(q2)*np.cos(q3)) + p4z*(-np.sin(q2)*np.cos(q3) - np.sin(q3)*np.cos(q2)), 0], 
                                    [0, np.cos(q1), np.cos(q1), np.cos(q1)], 
                                    [0, np.sin(q1), np.sin(q1), np.sin(q1)], 
                                    [1, 0, 0, 0]])
        return J

    def ik(self, xd, leg_name):
        alpha = 1 # step size
        error = 100
        ilimit = 1000
        count = 0
        q1, q2, q3 = 0, 0, 0
        while error > 0.001:
            x = self.fk([q1,q2,q3], leg_name)[0]
            print(x)
            J = self.jacobian([q1,q2,q3], leg_name)[0:3,:]
            print(J)
            inv_J = np.linalg.pinv(J)
            dx = np.matrix([xd[i]-x[i] for i in range(3)])[0]
            print(dx)
            error = np.linalg.norm(dx)
            print(error)
            dq = alpha * inv_J * dx.transpose()
            print(dq)
            q1 += dq[0,0]
            q2 += dq[1,0]
            q3 += dq[2,0]
            target_q = [q1,q2,q3]
            print(target_q)
            count += 1
            print(count)
            if count > ilimit:
                print('Ik did not find solution')
                target_q = [0,0,0]
                break
        return target_q


# k = Kinematic()
# print(k.fk([0,0,0], 'lf'))
# qd = k.ik([0.34, 0.2, -.4], 'lf')
# print(qd)
# print(k.fk(qd, 'lf'))

# print(k.ik([0.34,0.2, -0.2], 'lf'))
# theta = [.0]*3
# J = k.jacobian(theta, 'rb')
# print(J * np.matrix([[0], [0], [np.pi], [0]]))

# q1,q2,q3,q4 = symbols('q1 q2 q3 q4')
# p1x, p1y, p1z = symbols('p1x p1y p1z')
# p1 = Matrix([[p1x,p1y,p1z]])
# T1 = Matrix([[cos(q1), -1*sin(q1), 0, 0], 
#                         [sin(q1), cos(q1), 0, 0], 
#                         [0, 0, 1, 0],
#                         [0, 0, 0, 1]])
# T1[0:3,3] = p1.transpose()

# p2x, p2y, p2z = symbols('p2x p2y p2z')
# p2 = Matrix([[p2x,p2y,p2z]])
# T2 = Matrix([[1, 0, 0, 0], 
#                         [0, cos(q2), -1*sin(q2), 0], 
#                         [0, sin(q2), cos(q2), 0],
#                         [0, 0, 0, 1]])
# T2[0:3,3] = p2.transpose()

# p3x, p3y, p3z = symbols('p3x p3y p3z')
# p3 = Matrix([[p3x,p3y,p3z]])
# T3 = Matrix([[1, 0, 0, 0], 
#                         [0, cos(q3), -1*sin(q3), 0], 
#                         [0, sin(q3), cos(q3), 0],
#                         [0, 0, 0, 1]])
# T3[0:3,3] = p3.transpose()

# p4x, p4y, p4z = symbols('p4x p4y p4z')
# T4 = Matrix([[1, 0, 0, 0],
#                         [0, 1, 0, 0 ],
#                         [0, 0, 1, 0],
#                         [0, 0, 0, 1]])
# p4 = Matrix([[p4x, p4y, p4z]])
# T4[0:3,3] = p4.transpose()
# T = T1 * T2 * T3 * T4

# # tip = T * p4.transpose()

# J = Matrix([[diff(T[0,3], q1), diff(T[0,3], q2), diff(T[0,3], q3), diff(T[0,3], q4)], 
#                         [diff(T[1,3], q1), diff(T[1,3], q2), diff(T[1,3], q3), diff(T[1,3], q4)],
#                         [diff(T[2,3], q1), diff(T[2,3], q2), diff(T[2,3], q3), diff(T[2,3], q4)],
#                         [0,0,0,0],
#                         [0,0,0,0],
#                         [0,0,0,0]])

# wb1 = T1[0:3,0:3] * Matrix([[0,0,1]]).transpose()
# wb2 = T1[0:3,0:3] * T2[0:3,0:3] * Matrix([[1,0,0]]).transpose()
# wb3 = T1[0:3,0:3] * T2[0:3,0:3] * T3[0:3,0:3] * Matrix([[1,0,0]]).transpose()
# wb4 = T1[0:3,0:3] * T2[0:3,0:3] * T3[0:3,0:3] * T4[0:3,0:3] * Matrix([[1,0,0]]).transpose()

# J[3:6,0] =wb1
# J[3:6,1] =wb2
# J[3:6,2] =wb3
# J[3:6,3] =wb4

# print(J)