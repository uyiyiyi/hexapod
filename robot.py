import numpy as np
import pybullet as p
import math
import balance_controller
import time


NUM_MOTORS = 18
NUM_LEGS = 6
MOTOR_NAMES = [
    "body_lb",
    "thigh_lb",
    "shank_lb",
    "body_lm",
    "thigh_lm",
    "shank_lm",
    "body_lf",
    "thigh_lf",
    "shank_lf",
    "body_rf",
    "thigh_rf",
    "shank_rf",
    "body_rm",
    "thigh_rm",
    "shank_rm",
    "body_rb",
    "thigh_rb",
    "shank_rb",
]
MOTOR_INDEX = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
MOTOR_DICT = dict(zip(MOTOR_NAMES, MOTOR_INDEX))
FOOT_TIP_NAMES = ['LB', 'LM', 'LF', 'RF', 'RM', 'RB']
FOOT_TIP_LINK = [3, 7, 11, 15, 19, 23]
LEG_ID = [0, 1, 2, 3, 4, 5]
LEG_ID_DICT = dict(zip(FOOT_TIP_NAMES, LEG_ID))
FOOT_TIP_DICT = dict(zip(FOOT_TIP_NAMES, FOOT_TIP_LINK))
INIT_COM_POSITION = [0, 0, 0.456]
PI = math.pi
MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
_DEFAULT_HIP_POSITIONS = (
    (-0.34, -0.05, -0.021),
    (.0, -0.05, -0.021),
    (0.34, -0.05, -0.021),
    (0.34, 0.05, -0.021),
    (.0, 0.05, -0.021),
    (-0.34, 0.05, -0.021),
)
URDF_FILENAME = "simple_robot.urdf"
PLANE_FILENAME = "plane.urdf"

def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]


class Robot():
    def __init__(self):
        self.cid = p.connect(p.GUI)
        p.setGravity(0,0,-9.8)
        p.setTimeStep(0.001)
        self.hexapod = p.loadURDF(URDF_FILENAME,INIT_COM_POSITION, p.getQuaternionFromEuler([0,0,0]), useFixedBase=False)
        self.plane = p.loadURDF(PLANE_FILENAME, baseOrientation=p.getQuaternionFromEuler([0,0,0]))
        self.support_foot = []
        self.support_foot_position = []
        self.joint_angles = []
        self.body_mass = 17.206
        self.Ixx = 1.1941
        self.Iyy = 1.1576
        self.Izz = 0.21746
        self.Ixy = 0.003952
        self.Ixz = 9.5899E-05
        self.Iyy = 1.1576
        self.Iyz = 0.078153
        self.I_g = np.matrix([[self.Ixx, self.Ixy, self.Ixz], [self.Ixy, self.Iyy, self.Iyz], [self.Ixz, self.Iyz, self.Izz]])
        maxForce = [.0] * 18
        self.position = []
        self.rotation_matrix = []
        self.linear_velocity = []
        self.angular_velocity = []
        # for i in range(25):
        #     p.changeDynamics(self.hexapod, i, restitution=0.05)
        # p.changeDynamics(self.plane, -1, restitution=0.05)
        num_bullet_solver_iterations = 30
        p.setPhysicsEngineParameter(numSolverIterations=num_bullet_solver_iterations)
        p.setPhysicsEngineParameter(enableConeFriction=0)
        mode = p.VELOCITY_CONTROL
        p.setJointMotorControlArray(self.hexapod, MOTOR_INDEX,controlMode=mode, forces=maxForce)
        p.stepSimulation()

    def get_base_info(self):
        position, quaternion = p.getBasePositionAndOrientation(self.hexapod, self.cid)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.hexapod, self.cid)
        rotation_matrix = p.getMatrixFromQuaternion(quaternion)
        self.position = list(position)
        self.rotation_matrix = list(rotation_matrix)
        self.linear_velocity = list(linear_velocity)
        self.angular_velocity = list(angular_velocity)

    def get_foot_contact(self):
        support_foot = []
        support_foot_id = []
        keys = []
        for tip in FOOT_TIP_LINK:
            res = p.getContactPoints(self.hexapod, self.plane, tip, -1, self.cid)
            if res:
                support_foot += [tip]
                support_foot_id += [LEG_ID_DICT[get_key(FOOT_TIP_DICT, tip)[0]]]
                keys += get_key(FOOT_TIP_DICT, tip)
        self.support_foot_dict = dict(zip(keys, support_foot))
        self.support_foot = support_foot
        self.support_foot_id = support_foot_id
    
    def get_contact_foot_position(self):
        position = []
        self.get_foot_contact()
        for i, foot in enumerate(self.support_foot):
            position += [list(p.getLinkState(self.hexapod, foot, self.cid)[0])]
        self.support_foot_position_dict = dict(zip(self.support_foot_dict.keys(), position))
        self.support_foot_position = position
    
    def get_joint_angle(self):
        joint_angles = []
        res = p.getJointStates(self.hexapod, MOTOR_INDEX, self.cid)
        for i in range(len(res)):
            joint_angles += [res[i][0]]
        self.joint_angles = joint_angles

    def set_joint_torque(self, torques):
        legs = self.support_foot
        joints = sum([[i-3, i-2, i-1] for i in legs], [])
        print('joints=', joints)
        p.setJointMotorControlArray(self.hexapod, joints, p.TORQUE_CONTROL, torques)
        p.stepSimulation()



robot = Robot()
bc = balance_controller.BalanceController(robot)
p.resetBasePositionAndOrientation(robot.hexapod, INIT_COM_POSITION, p.getQuaternionFromEuler([0,0,0]), robot.cid)
for i in range(23):
    p.enableJointForceTorqueSensor(robot.hexapod, i, 1, robot.cid)
# for i in range(10):
#     p.stepSimulation()
p.stepSimulation()

while True:
    robot.get_base_info()
    robot.get_contact_foot_position()
    robot.get_joint_angle()
    F = bc.controller()
    torques = bc.MapContactForceToJointTorques(F)
    print('torques=', torques)
    robot.set_joint_torque(torques)
    # force = []
    # res = p.getJointStates(robot.hexapod, MOTOR_INDEX, robot.cid)
    # for i, motor in enumerate(MOTOR_INDEX):
    #     force += [res[i][3]]
    # print('force=', force)
    p.stepSimulation()
    # time.sleep(0.001)
    input()