import pybullet as p
import time
import pybullet_data
import numpy as np
import robot, balance_controller

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)

# engine parameters
p.setPhysicsEngineParameter(fixedTimeStep=0.001,
                            numSolverIterations=50,
                            erp=0.2,
                            contactERP=0.2,
                            frictionERP=0.2)

# planeId = p.loadURDF("urdf/plane.urdf")
# p.changeDynamics(bodyUniqueId=planeId,
#                  linkIndex=-1,
#                  lateralFriction=1.0)

# variables
numJoint = 0    # will be updated!
numrow = 1
robotIds = []

# joint configuration
# jointConfig = np.array([
#     0.03, 0.4, -0.8, 0,   # LF
#     -0.03, 0.4, -0.8, 0,  # RF
#     0.03, -0.4, 0.8, 0,   # LH
#     -0.03, -0.4, 0.8, 0,  # RH
#     0
# ])
jointConfig = np.array([.0, .0, .0]*6)

# robot load
startPos = [0, 0, 0.456]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])


robot = robot.Robot(physicsClient)
bc = balance_controller.BalanceController(robot)

# URDF1 = "robot_liuzu.urdf"
# URDF2 = "simple_robot.urdf"
# robotId = p.loadURDF(URDF1, startPos, startOrientation, useMaximalCoordinates=0, flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_IMPLICIT_CYLINDER | p.URDF_USE_SELF_COLLISION)

jointIdx = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22])
zeros = np.ones(18)

for i, joint in enumerate(jointIdx):
    p.resetJointState(bodyUniqueId=robot.hexapod, jointIndex=joint, targetValue=jointConfig[i])
    # p.changeDynamics(bodyUniqueId=planeId,  linkIndex=joint, lateralFriction=0.8)

p.setJointMotorControlArray(bodyUniqueId=robot.hexapod, jointIndices=jointIdx, controlMode=p.VELOCITY_CONTROL, forces=zeros)


# gains (DO NOT CHANGE!)
kp = 80
kd = 1



# simulation step
for t in range (20000):



    # # control
    # jointStates = p.getJointStates(bodyUniqueId=robot.hexapod,
    #                                    jointIndices=jointIdx)
    # jointPositions = np.array([s[0] for s in jointStates])
    # jointVelocities = np.array([s[1] for s in jointStates])

    # jointTorques = kp * (jointConfig - jointPositions) - kd * jointVelocities

    # print(jointTorques)

    # p.setJointMotorControlArray(bodyUniqueId=robot.hexapod,
    #                                 jointIndices=jointIdx,
    #                                 controlMode=p.TORQUE_CONTROL,
    #                                 forces=jointTorques)

    # # simulation step
    # p.stepSimulation()


    robot.get_base_info()
    robot.get_contact_foot_position()
    robot.get_joint_angle()
    F = bc.controller()
    torques = bc.MapContactForceToJointTorques(F)
    robot.set_joint_torque(torques)
    # print(torques)
    p.stepSimulation()

    time.sleep(1./240.)
    # info = p.getContactPoints()

p.disconnect()