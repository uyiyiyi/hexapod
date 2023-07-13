# 本脚本需要实现六足机器人跟踪落足点的功能
import pybullet as p
import pybullet_data
import numpy as np
import open3d as o3d
import os
import time
import math

NUM_LEGS = 6
CYCLE_LENGTH = 100 # 50 time steps
HALF_CYCLE_LENGTH = int(CYCLE_LENGTH/2)
LEG_LIFT_HEIGHT = 0.1

class Hexapod(object):
    def __init__(self) -> None:
        self.physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        p.setGravity(0,0,-9.8)
        self.timestep = 0.01
        p.setTimeStep(self.timestep, self.physicsClient)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        self.planeId = p.loadURDF("urdf/plane.urdf", baseOrientation=p.getQuaternionFromEuler([0,0,0]))
        # self.planeId = p.loadURDF("terrain.urdf", baseOrientation=p.getQuaternionFromEuler([0,0,0]))
        self.hexapod = p.loadURDF("robot_liuzu/urdf/robot_liuzu.urdf",[0,0,0.46], p.getQuaternionFromEuler([0,0,np.pi/2]))
        p.setJointMotorControlArray(self.hexapod, [i for i in range(18)], p.POSITION_CONTROL, [0.0] * 18)
        self.forward_debugId = p.addUserDebugParameter("forward_length", 0.005, 0.1, 0.05, self.physicsClient)
        self.turn_debugId = p.addUserDebugParameter("turn_radius", -0.4, 0.4, 0.0001, self.physicsClient)
        self.cycle_leg_number_ =[0,1,0,1,0,1]

    def gait(self):
        forward_length = p.readUserDebugParameter(self.forward_debugId, self.physicsClient) / 1.4
        turn_radius = p.readUserDebugParameter(self.turn_debugId, self.physicsClient) * 1.4
        width = 0.41
        r = forward_length / abs(np.tan(turn_radius)) - 0.41
        delta = forward_length * 0.41 / (r + 0.41)
        stanceLegGoalAngle = - np.arcsin(forward_length / width)
        stanceLegTrajectory = np.zeros((CYCLE_LENGTH,3))
        swingLegTrajectory = np.zeros((CYCLE_LENGTH,3))
        allLegTrajectory = np.zeros((6,CYCLE_LENGTH,3))
        for i in range(NUM_LEGS):
            # stance leg
            if self.cycle_leg_number_[i] == 1:
                body_init = p.getJointState(self.hexapod, 3*i, self.physicsClient)[0]
                stanceLegTrajectory[:,0] = np.linspace(body_init, stanceLegGoalAngle, CYCLE_LENGTH)
                # if turn_radius >= 0 and i <= 2:
                #     stanceLegGoalAngle = np.arcsin((forward_length - delta) / width)
                #     body_init = p.getJointState(self.hexapod, 3*i, self.physicsClient)[0]
                #     stanceLegTrajectory[:,0] = np.linspace(body_init, stanceLegGoalAngle, CYCLE_LENGTH)
                # if turn_radius >= 0 and i > 2:
                #     stanceLegGoalAngle = np.arcsin((forward_length + delta) / width)
                #     body_init = p.getJointState(self.hexapod, 3*i, self.physicsClient)[0]
                #     stanceLegTrajectory[:,0] = np.linspace(body_init, stanceLegGoalAngle, CYCLE_LENGTH)
                # if turn_radius < 0 and i <= 2:
                #     stanceLegGoalAngle = np.arcsin((forward_length + delta) / width)
                #     body_init = p.getJointState(self.hexapod, 3*i, self.physicsClient)[0]
                #     stanceLegTrajectory[:,0] = np.linspace(body_init, stanceLegGoalAngle, CYCLE_LENGTH)
                # if turn_radius < 0 and i > 2:
                #     stanceLegGoalAngle = np.arcsin((forward_length - delta) / width)
                #     body_init = p.getJointState(self.hexapod, 3*i, self.physicsClient)[0]
                #     stanceLegTrajectory[:,0] = np.linspace(body_init, stanceLegGoalAngle, CYCLE_LENGTH)
                allLegTrajectory[i] = stanceLegTrajectory
            # swing leg
            if self.cycle_leg_number_[i] == 0:
                if turn_radius >= 0 and i <= 2:
                    swingLegGoalAngle = np.arcsin((forward_length - delta) / width)
                    body_init = p.getJointState(self.hexapod, 3*i, self.physicsClient)[0]
                    swingLegTrajectory[:,0] = np.linspace(body_init, swingLegGoalAngle, CYCLE_LENGTH)
                if turn_radius >= 0 and i > 2:
                    swingLegGoalAngle = np.arcsin((forward_length + delta) / width)
                    body_init = p.getJointState(self.hexapod, 3*i, self.physicsClient)[0]
                    swingLegTrajectory[:,0] = np.linspace(body_init, swingLegGoalAngle, CYCLE_LENGTH)
                if turn_radius < 0 and i <= 2:
                    swingLegGoalAngle = np.arcsin((forward_length + delta) / width)
                    body_init = p.getJointState(self.hexapod, 3*i, self.physicsClient)[0]
                    swingLegTrajectory[:,0] = np.linspace(body_init, swingLegGoalAngle, CYCLE_LENGTH)
                if turn_radius < 0 and i > 2:
                    swingLegGoalAngle = np.arcsin((forward_length - delta) / width)
                    body_init = p.getJointState(self.hexapod, 3*i, self.physicsClient)[0]
                    swingLegTrajectory[:,0] = np.linspace(body_init, swingLegGoalAngle, CYCLE_LENGTH)

                swingLegTrajectory[0:HALF_CYCLE_LENGTH,1] = np.linspace(0, np.pi/12, HALF_CYCLE_LENGTH)
                swingLegTrajectory[HALF_CYCLE_LENGTH:CYCLE_LENGTH,1] = np.linspace(np.pi/12, 0, HALF_CYCLE_LENGTH)
                swingLegTrajectory[0:HALF_CYCLE_LENGTH,2] = np.linspace(0, -np.pi/4, HALF_CYCLE_LENGTH)
                swingLegTrajectory[HALF_CYCLE_LENGTH:CYCLE_LENGTH,2] = np.linspace(-np.pi/4, 0, HALF_CYCLE_LENGTH)
                allLegTrajectory[i] = swingLegTrajectory
        self.move_joints(allLegTrajectory)

    def move_joints(self, trajectory):
        [x,y,z] = p.getBasePositionAndOrientation(self.hexapod, self.physicsClient)[0]
        q = p.getBasePositionAndOrientation(self.hexapod, self.physicsClient)[1]
        yaw = p.getEulerFromQuaternion(q, self.hexapod)[2]
        for i in range(CYCLE_LENGTH):
            # p.setJointMotorControlArray(robot.hexapod, [0,1,2], p.POSITION_CONTROL, trajectory[0][i])
            # p.setJointMotorControlArray(robot.hexapod, [4,5,6], p.POSITION_CONTROL, trajectory[1][i])
            # p.setJointMotorControlArray(robot.hexapod, [8,9,10], p.POSITION_CONTROL, trajectory[2][i])
            # p.setJointMotorControlArray(robot.hexapod, [12,13,14], p.POSITION_CONTROL, trajectory[3][i])
            # p.setJointMotorControlArray(robot.hexapod, [16,17,18], p.POSITION_CONTROL, trajectory[4][i])
            # p.setJointMotorControlArray(robot.hexapod, [20,21,22], p.POSITION_CONTROL, trajectory[5][i])
            p.setJointMotorControlArray(robot.hexapod, [0,1,2], p.POSITION_CONTROL, trajectory[0][i])
            p.setJointMotorControlArray(robot.hexapod, [3,4,5], p.POSITION_CONTROL, trajectory[1][i])
            p.setJointMotorControlArray(robot.hexapod, [6,7,8], p.POSITION_CONTROL, trajectory[2][i])
            p.setJointMotorControlArray(robot.hexapod, [9,10,11], p.POSITION_CONTROL, trajectory[3][i])
            p.setJointMotorControlArray(robot.hexapod, [12,13,14], p.POSITION_CONTROL, trajectory[4][i])
            p.setJointMotorControlArray(robot.hexapod, [15,16,17], p.POSITION_CONTROL, trajectory[5][i])
            p.stepSimulation()
            time.sleep(self.timestep)
        [x_,y_,z_] = p.getBasePositionAndOrientation(self.hexapod, self.physicsClient)[0]
        diff = [x_-x, y_-y, z_-z]
        vel = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
        q_ = p.getBasePositionAndOrientation(self.hexapod, self.physicsClient)[1]
        yaw_ = p.getEulerFromQuaternion(q_, self.hexapod)[2]
        print("v:", vel)
        print("w:", yaw_-yaw)

def sequence_change(list):
    for i in range(len(list)):
        if list[i] == 0:
            list[i] = 1
        else:
            list[i] = 0

if __name__ == "__main__":
    robot = Hexapod()
    p.stepSimulation()
    
    # foottip = list(p.getLinkState(robot.hexapod, 7)[0])
    # foottip_ = foottip
    # foottip_[0] += 0.02
    # pos = str(foottip)
    # p.addUserDebugLine(foottip, foottip_, lineColorRGB=[1,0,0], lineWidth=100, lifeTime=0)
    # p.addUserDebugText(pos, foottip, textColorRGB=[1,0,0], textSize=1, lifeTime=0)

    while True:
        robot.gait()
        sequence_change(robot.cycle_leg_number_)