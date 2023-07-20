# 本脚本需要实现六足机器人跟踪落足点的功能
import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import math

NUM_LEGS = 6
CYCLE_LENGTH = 100 # 50 time steps
HALF_CYCLE_LENGTH = int(CYCLE_LENGTH/2)
LEG_LIFT_HEIGHT = 0.1


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setGravity(0,0,-9.8)
p.setRealTimeSimulation(1, physicsClient)
# timestep = 0.01
# p.setTimeStep(timestep, physicsClient)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
planeId = p.loadURDF("urdf/plane.urdf", baseOrientation=p.getQuaternionFromEuler([0,0,0]))
# self.planeId = p.loadURDF("terrain.urdf", baseOrientation=p.getQuaternionFromEuler([0,0,0]))
hexapod = p.loadURDF("robot_liuzu/urdf/robot_liuzu.urdf",[0,0,0.46], p.getQuaternionFromEuler([0,0,np.pi/2]))
p.setJointMotorControlArray(hexapod, [i for i in range(18)], p.POSITION_CONTROL, [0.0] * 18)
lbb_debug = p.addUserDebugParameter("rf1", -2, 2, 0, physicsClient)
lbt_debug = p.addUserDebugParameter("rf2",  -2, 2, 0, physicsClient)
lbs_debug = p.addUserDebugParameter("rf3",  -2, 2, 0, physicsClient)
lmb_debug = p.addUserDebugParameter("lf1", -2, 2, 0, physicsClient)
lmt_debug = p.addUserDebugParameter("lf2",  -2, 2, 0, physicsClient)
lms_debug = p.addUserDebugParameter("lf3",  -2, 2, 0, physicsClient)
lfb_debug = p.addUserDebugParameter("rm1", -2, 2, 0, physicsClient)
lft_debug = p.addUserDebugParameter("rm2",  -2, 2, 0, physicsClient)
lfs_debug = p.addUserDebugParameter("rm3",  -2, 2, 0, physicsClient)

rfb_debug = p.addUserDebugParameter("lm1", -2, 2, 0, physicsClient)
rft_debug = p.addUserDebugParameter("lm2",  -2, 2, 0, physicsClient)
rfs_debug = p.addUserDebugParameter("lm3",  -2, 2, 0, physicsClient)
rmb_debug = p.addUserDebugParameter("rb1", -2, 2, 0, physicsClient)
rmt_debug = p.addUserDebugParameter("rb2",  -2, 2, 0, physicsClient)
rms_debug = p.addUserDebugParameter("rb3",  -2, 2, 0, physicsClient)
rbb_debug = p.addUserDebugParameter("lb1", -2, 2, 0, physicsClient)
rbt_debug = p.addUserDebugParameter("lb2",  -2, 2, 0, physicsClient)
rbs_debug = p.addUserDebugParameter("lb3",  -2, 2, 0, physicsClient)


if __name__ == "__main__":
    while True:
        lbb = p.readUserDebugParameter(lbb_debug, physicsClient)
        lbt = p.readUserDebugParameter(lbt_debug, physicsClient)
        lbs = p.readUserDebugParameter(lbs_debug, physicsClient)
        lmb = p.readUserDebugParameter(lmb_debug, physicsClient)
        lmt = p.readUserDebugParameter(lmt_debug, physicsClient)
        lms = p.readUserDebugParameter(lms_debug, physicsClient)
        lfb = p.readUserDebugParameter(lfb_debug, physicsClient)
        lft = p.readUserDebugParameter(lft_debug, physicsClient)
        lfs = p.readUserDebugParameter(lfs_debug, physicsClient)
        rfb = p.readUserDebugParameter(rfb_debug, physicsClient)
        rft = p.readUserDebugParameter(rft_debug, physicsClient)
        rfs = p.readUserDebugParameter(rfs_debug, physicsClient)
        rmb = p.readUserDebugParameter(rmb_debug, physicsClient)
        rmt = p.readUserDebugParameter(rmt_debug, physicsClient)
        rms = p.readUserDebugParameter(rms_debug, physicsClient)
        rbb = p.readUserDebugParameter(rbb_debug, physicsClient)
        rbt = p.readUserDebugParameter(rbt_debug, physicsClient)
        rbs = p.readUserDebugParameter(rbs_debug, physicsClient)
        for i, joint_pos in enumerate([lbb,lbt,lbs,lmb,lmt,lms,lfb,lft,lfs,rfb,rft,rfs,rmb,rmt,rms,rbb,rbt,rbs]):
            if joint_pos != 0:
                p.setJointMotorControl2(hexapod, i, p.POSITION_CONTROL, joint_pos, force=100)

        # p.stepSimulation()
        