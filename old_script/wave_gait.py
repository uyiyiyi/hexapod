import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from PolynomialInterpolation import Polynomial5Interpolation

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("urdf/plane.urdf")
cubeStartPos = [0,0,0.4]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,np.pi/2])
hexapod = p.loadURDF("urdf/robot_hexapod.urdf",cubeStartPos, cubeStartOrientation)

time.sleep(10)

p.setJointMotorControlArray(hexapod, [0,3,6,9,12,15], p.POSITION_CONTROL, [-np.pi/12]*6)
p.stepSimulation()
# leg 1 left-back
# leg 2 left-mid
# leg 3 left-front
# leg 4 right-front
# leg 5 right-mid
# leg 6 right-back

def singe_leg(leg_i):
	joint_index = [3*leg_i, 3*leg_i + 1, 3*leg_i + 2]
	index = 3 * leg_i
	other_joint = [0, 3, 6, 9, 12, 15]
	other_joint.remove(index)
	other_thigh = [1, 4, 7, 10, 13, 16]
	other_thigh.remove(index+1)

	q_given = np.array([[-np.pi/12, 0, np.pi/12],
						[0, np.pi/6, 0],
						[0, 0, 0]]).transpose()
	t_given = np.array([0, 1, 2]).transpose()

	q_given1 = np.array([[0, np.pi/36, np.pi/18, np.pi/36, 0],
						[0, np.pi/10, 0, -np.pi/10, 0],
						[0, 0, 0, 0, 0]]).transpose()
	t_given1 = np.array([0, 0.5, 1, 1.5, 2]).transpose()

	# time for interpolation
	t = np.linspace(t_given[0], t_given[-1], 480)

	polynomial5_interpolation = Polynomial5Interpolation('Polynomial5', q_given, t_given)
	body_joint_trajectory = np.zeros((t.shape[0], 3)) # N x 3 array: position, velocity, acceleration

	for i in range(t.shape[0]):
		body_joint_trajectory[i,:] = polynomial5_interpolation.getPosition(t[i])

	polynomial5_interpolation = Polynomial5Interpolation('Polynomial5', q_given1, t_given1)
	thigh_and_shank_joint_trajectory = np.zeros((t.shape[0], 3)) # N x 3 array: position, velocity, acceleration

	for i in range(t.shape[0]):
		thigh_and_shank_joint_trajectory[i,:] = polynomial5_interpolation.getPosition(t[i])

	js = [0] * 5
	for i, joint in enumerate(other_joint):
		js[i] = p.getJointState(hexapod, joint)[0]
	print(js)

	other_joint_trajectory = np.zeros((t.shape[0], 5))
	for i in range(t.shape[0]):
		list1 = [-i * np.pi / 14400] * 5 
		other_joint_trajectory[i,:] = np.sum([list1, js], axis=0).tolist()

	# print(index)
	# print("---------------------------------")
	# print(other_joint_trajectory)

	for i in range(t.shape[0]):
		p.setJointMotorControlArray(hexapod, joint_index, p.POSITION_CONTROL, [body_joint_trajectory[i,0], thigh_and_shank_joint_trajectory[i,0], thigh_and_shank_joint_trajectory[i,0]], [body_joint_trajectory[i,1], thigh_and_shank_joint_trajectory[i,1], thigh_and_shank_joint_trajectory[i,1]])
		p.setJointMotorControlArray(hexapod, other_joint, p.POSITION_CONTROL, other_joint_trajectory[i])
		p.setJointMotorControlArray(hexapod, other_thigh, p.POSITION_CONTROL, [np.pi/360]*5)
		p.stepSimulation()
		# time.sleep(1./960.)

def single_wave_gait():
	for i in range(6):
		singe_leg(i)

for i in range(100):
	single_wave_gait()
# jn = p.getNumJoints(hexapod)
# print(jn)

# for i in range(jn):
# 	ji = p.getJointInfo(hexapod,i)
# 	print(ji)

# js = p.getJointStates(hexapod, [i for i in range(jn)])
# print(js)

# for i in range (10000):
# 	p.setJointMotorControlArray(hexapod, [15,16,17], p.POSITION_CONTROL, [np.pi/6, np.pi/3, np.pi / 6])
# 	p.setJointMotorControlArray(hexapod, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], p.POSITION_CONTROL, [0]*15)
# 	if i%100 == 0:
# 		for j in range(jn):
# 			js = p.getJointState(hexapod, j)
# 		# 	print(js[0])
# 		# print("------------------------------------------------------")
# 	p.stepSimulation()
# 	time.sleep(1./240.)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(hexapod)
# print(cubePos,cubeOrn)
# p.disconnect()
