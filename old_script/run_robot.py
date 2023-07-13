import kinematic_model
import stance_controller_hexapod
import numpy as np
import time

if __name__ == "__main__":
    k = kinematic_model.Kinematic()
    sc = stance_controller_hexapod.StanceController()
    # time.sleep(1)
    while True:
        F = sc.controller(k)
        # input("Press Enter to continue...")
        # torques = sc.MapContactForceToJointTorques(k, F)
        # print(F.transpose())
        # print(torques)
        # sc.actuate_motors(torques)
        # sc.apply_force(F)
        # time.sleep(1)
        # sc.renderForce(k, F)


