import pybullet as p
import numpy as np
import pybullet_data
import pandas as pd
import primitives as pm

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

data = {"Primitive": [], "right_gripper_pole_x_1": [], "right_gripper_pole_y_1": [], "right_gripper_pole_z_1": [],
        "left_gripper_pole_x_1": [], "left_gripper_pole_y_1": [], "left_gripper_pole_z_1": [],
        "right_gripper_pole_q_11": [], "right_gripper_pole_q_12": [], "right_gripper_pole_q_13": [], "right_gripper_pole_q_14": [],
        "left_gripper_pole_q_11": [], "left_gripper_pole_q_12": [], "left_gripper_pole_q_13": [], "left_gripper_pole_q_14": [],
        "table1_x_1": [], "table1_y_1": [], "table1_z_1": [], "table1_quat1_1": [], "table1_quat2_1": [], "table1_quat3_1": [], "table1_quat4_1": [],
        "right_gripper_pole_x_2": [], "right_gripper_pole_y_2": [], "right_gripper_pole_z_2": [],
        "left_gripper_pole_x_2": [], "left_gripper_pole_y_2": [], "left_gripper_pole_z_2": [],
        "right_gripper_pole_q_21": [], "right_gripper_pole_q_22": [], "right_gripper_pole_q_23": [], "right_gripper_pole_q_24": [],
        "left_gripper_pole_q_21": [], "left_gripper_pole_q_22": [], "left_gripper_pole_q_23": [], "left_gripper_pole_q_24": [],
        "table1_x_2": [], "table1_y_2": [], "table1_z_2": [], "table1_quat1_2": [], "table1_quat2_2": [], "table1_quat3_2": [], "table1_quat4_2": [],
        "x_displacement1": [], "y_displacement1": [], "x_displacement2": [], "y_displacement2": [], "grippers_open": [], "label": []}

p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.setTimeStep(0.01)
planeId = p.loadURDF("plane.urdf")
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
baxterStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

botId = p.loadURDF("baxter.xml",
                [0, 0, 0],
                baxterStartOrientation, useFixedBase=1)


def _append_grippers_open(data, value):
    primitive_count = len(data["Primitive"])
    grippers_count = len(data["grippers_open"])
    diff = primitive_count - grippers_count
    for _ in range(diff):
        data["grippers_open"].append(value)


for iter in range(5000):

    print(iter)

    p.setJointMotorControlArray(botId,
                                jointIndices=[13, 14, 15, 16, 17, 19, 20, 36, 37, 38, 39, 40, 42, 43],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.75, -0.9, 0, 1.8, 0, -0.9, 0, -0.75, -0.9, 0, 1.8, 0, -0.9, 0])

    revoluteJoints = pm.getRevoluteJoints(botId)

    for i in range(10):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.75]*4,
                                forces=[10000]*4)

    p.setJointMotorControlArray(botId,
                                jointIndices=[20, 43],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[1.8, 1.8])

    for i in range(100):
        p.stepSimulation()

    x_displacement1 = np.random.uniform(0, 0.1)
    y_displacement1 = np.random.uniform(0, 0.1)
    x_displacement2 = np.random.uniform(0, 0.1)
    y_displacement2 = np.random.uniform(0, 0.1) * -1

    tableId1 = p.loadURDF("glass_panel.xml",
                      [0.85 + x_displacement1, 0.0 + y_displacement1, 0],
                      cubeStartOrientation,
                      useFixedBase=1)

    tableId2 = None

    glass_front_y = 0.0 + y_displacement1 + 0.025 + 0.08
    glass_center_z = 0.5

    right_approach = [0.85 + x_displacement1, glass_front_y + 0.25, glass_center_z]
    left_approach = [0.85 + x_displacement1, glass_front_y - 0.25, glass_center_z]

    gripperPosition1, gripperPosition2 = pm.approach_surface(botId, right_approach, left_approach, data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Approach", is_front_grasp=False)
    _append_grippers_open(data, 1)

    for i in range(100):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0]*4,
                                forces=[10]*4)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = pm.press_down(botId, 0.03, data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Press_Down", is_front_grasp=False)
    _append_grippers_open(data, 0)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = pm.wipe_vertical(botId, 0.3, data, steps=12, tableId1=tableId1, tableId2=None, both=None, primitive_name="Wipe_Vertical", is_front_grasp=False)
    _append_grippers_open(data, 0)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = pm.shift_position(botId, 0.0, -0.15, data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Shift_Position", is_front_grasp=False)
    _append_grippers_open(data, 0)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = pm.wipe_vertical(botId, 0.3, data, steps=12, tableId1=tableId1, tableId2=None, both=None, primitive_name="Wipe_Vertical_2", is_front_grasp=False)
    _append_grippers_open(data, 0)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = pm.shift_position(botId, 0.0, -0.15, data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Shift_Position_2", is_front_grasp=False)
    _append_grippers_open(data, 0)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = pm.wipe_circular(botId, 0.06, data, steps=24, tableId1=tableId1, tableId2=None, both=None, primitive_name="Wipe_Circular", is_front_grasp=False)
    _append_grippers_open(data, 0)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = pm.lift_off(botId, 0.1, data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Lift_Off", is_front_grasp=False)
    _append_grippers_open(data, 0)

    for i in range(100):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.75]*4,
                                forces=[10000]*4)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = pm.retract(botId, 0.02, 0.02, data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Retract", is_front_grasp=False)
    _append_grippers_open(data, 1)

    for i in range(100):
        p.stepSimulation()

    label = 1

    new_entries = len(data["Primitive"]) - len(data["x_displacement1"])
    for _ in range(new_entries):
        data["x_displacement1"].append(x_displacement1)
        data["y_displacement1"].append(y_displacement1)
        data["x_displacement2"].append(x_displacement2)
        data["y_displacement2"].append(y_displacement2)
        data["label"].append(label)

    if tableId1 is not None:
        p.removeBody(tableId1)

    df = pd.DataFrame(data)
    df.to_csv("clean_glass_primitive_data.csv")

p.disconnect()
