"""
清洁桌面任务仿真数据采集模块
============================

模块功能:
    基于 PyBullet 物理引擎，使用 Baxter 双臂机器人执行"清洁桌面"任务，
    通过组合运动基元(primitive)生成大规模演示数据，用于分层决策与模仿学习(HDR-IL)训练。

任务流程 (单次迭代):
    1. 重置仿真环境，将 Baxter 双臂移动至初始就位姿态
    2. 张开夹爪，加载待清洁桌面和两块海绵(左右各一块)
    3. Approach: 双臂从当前位置直线运动接近桌面目标位置(保持安全高度)
    4. 闭合夹爪，通过固定约束将海绵固连到夹爪末端
    5. Press_Down: 双臂竖直下压，使海绵接触桌面表面
    6. Wipe_Horizontal: 双臂沿 y 轴水平擦拭(0.4m)
    7. Shift_Position: 双臂沿 y 轴负方向平移(0.2m)，为下一行擦拭定位
    8. Wipe_Horizontal_2: 第二次水平擦拭(0.4m)
    9. Shift_Position_2: 第二次 y 轴负方向平移(0.2m)
    10. Wipe_Circular: 双臂做圆周擦拭运动(半径0.05m)，清除局部污渍
    11. Lift_Off: 双臂竖直抬起，脱离桌面
    12. 张开夹爪，释放海绵
    13. Retract: 双臂后退回收至初始区域
    14. 记录标签和位移参数，清理仿真物体，保存数据

数据输出格式:
    CSV 文件 (clean_table_primitive_data.csv)，每行对应一个基元执行步骤的快照，
    包含以下字段:
        - Primitive:          基元名称(如 Approach, Press_Down 等)
        - right/left_gripper_pole_x/y/z_{1,2}: 左右夹爪在基元执行前(1)和后(2)的位置坐标
        - right/left_gripper_pole_q_{1-4}_{1,2}: 左右夹爪在基元执行前(1)和后(2)的四元数姿态
        - table1_x/y/z_{1,2}: 桌面在基元执行前(1)和后(2)的位置坐标
        - table1_quat{1-4}_{1,2}: 桌面在基元执行前(1)和后(2)的四元数姿态
        - x/y_displacement{1,2}: 桌面和海绵的随机位移量(域随机化)
        - grippers_open:       夹爪状态标志(1=张开, 0=闭合)
        - label:               任务标签(清洁桌面任务固定为1)

运行参数:
    - 总迭代次数: 5000
    - 物理仿真步长: 0.01s
    - 重力加速度: -9.81 m/s² (沿 z 轴负方向)
"""

import pybullet as p
import numpy as np
import pybullet_data
import pandas as pd
import primitives as pm

# 连接 PyBullet GUI 后端，启动可视化仿真窗口
p.connect(p.GUI)
# 设置 PyBullet 内置资源搜索路径(包含 plane.urdf 等标准模型)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# ============================================================================
# 数据采集字典: 存储每个基元步骤的完整状态快照
# 字段命名规则: {物体}_{属性}_{序号}
#   - _1 后缀: 基元执行前的状态 (sequence=1)
#   - _2 后缀: 基元执行后的状态 (sequence=2)
#   - q_1~q_4: 四元数的四个分量 (w, x, y, z)
# ============================================================================
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

# ============================================================================
# 仿真环境初始化
# ============================================================================
p.resetSimulation()
# 设置重力加速度: z 轴负方向 9.81 m/s²，模拟真实地球重力环境
p.setGravity(0, 0, -9.81)
# 物理仿真步长设为 0.01s (100Hz)，兼顾仿真精度与计算效率
p.setTimeStep(0.01)
# 加载地面平面，作为仿真世界的基础碰撞体
planeId = p.loadURDF("plane.urdf")
# 默认朝向: 无旋转(单位四元数)，用于桌面和海绵的初始朝向
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
# Baxter 机器人初始朝向: 无旋转
baxterStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

# 加载 Baxter 双臂机器人 URDF 模型，固定基座(不会因重力下落)
botId = p.loadURDF("baxter.xml",
                [0, 0, 0],
                baxterStartOrientation, useFixedBase=1)


def _append_grippers_open(data, value):
    """辅助函数: 对齐 grippers_open 字段与 Primitive 字段的长度。

    由于基元函数内部只记录 Primitive 和位姿数据，不记录夹爪状态，
    此函数根据 Primitive 列表与 grippers_open 列表的长度差，
    将当前夹爪状态(value)填充到 grippers_open 列表末尾，确保数据对齐。

    Args:
        data: 数据采集字典
        value: 夹爪状态 (1=张开, 0=闭合)
    """
    primitive_count = len(data["Primitive"])
    grippers_count = len(data["grippers_open"])
    diff = primitive_count - grippers_count
    for _ in range(diff):
        data["grippers_open"].append(value)


# ============================================================================
# 主循环: 执行 5000 次清洁桌面任务，每次生成一条完整的基元序列数据
# ============================================================================
for iter in range(5000):

    print(iter)

    # ------------------------------------------------------------------
    # 步骤1: 将 Baxter 双臂关节驱动至初始就位姿态
    # 关节索引对应 Baxter 的 7-DOF 手臂关节:
    #   右臂: [13,14,15,16,17,19,20] -> [S0,S1,E0,E1,W0,W1,W2]
    #   左臂: [36,37,38,39,40,42,43] -> [S0,S1,E0,E1,W0,W1,W2]
    # 目标位置使双臂展开并朝向前方桌面方向
    # ------------------------------------------------------------------
    p.setJointMotorControlArray(botId,
                                jointIndices=[13, 14, 15, 16, 17, 19, 20, 36, 37, 38, 39, 40, 42, 43],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.75, -0.9, 0, 1.8, 0, -0.9, 0, -0.75, -0.9, 0, 1.8, 0, -0.9, 0])

    # 获取 Baxter 所有关节索引列表，供后续基元函数内部使用
    revoluteJoints = pm.getRevoluteJoints(botId)

    # 短暂仿真 10 步，使关节开始向目标位置运动
    for i in range(10):
        p.stepSimulation()

    # ------------------------------------------------------------------
    # 步骤2: 张开双侧夹爪，为抓取海绵做准备
    # 关节 [29,31] 为右夹爪的两个手指关节，[52,54] 为左夹爪的两个手指关节
    # 目标位置 0.75 表示夹爪完全张开，高力矩(10000)确保夹爪可靠张开
    # ------------------------------------------------------------------
    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.75]*4,
                                forces=[10000]*4)

    # 将手腕关节(W2)旋转至 1.8 rad，使夹爪朝向适合接近桌面的角度
    # 关节 20=右臂W2, 关节 43=左臂W2
    p.setJointMotorControlArray(botId,
                                jointIndices=[20, 43],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[1.8, 1.8])

    # 仿真 100 步(1s)，等待关节运动至目标位置并稳定
    for i in range(100):
        p.stepSimulation()

    # ------------------------------------------------------------------
    # 步骤3: 域随机化 — 为桌面和海绵位置添加随机偏移
    # 每次迭代的桌面位置略有不同，增强训练数据的多样性，
    # 使模型对位置变化具有鲁棒性(域随机化策略)
    # ------------------------------------------------------------------
    x_displacement1 = np.random.uniform(0, 0.1)
    y_displacement1 = np.random.uniform(0, 0.1)
    x_displacement2 = np.random.uniform(0, 0.1)
    y_displacement2 = np.random.uniform(0, 0.1) * -1

    # ------------------------------------------------------------------
    # 步骤4: 加载仿真物体 — 桌面和海绵
    # ------------------------------------------------------------------
    # 加载待清洁桌面: 位于机器人前方约 0.9m 处，高度 0.35m，固定基座(不可移动)
    tableId1 = p.loadURDF("cleaning_table.xml",
                      [0.9 + x_displacement1, 0.0, 0.35],
                      cubeStartOrientation,
                      useFixedBase=1)

    tableId2 = None

    # 加载右侧海绵: 位于桌面右侧(y=0.45m)，高度 0.39m，非固定基座(可被夹爪抓取)
    # z=0.39 约等于桌面顶部(0.35+0.04)高度，使海绵初始放置在桌面上
    sponge_right_id = p.loadURDF("sponge.xml",
                          [0.9 + x_displacement1, 0.45 + y_displacement1, 0.39],
                          cubeStartOrientation,
                          useFixedBase=0)
    # 加载左侧海绵: 位于桌面左侧(y=-0.45m)，与右侧海绵对称
    sponge_left_id = p.loadURDF("sponge.xml",
                         [0.9 + x_displacement1, -0.45 + y_displacement1, 0.39],
                         cubeStartOrientation,
                         useFixedBase=0)

    # 设置海绵的接触动力学参数，模拟真实海绵与桌面的摩擦和弹性交互:
    #   - lateralFriction=1.0: 侧向摩擦系数，保证擦拭时海绵不会轻易滑动
    #   - contactStiffness=1000: 接触刚度，控制海绵与桌面接触时的弹性变形阻力
    #   - contactDamping=100: 接触阻尼，吸收接触时的振荡能量，防止弹跳
    p.changeDynamics(sponge_right_id, -1, lateralFriction=1.0, contactStiffness=1000, contactDamping=100)
    p.changeDynamics(sponge_left_id, -1, lateralFriction=1.0, contactStiffness=1000, contactDamping=100)

    # ==================================================================
    # z 高度计算: 精确控制夹爪在不同阶段的高度
    # ==================================================================
    # 桌面顶部 z 坐标 = 桌面基准高度(0.35) + 桌面厚度(0.04)
    table_top_z = 0.35 + 0.04
    # 夹爪末端到夹爪指尖的垂直偏移量(夹爪指垫底部相对于夹爪连杆坐标系原点)
    pad_bottom_offset = 0.08
    # 海绵自身高度
    sponge_height = 0.04
    # 接近高度: 在桌面顶部上方再抬高 0.10m 作为安全余量，
    # 防止接近过程中海绵与桌面发生碰撞
    approach_z = table_top_z + pad_bottom_offset + sponge_height + 0.10
    # 擦拭高度: 海绵底部恰好接触桌面顶部，无额外安全余量
    wipe_z = table_top_z + pad_bottom_offset + sponge_height

    # 计算左右夹爪的接近目标位置: (x, y, approach_z)
    right_approach = [0.9 + x_displacement1, 0.45 + y_displacement1, approach_z]
    left_approach = [0.9 + x_displacement1, -0.45 + y_displacement1, approach_z]

    # ==================================================================
    # 基元1: Approach — 双臂接近桌面目标位置
    # 物理意义: 夹爪从当前位置沿直线运动到海绵正上方，保持安全高度避免碰撞。
    # 内部通过逆运动学(IK)计算关节角，分 steps 步逐步插值逼近目标。
    # ==================================================================
    gripperPosition1, gripperPosition2 = pm.approach_surface(botId, right_approach, left_approach, data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Approach", is_front_grasp=False)
    # 接近阶段夹爪处于张开状态
    _append_grippers_open(data, 1)

    # 仿真 100 步，等待接近运动完成并稳定
    for i in range(100):
        p.stepSimulation()

    # ------------------------------------------------------------------
    # 步骤5: 闭合夹爪 — 准备抓取海绵
    # 目标位置设为 0(完全闭合)，但力矩仅 10N，使夹爪"软闭合"：
    # 夹爪手指会贴合海绵表面但不会施加过大夹持力，避免将海绵挤出
    # ------------------------------------------------------------------
    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0]*4,
                                forces=[10]*4)

    # 仿真 100 步，等待夹爪闭合完成
    for i in range(100):
        p.stepSimulation()

    # ------------------------------------------------------------------
    # 步骤6: 创建固定约束 — 将海绵固连到夹爪末端
    # 物理意义: 模拟夹爪"抓取"海绵的效果。通过 JOINT_FIXED 约束，
    # 使海绵与夹爪末端形成刚性连接，随夹爪一起运动。
    # ------------------------------------------------------------------
    # 右夹爪-右海绵约束:
    #   - parentFramePosition=[0,0,-0.08]: 约束点在夹爪连杆坐标系下偏移 -0.08m(沿 z 轴向下)，
    #     对应夹爪指垫底部的位置
    #   - childFramePosition=[0,0,0.04]: 约束点在海绵坐标系下偏移 +0.04m(沿 z 轴向上)，
    #     对应海绵顶部中心，使海绵顶部与夹爪指垫底部对齐
    #   - maxForce=500: 约束最大力限制，防止约束力过大导致仿真不稳定
    constraint_right = p.createConstraint(
        parentBodyUniqueId=botId,
        parentLinkIndex=pm.RIGHT_GRIPPER_LINK,
        childBodyUniqueId=sponge_right_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, -0.08],
        childFramePosition=[0, 0, 0.04])
    p.changeConstraint(constraint_right, maxForce=500)

    # 左夹爪-左海绵约束: 参数与右侧对称
    constraint_left = p.createConstraint(
        parentBodyUniqueId=botId,
        parentLinkIndex=pm.LEFT_GRIPPER_LINK,
        childBodyUniqueId=sponge_left_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, -0.08],
        childFramePosition=[0, 0, 0.04])
    p.changeConstraint(constraint_left, maxForce=500)

    # 仿真 50 步，让约束生效并稳定
    for i in range(50):
        p.stepSimulation()

    # ==================================================================
    # 基元2: Press_Down — 双臂竖直下压
    # 物理意义: 将海绵从接近高度下压至接触桌面表面，建立海绵与桌面的
    # 接触力，为后续擦拭动作提供正压力。下压距离 0.10m。
    # min_z=table_top_z+0.01 为安全下限，防止夹爪穿透桌面。
    # ==================================================================
    gripperPosition1, gripperPosition2 = pm.press_down(botId, 0.10, data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Press_Down", is_front_grasp=False, min_z=table_top_z + 0.01)
    # 下压阶段夹爪处于闭合状态(夹持海绵)
    _append_grippers_open(data, 0)

    # 仿真 100 步，等待下压运动完成并稳定接触
    for i in range(100):
        p.stepSimulation()

    # ==================================================================
    # 基元3: Wipe_Horizontal — 双臂沿 y 轴水平擦拭
    # 物理意义: 在保持正压力的同时，双臂沿 y 轴正方向平移 0.4m，
    # 模拟擦拭桌面的直线往复动作。海绵与桌面的摩擦力实现清洁效果。
    # ==================================================================
    gripperPosition1, gripperPosition2 = pm.wipe_horizontal(botId, 0.4, data, steps=12, tableId1=tableId1, tableId2=None, both=None, primitive_name="Wipe_Horizontal", is_front_grasp=False)
    _append_grippers_open(data, 0)

    for i in range(100):
        p.stepSimulation()

    # ==================================================================
    # 基元4: Shift_Position — 双臂沿 y 轴负方向平移
    # 物理意义: 擦拭完一行后，将双臂平移到下一行起始位置，
    # 类似"回车换行"操作。dy=-0.2m 表示向 y 轴负方向移动 0.2m。
    # ==================================================================
    gripperPosition1, gripperPosition2 = pm.shift_position(botId, 0.0, -0.2, data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Shift_Position", is_front_grasp=False)
    _append_grippers_open(data, 0)

    for i in range(100):
        p.stepSimulation()

    # ==================================================================
    # 基元5: Wipe_Horizontal_2 — 第二次水平擦拭
    # 物理意义: 在平移后的新位置再次沿 y 轴正方向擦拭 0.4m，
    # 覆盖桌面的第二行区域。
    # ==================================================================
    gripperPosition1, gripperPosition2 = pm.wipe_horizontal(botId, 0.4, data, steps=12, tableId1=tableId1, tableId2=None, both=None, primitive_name="Wipe_Horizontal_2", is_front_grasp=False)
    _append_grippers_open(data, 0)

    for i in range(100):
        p.stepSimulation()

    # ==================================================================
    # 基元6: Shift_Position_2 — 第二次 y 轴负方向平移
    # 物理意义: 继续向 y 轴负方向平移 0.2m，定位到第三行擦拭起始位置。
    # ==================================================================
    gripperPosition1, gripperPosition2 = pm.shift_position(botId, 0.0, -0.2, data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Shift_Position_2", is_front_grasp=False)
    _append_grippers_open(data, 0)

    for i in range(100):
        p.stepSimulation()

    # ==================================================================
    # 基元7: Wipe_Circular — 双臂圆周擦拭
    # 物理意义: 在当前位置做半径 0.05m 的圆周运动(24步完成一圈)，
    # 模拟对局部顽固污渍的旋转擦拭动作。圆周运动在 y-z 平面内进行，
    # 同时保持对桌面的正压力。
    # ==================================================================
    gripperPosition1, gripperPosition2 = pm.wipe_circular(botId, 0.05, data, steps=24, tableId1=tableId1, tableId2=None, both=None, primitive_name="Wipe_Circular", is_front_grasp=False)
    _append_grippers_open(data, 0)

    for i in range(100):
        p.stepSimulation()

    # ==================================================================
    # 基元8: Lift_Off — 双臂竖直抬起
    # 物理意义: 将海绵从桌面表面抬起 0.1m，脱离与桌面的接触，
    # 结束擦拭动作。抬起后海绵与桌面之间不再有接触力。
    # ==================================================================
    gripperPosition1, gripperPosition2 = pm.lift_off(botId, 0.1, data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Lift_Off", is_front_grasp=False)
    _append_grippers_open(data, 0)

    for i in range(100):
        p.stepSimulation()

    # ------------------------------------------------------------------
    # 步骤12: 张开夹爪，释放海绵
    # 夹爪目标位置设为 0.75(完全张开)，高力矩(10000)确保可靠释放
    # ------------------------------------------------------------------
    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.75]*4,
                                forces=[10000]*4)

    for i in range(100):
        p.stepSimulation()

    # ==================================================================
    # 基元9: Retract — 双臂后退回收
    # 物理意义: 夹爪沿 x 轴负方向后退 0.02m、沿 y 轴正方向偏移 0.02m，
    # 使双臂远离桌面区域，回到安全的待命位置，完成整个清洁任务序列。
    # ==================================================================
    gripperPosition1, gripperPosition2 = pm.retract(botId, 0.02, 0.02, data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Retract", is_front_grasp=False)
    # 回收阶段夹爪已张开(释放海绵)
    _append_grippers_open(data, 1)

    for i in range(100):
        p.stepSimulation()

    # ==================================================================
    # 数据记录与清理
    # ==================================================================
    # 清洁桌面任务的标签固定为 1(正样本)
    label = 1

    # 将本次迭代的域随机化位移量和标签填充到数据字典中。
    # 由于基元函数内部不记录这些参数，需要根据 Primitive 列表长度差
    # 计算需要填充的条目数，确保所有字段长度一致。
    new_entries = len(data["Primitive"]) - len(data["x_displacement1"])
    for _ in range(new_entries):
        data["x_displacement1"].append(x_displacement1)
        data["y_displacement1"].append(y_displacement1)
        data["x_displacement2"].append(x_displacement2)
        data["y_displacement2"].append(y_displacement2)
        data["label"].append(label)

    # 清理仿真物体: 移除桌面，为下一次迭代腾出空间
    if tableId1 is not None:
        p.removeBody(tableId1)

    # 清理仿真物体: 先移除约束，再移除海绵，避免悬挂约束导致仿真错误
    if sponge_right_id is not None:
        p.removeConstraint(constraint_right)
        p.removeConstraint(constraint_left)
        p.removeBody(sponge_right_id)
        p.removeBody(sponge_left_id)

    # 每次迭代结束后，将当前数据保存为 CSV 文件
    # 增量式保存策略: 即使程序中断，已采集的数据也不会丢失
    df = pd.DataFrame(data)
    df.to_csv("clean_table_primitive_data.csv")

# 断开 PyBullet 连接，释放仿真资源
p.disconnect()
