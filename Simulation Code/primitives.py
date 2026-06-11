"""双臂机器人仿真运动原语模块。

本模块定义了基于 PyBullet 物理引擎的双臂机器人（如 Baxter）仿真运动原语（primitives），
用于构建机器人操作任务的基本动作序列。模块提供以下核心功能：

1. **碰撞检测**：检测机器人自身碰撞。
2. **运动学查询**：获取连杆位置、关节状态、旋转关节索引等。
3. **数据采集**：记录夹爪和桌面位姿数据，用于仿真回放与分析。
4. **运动原语**：包括前向抓取、接近表面、水平/垂直/圆弧擦拭、
   按压、抬升、侧移、连接、提升、侧移抓取、回撤、下降等基本动作。

典型使用场景为 HDR-IL（分层决策模仿学习）框架中的仿真数据生成，
通过组合不同的运动原语来构建复杂的机器人操作任务。

依赖:
    pybullet: 物理仿真引擎，提供机器人模型加载、逆运动学求解、
              关节控制及碰撞检测等核心功能。

作者:
    HDR-IL 项目组
"""

import pybullet as p

RIGHT_GRIPPER_LINK = 28   # 右夹爪连杆在 URDF 模型中的索引号，对应 Baxter 右手末端执行器
LEFT_GRIPPER_LINK = 51    # 左夹爪连杆在 URDF 模型中的索引号，对应 Baxter 左手末端执行器
PAD_BOTTOM_OFFSET = 0.08  # 夹爪垫底部的垂直偏移量（单位：米），用于计算夹爪有效底面高度


def check_self_collision(botId):
    """检测机器人是否发生自身碰撞。

    通过 PyBullet 的接触点检测接口查询机器人各连杆之间的自碰撞情况。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符（body unique ID）。

    Returns:
        bool: 若存在自身碰撞返回 True，否则返回 False。

    Example:
        >>> botId = p.loadURDF("baxter.urdf")
        >>> has_collision = check_self_collision(botId)
        >>> print(f"自碰撞: {has_collision}")
    """
    contacts = p.getContactPoints(botId, botId)  # 查询同一机器人内部的接触点
    return len(contacts) > 0


def get_link_z(botId, link_index):
    """获取指定连杆在世界坐标系下的 Z 轴高度。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        link_index (int): 目标连杆的索引号。

    Returns:
        float: 该连link质心在世界坐标系下的 Z 轴坐标（高度值，单位：米）。

    Example:
        >>> z_height = get_link_z(botId, RIGHT_GRIPPER_LINK)
        >>> print(f"右夹爪高度: {z_height:.4f} m")
    """
    link_state = p.getLinkState(botId, link_index)  # 获取连杆状态，索引0为质心位置，索引1为质心姿态
    return link_state[0][2]  # 返回质心位置的 Z 分量


def clamp_gripper_z(gripper_z, min_z):
    """限制夹爪 Z 轴高度不低于最小值，防止夹爪穿透桌面。

    考虑夹爪垫底部的偏移量，确保夹爪有效底面不低于指定的最小高度。

    Args:
        gripper_z (float): 夹爪当前的 Z 轴坐标（单位：米）。
        min_z (float): 允许的最小 Z 轴高度（单位：米），通常为桌面高度。

    Returns:
        float: 经钳位后的夹爪 Z 轴坐标。若有效底面低于 min_z，
               则返回 min_z + PAD_BOTTOM_OFFSET；否则返回原始值。

    Example:
        >>> clamped = clamp_gripper_z(0.85, 0.80)
        >>> print(f"钳位后高度: {clamped:.4f} m")
    """
    effective_bottom = gripper_z - PAD_BOTTOM_OFFSET  # 计算夹爪有效底面高度（减去垫底部偏移）
    if effective_bottom < min_z:  # 若有效底面低于最小允许高度
        return min_z + PAD_BOTTOM_OFFSET  # 将夹爪抬升至刚好不穿透桌面的高度
    return gripper_z


def getLinkNames(model_id):
    """获取机器人模型中所有连杆的名称到索引的映射。

    包括基座（base link，索引为 -1）和所有关节对应的连杆。

    Args:
        model_id (int): PyBullet 中加载的机器人模型唯一标识符。

    Returns:
        dict[str, int]: 连杆名称到索引的映射字典。基座连杆的索引为 -1，
            其余连杆索引从 0 开始递增。

    Example:
        >>> link_map = getLinkNames(botId)
        >>> print(link_map["right_gripper"])
        28
    """
    _link_name_to_index = {p.getBodyInfo(model_id)[0].decode('UTF-8'): -1}  # 基座连杆，索引为 -1
    for _id in range(p.getNumJoints(model_id)):
        _name = p.getJointInfo(model_id, _id)[12].decode('UTF-8')  # 索引12为连杆名称的字节串
        _link_name_to_index[_name] = _id
    return _link_name_to_index


def getJointNames(botId):
    """打印机器人模型中所有关节的索引和名称（调试用）。

    依次遍历所有关节，输出每个关节的索引号和名称。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。

    Example:
        >>> getJointNames(botId)
        (0, b'joint1')
        (1, b'joint2')
        ...
    """
    for i in range(p.getNumJoints(botId)):
        print(p.getJointInfo(botId, i)[0:2])  # 索引0为关节索引，索引1为关节名称


def getRevoluteJoints(botId):
    """获取机器人模型中所有旋转关节（revolute joint）的索引列表。

    PyBullet 中关节类型编码：0 = 旋转关节（REVOLUTE），
    1 = 棱柱关节（PRISMATIC），2 = 球关节（SPHERICAL），4 = 固定关节（FIXED）。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。

    Returns:
        list[int]: 所有旋转关节的索引列表，按 URDF 中定义的顺序排列。

    Example:
        >>> revolute_joints = getRevoluteJoints(botId)
        >>> print(f"旋转关节数: {len(revolute_joints)}")
    """
    revoluteJoints = []
    for i in range(p.getNumJoints(botId)):
        if p.getJointInfo(botId, i)[2] == 0:  # 关节类型为 0 表示旋转关节（REVOLUTE）
            revoluteJoints.append(i)
    return revoluteJoints


def getJointStates(botId):
    """获取机器人所有旋转关节的当前关节角度。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。

    Returns:
        list[float]: 所有旋转关节的当前关节角度列表（单位：弧度），
            顺序与 getRevoluteJoints() 返回的索引顺序一致。

    Example:
        >>> states = getJointStates(botId)
        >>> print(f"关节角度: {states}")
    """
    jointStates = []
    revoluteJoints = getRevoluteJoints(botId)
    for joint in revoluteJoints:
        jointStates.append(p.getJointState(botId, joint)[0])  # 索引0为关节位置（角度）
    return jointStates


def _append_gripper_data(data, botId, suffix):
    """将双夹爪的位姿数据追加到数据字典中（内部辅助函数）。

    采集左右夹爪连杆的位置（3D坐标）和姿态（四元数），
    并以指定后缀作为键名追加到数据字典的对应列表中。

    Args:
        data (dict[str, list]): 数据采集字典，键为数据名称，值为数据列表。
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        suffix (str): 键名后缀，用于区分不同采集阶段（如 "1" 或 "2"）。

    Note:
        数据键命名规则：
        - 位置: `{side}_gripper_pole_{axis}_{suffix}`
        - 姿态: `{side}_gripper_pole_q_{suffix}{quat_index}`
        其中 side 为 "right" 或 "left"，axis 为 "x"/"y"/"z"，quat_index 为 1~4。
    """
    right_pos = p.getLinkState(botId, RIGHT_GRIPPER_LINK)[0]  # 右夹爪位置 [x, y, z]
    right_orn = p.getLinkState(botId, RIGHT_GRIPPER_LINK)[1]  # 右夹爪姿态四元数 [qx, qy, qz, qw]
    left_pos = p.getLinkState(botId, LEFT_GRIPPER_LINK)[0]    # 左夹爪位置 [x, y, z]
    left_orn = p.getLinkState(botId, LEFT_GRIPPER_LINK)[1]    # 左夹爪姿态四元数 [qx, qy, qz, qw]

    data[f"right_gripper_pole_x_{suffix}"].append(right_pos[0])
    data[f"right_gripper_pole_y_{suffix}"].append(right_pos[1])
    data[f"right_gripper_pole_z_{suffix}"].append(right_pos[2])
    data[f"left_gripper_pole_x_{suffix}"].append(left_pos[0])
    data[f"left_gripper_pole_y_{suffix}"].append(left_pos[1])
    data[f"left_gripper_pole_z_{suffix}"].append(left_pos[2])

    data[f"right_gripper_pole_q_{suffix}1"].append(right_orn[0])  # 四元数分量 qx
    data[f"right_gripper_pole_q_{suffix}2"].append(right_orn[1])  # 四元数分量 qy
    data[f"right_gripper_pole_q_{suffix}3"].append(right_orn[2])  # 四元数分量 qz
    data[f"right_gripper_pole_q_{suffix}4"].append(right_orn[3])  # 四元数分量 qw
    data[f"left_gripper_pole_q_{suffix}1"].append(left_orn[0])
    data[f"left_gripper_pole_q_{suffix}2"].append(left_orn[1])
    data[f"left_gripper_pole_q_{suffix}3"].append(left_orn[2])
    data[f"left_gripper_pole_q_{suffix}4"].append(left_orn[3])


def _append_table_data(data, tableId, table_num, suffix):
    """将桌面位姿数据追加到数据字典中（内部辅助函数）。

    采集指定桌面的位置（3D坐标）和姿态（四元数），
    并以桌面编号和后缀作为键名追加到数据字典中。

    Args:
        data (dict[str, list]): 数据采集字典，键为数据名称，值为数据列表。
        tableId (int): PyBullet 中加载的桌面模型唯一标识符。
        table_num (int): 桌面编号（1 或 2），用于区分不同桌面。
        suffix (str): 键名后缀，用于区分不同采集阶段。

    Note:
        数据键命名规则：
        - 位置: `table{N}_x_{suffix}`, `table{N}_y_{suffix}`, `table{N}_z_{suffix}`
        - 姿态: `table{N}_quat{I}_{suffix}`，I 为四元数分量索引 1~4。
    """
    pos, orn = p.getBasePositionAndOrientation(tableId)  # 获取桌面基座的位置和姿态
    data[f"table{table_num}_x_{suffix}"].append(pos[0])
    data[f"table{table_num}_y_{suffix}"].append(pos[1])
    data[f"table{table_num}_z_{suffix}"].append(pos[2])
    data[f"table{table_num}_quat1_{suffix}"].append(orn[0])
    data[f"table{table_num}_quat2_{suffix}"].append(orn[1])
    data[f"table{table_num}_quat3_{suffix}"].append(orn[2])
    data[f"table{table_num}_quat4_{suffix}"].append(orn[3])


def get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=None, tableId2=None, is_front_grasp=True, both=True):
    """采集并记录当前仿真步的运动原语数据。

    根据抓取类型和序列号，将夹爪位姿数据（以及可选的桌面位姿数据）
    追加到数据字典中。sequence=1 表示动作开始前采集，sequence=2 表示动作完成后采集。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        data (dict[str, list]): 数据采集字典，键为数据名称，值为数据列表。
        primitive_name (str): 当前运动原语的名称标识。
        sequence (int, optional): 采集序列号，1 表示动作前，2 表示动作后。默认为 1。
        tableId1 (int | None, optional): 第一张桌面的模型标识符。默认为 None。
        tableId2 (int | None, optional): 第二张桌面的模型标识符。默认为 None。
        is_front_grasp (bool, optional): 是否为前向抓取模式。
            若为 True，仅采集夹爪数据；若为 False，同时采集桌面数据。默认为 True。
        both (bool, optional): 是否同时记录两张桌面的数据（仅在 is_front_grasp=False 时生效）。
            默认为 True。

    Example:
        >>> get_primitive_data(botId, data, "approach", sequence=1,
        ...                    tableId1=table1, tableId2=table2,
        ...                    is_front_grasp=False, both=True)
    """
    if sequence == 1:
        data["Primitive"].append(primitive_name)  # 仅在动作开始时记录原语名称

    suffix = "1" if sequence == 1 else "2"  # 后缀用于区分动作前后的数据

    if is_front_grasp:
        _append_gripper_data(data, botId, suffix)  # 前向抓取模式：仅记录夹爪数据
    else:
        _append_gripper_data(data, botId, suffix)  # 非前向抓取模式：记录夹爪 + 桌面数据

        if both and tableId1 is not None and tableId2 is not None:  # 同时记录两张桌面
            _append_table_data(data, tableId1, 1, suffix)
            _append_table_data(data, tableId2, 2, suffix)
        elif tableId1 is not None:  # 仅记录第一张桌面
            _append_table_data(data, tableId1, 1, suffix)
        elif tableId2 is not None:  # 仅记录第二张桌面
            _append_table_data(data, tableId2, 2, suffix)


def front_grasp(botId, rightCoords, leftCoords, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, is_front_grasp=False):
    """执行前向抓取动作，双臂沿 X/Y 平面逐步移动到目标位置。

    该原语使左右夹爪从当前位置沿 X 和 Y 方向线性插值移动到目标坐标，
    Z 轴高度保持不变（由目标坐标指定）。每步执行逆运动学求解并驱动关节。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        rightCoords (list[float]): 右夹爪目标位置 [x, y, z]（单位：米）。
        leftCoords (list[float]): 左夹爪目标位置 [x, y, z]（单位：米）。
        data (dict[str, list]): 数据采集字典。
        steps (int, optional): 插值步数，步数越多运动越平滑。默认为 10。
        tableId1 (int | None, optional): 第一张桌面的模型标识符。默认为 None。
        tableId2 (int | None, optional): 第二张桌面的模型标识符。默认为 None。
        both (bool | None, optional): 是否同时记录两张桌面数据。默认为 None。
        primitive_name (str | None, optional): 运动原语名称标识。默认为 None。
        is_front_grasp (bool, optional): 是否为前向抓取模式。默认为 False。

    Returns:
        tuple[list[float], list[float]]: 右臂和左臂最终的逆运动学解
            （关节角度列表，单位：弧度）。

    Example:
        >>> right_ik, left_ik = front_grasp(
        ...     botId, [0.6, -0.3, 0.9], [0.6, 0.3, 0.9],
        ...     data, steps=15, primitive_name="front_grasp")
    """
    revoluteJoints = getRevoluteJoints(botId)

    right_pos = list(p.getLinkState(botId, RIGHT_GRIPPER_LINK)[0])  # 获取右夹爪当前位置
    left_pos = list(p.getLinkState(botId, LEFT_GRIPPER_LINK)[0])    # 获取左夹爪当前位置

    iter_right_x = (rightCoords[0] - right_pos[0]) / steps  # 右夹爪 X 方向每步增量
    iter_left_x = (leftCoords[0] - left_pos[0]) / steps     # 左夹爪 X 方向每步增量
    iter_right_y = (rightCoords[1] - right_pos[1]) / steps  # 右夹爪 Y 方向每步增量
    iter_left_y = (leftCoords[1] - left_pos[1]) / steps     # 左夹爪 Y 方向每步增量

    for _ in range(steps):
        right_pos[0] += iter_right_x   # 右夹爪 X 递增
        left_pos[0] += iter_left_x     # 左夹爪 X 递增
        right_pos[1] += iter_right_y   # 右夹爪 Y 递增
        left_pos[1] += iter_left_y     # 左夹爪 Y 递增

        gripperPosition1 = p.calculateInverseKinematics(botId, RIGHT_GRIPPER_LINK, [right_pos[0], right_pos[1], rightCoords[2]], maxNumIterations=1000)  # 右臂逆运动学求解，Z 轴保持目标高度
        gripperPosition2 = p.calculateInverseKinematics(botId, LEFT_GRIPPER_LINK, [left_pos[0], left_pos[1], leftCoords[2]], maxNumIterations=1000)  # 左臂逆运动学求解

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[0:10],   # 右臂的 10 个旋转关节
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition1[0:10])

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[10:],    # 左臂的 9 个旋转关节
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition2[10:])

        for _ in range(10):
            p.stepSimulation()  # 推进仿真 10 步，等待关节运动到位

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

    return gripperPosition1, gripperPosition2


def front_grasp2(botId, rightCoords, leftCoords, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, is_front_grasp=False):
    """执行前向抓取动作变体，双臂仅沿 X 轴方向逐步移动。

    与 front_grasp 不同，此变体仅沿 X 方向线性插值，
    Y 和 Z 坐标直接使用目标值。适用于仅需要 X 方向接近的场景。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        rightCoords (list[float]): 右夹爪目标位置 [x, y, z]（单位：米）。
        leftCoords (list[float]): 左夹爪目标位置 [x, y, z]（单位：米）。
        data (dict[str, list]): 数据采集字典。
        steps (int, optional): 插值步数。默认为 10。
        tableId1 (int | None, optional): 第一张桌面的模型标识符。默认为 None。
        tableId2 (int | None, optional): 第二张桌面的模型标识符。默认为 None。
        both (bool | None, optional): 是否同时记录两张桌面数据。默认为 None。
        primitive_name (str | None, optional): 运动原语名称标识。默认为 None。
        is_front_grasp (bool, optional): 是否为前向抓取模式。默认为 False。

    Returns:
        tuple[list[float], list[float]]: 右臂和左臂最终的逆运动学解。

    Example:
        >>> right_ik, left_ik = front_grasp2(
        ...     botId, [0.7, -0.2, 0.85], [0.7, 0.2, 0.85],
        ...     data, steps=12, primitive_name="front_grasp2")
    """
    revoluteJoints = getRevoluteJoints(botId)

    x_prev = p.getLinkState(botId, RIGHT_GRIPPER_LINK)[0][0]  # 获取右夹爪当前 X 坐标
    x_iter = (rightCoords[0] - x_prev) / steps  # X 方向每步增量

    for _ in range(steps):
        x_prev += x_iter  # X 坐标逐步递增
        gripperPosition1 = p.calculateInverseKinematics(botId, RIGHT_GRIPPER_LINK, [x_prev, rightCoords[1], rightCoords[2]], maxNumIterations=1000)  # Y/Z 直接使用目标值
        gripperPosition2 = p.calculateInverseKinematics(botId, LEFT_GRIPPER_LINK, [x_prev, leftCoords[1], leftCoords[2]], maxNumIterations=1000)  # 左右臂 X 同步移动

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[0:10],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition1[0:10])

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[10:],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition2[10:])

        for _ in range(10):
            p.stepSimulation()

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

    return gripperPosition1, gripperPosition2


def approach_surface(botId, rightCoords, leftCoords, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, is_front_grasp=False):
    """执行接近表面动作，双臂沿 X/Y/Z 三轴线性插值移动到目标位置。

    与 front_grasp 不同，此原语同时在三个轴方向进行插值，
    适用于夹爪需要从远处逐步接近目标表面的场景。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        rightCoords (list[float]): 右夹爪目标位置 [x, y, z]（单位：米）。
        leftCoords (list[float]): 左夹爪目标位置 [x, y, z]（单位：米）。
        data (dict[str, list]): 数据采集字典。
        steps (int, optional): 插值步数。默认为 10。
        tableId1 (int | None, optional): 第一张桌面的模型标识符。默认为 None。
        tableId2 (int | None, optional): 第二张桌面的模型标识符。默认为 None。
        both (bool | None, optional): 是否同时记录两张桌面数据。默认为 None。
        primitive_name (str | None, optional): 运动原语名称标识。默认为 None。
        is_front_grasp (bool, optional): 是否为前向抓取模式。默认为 False。

    Returns:
        tuple[list[float], list[float]]: 右臂和左臂最终的逆运动学解。

    Example:
        >>> right_ik, left_ik = approach_surface(
        ...     botId, [0.6, -0.3, 0.82], [0.6, 0.3, 0.82],
        ...     data, steps=15, primitive_name="approach")
    """
    revoluteJoints = getRevoluteJoints(botId)

    right_pos = list(p.getLinkState(botId, RIGHT_GRIPPER_LINK)[0])  # 右夹爪当前位置
    left_pos = list(p.getLinkState(botId, LEFT_GRIPPER_LINK)[0])    # 左夹爪当前位置

    iter_right_x = (rightCoords[0] - right_pos[0]) / steps  # X 方向每步增量
    iter_left_x = (leftCoords[0] - left_pos[0]) / steps
    iter_right_y = (rightCoords[1] - right_pos[1]) / steps  # Y 方向每步增量
    iter_left_y = (leftCoords[1] - left_pos[1]) / steps
    iter_right_z = (rightCoords[2] - right_pos[2]) / steps  # Z 方向每步增量
    iter_left_z = (leftCoords[2] - left_pos[2]) / steps

    for _ in range(steps):
        right_pos[0] += iter_right_x
        left_pos[0] += iter_left_x
        right_pos[1] += iter_right_y
        left_pos[1] += iter_left_y
        right_pos[2] += iter_right_z  # Z 轴也逐步变化
        left_pos[2] += iter_left_z

        gripperPosition1 = p.calculateInverseKinematics(botId, RIGHT_GRIPPER_LINK, right_pos, maxNumIterations=1000)  # 使用三轴插值后的位置求解 IK
        gripperPosition2 = p.calculateInverseKinematics(botId, LEFT_GRIPPER_LINK, left_pos, maxNumIterations=1000)

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[0:10],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition1[0:10])

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[10:],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition2[10:])

        for _ in range(10):
            p.stepSimulation()

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

    return gripperPosition1, gripperPosition2


def wipe_horizontal(botId, wipe_width, data, steps=12, tableId1=None, tableId2=None, both=None, primitive_name=None, is_front_grasp=False):
    """执行水平擦拭动作，双臂沿 Y 轴方向往复运动。

    夹爪先沿 Y 轴正方向移动指定宽度，再沿 Y 轴负方向返回原位，
    形成一个完整的水平往复擦拭轨迹。运动过程中保持夹爪姿态不变。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        wipe_width (float): 水平擦拭宽度（单位：米），即 Y 轴方向的单程移动距离。
        data (dict[str, list]): 数据采集字典。
        steps (int, optional): 单程擦拭的插值步数。默认为 12。
        tableId1 (int | None, optional): 第一张桌面的模型标识符。默认为 None。
        tableId2 (int | None, optional): 第二张桌面的模型标识符。默认为 None。
        both (bool | None, optional): 是否同时记录两张桌面数据。默认为 None。
        primitive_name (str | None, optional): 运动原语名称标识。默认为 None。
        is_front_grasp (bool, optional): 是否为前向抓取模式。默认为 False。

    Returns:
        tuple[list[float], list[float]]: 右臂和左臂最终的逆运动学解。

    Example:
        >>> right_ik, left_ik = wipe_horizontal(
        ...     botId, 0.15, data, steps=12, primitive_name="wipe_h")
    """
    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, RIGHT_GRIPPER_LINK)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, LEFT_GRIPPER_LINK)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)
    iter_y = wipe_width / steps  # Y 轴方向每步增量

    # 阶段一：沿 Y 轴正方向移动（正向擦拭）
    for _ in range(steps):
        rightGripperPosition[1] += iter_y  # 右夹爪 Y 递增
        leftGripperPosition[1] += iter_y   # 左夹爪 Y 递增

        gripperPosition1 = p.calculateInverseKinematics(botId, RIGHT_GRIPPER_LINK, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)  # 保持姿态不变，仅改变位置
        gripperPosition2 = p.calculateInverseKinematics(botId, LEFT_GRIPPER_LINK, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[0:10], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition1[0:10], forces=[200]*10)  # 施加 200N 力控制右臂
        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[10:], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition2[10:], forces=[200]*9)   # 施加 200N 力控制左臂

        for _ in range(10):
            p.stepSimulation()

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

    # 阶段二：沿 Y 轴负方向返回（回程擦拭）
    for _ in range(steps):
        rightGripperPosition[1] -= iter_y  # 右夹爪 Y 递减
        leftGripperPosition[1] -= iter_y   # 左夹爪 Y 递减

        gripperPosition1 = p.calculateInverseKinematics(botId, RIGHT_GRIPPER_LINK, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, LEFT_GRIPPER_LINK, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[0:10], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition1[0:10], forces=[200]*10)
        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[10:], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition2[10:], forces=[200]*9)

        for _ in range(10):
            p.stepSimulation()

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

    return gripperPosition1, gripperPosition2


def wipe_vertical(botId, wipe_height, data, steps=12, tableId1=None, tableId2=None, both=None, primitive_name=None, is_front_grasp=False):
    """执行垂直擦拭动作，双臂沿 Z 轴方向往复运动。

    夹爪先沿 Z 轴正方向（向上）移动指定高度，再沿 Z 轴负方向（向下）返回原位，
    形成一个完整的垂直往复擦拭轨迹。运动过程中保持夹爪姿态不变。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        wipe_height (float): 垂直擦拭高度（单位：米），即 Z 轴方向的单程移动距离。
        data (dict[str, list]): 数据采集字典。
        steps (int, optional): 单程擦拭的插值步数。默认为 12。
        tableId1 (int | None, optional): 第一张桌面的模型标识符。默认为 None。
        tableId2 (int | None, optional): 第二张桌面的模型标识符。默认为 None。
        both (bool | None, optional): 是否同时记录两张桌面数据。默认为 None。
        primitive_name (str | None, optional): 运动原语名称标识。默认为 None。
        is_front_grasp (bool, optional): 是否为前向抓取模式。默认为 False。

    Returns:
        tuple[list[float], list[float]]: 右臂和左臂最终的逆运动学解。

    Example:
        >>> right_ik, left_ik = wipe_vertical(
        ...     botId, 0.10, data, steps=12, primitive_name="wipe_v")
    """
    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, RIGHT_GRIPPER_LINK)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, LEFT_GRIPPER_LINK)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)
    iter_z = wipe_height / steps  # Z 轴方向每步增量

    # 阶段一：沿 Z 轴正方向移动（向上擦拭）
    for _ in range(steps):
        rightGripperPosition[2] += iter_z  # 右夹爪 Z 递增（向上）
        leftGripperPosition[2] += iter_z   # 左夹爪 Z 递增（向上）

        gripperPosition1 = p.calculateInverseKinematics(botId, RIGHT_GRIPPER_LINK, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, LEFT_GRIPPER_LINK, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[0:10], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition1[0:10], forces=[200]*10)
        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[10:], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition2[10:], forces=[200]*9)

        for _ in range(10):
            p.stepSimulation()

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

    # 阶段二：沿 Z 轴负方向返回（向下擦拭）
    for _ in range(steps):
        rightGripperPosition[2] -= iter_z  # 右夹爪 Z 递减（向下）
        leftGripperPosition[2] -= iter_z   # 左夹爪 Z 递减（向下）

        gripperPosition1 = p.calculateInverseKinematics(botId, RIGHT_GRIPPER_LINK, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, LEFT_GRIPPER_LINK, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[0:10], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition1[0:10], forces=[200]*10)
        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[10:], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition2[10:], forces=[200]*9)

        for _ in range(10):
            p.stepSimulation()

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

    return gripperPosition1, gripperPosition2


def wipe_circular(botId, radius, data, steps=24, tableId1=None, tableId2=None, both=None, primitive_name=None, is_front_grasp=False):
    """执行圆弧擦拭动作，双臂在 Y-Z 平面内做圆周运动。

    夹爪以当前位置为圆心，在 Y-Z 平面内沿圆弧轨迹运动一圈。
    适用于需要旋转擦拭的场景（如擦拭圆形表面）。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        radius (float): 圆弧半径（单位：米）。
        data (dict[str, list]): 数据采集字典。
        steps (int, optional): 圆弧轨迹的离散步数，步数越多轨迹越平滑。默认为 24。
        tableId1 (int | None, optional): 第一张桌面的模型标识符。默认为 None。
        tableId2 (int | None, optional): 第二张桌面的模型标识符。默认为 None。
        both (bool | None, optional): 是否同时记录两张桌面数据。默认为 None。
        primitive_name (str | None, optional): 运动原语名称标识。默认为 None。
        is_front_grasp (bool, optional): 是否为前向抓取模式。默认为 False。

    Returns:
        tuple[list[float], list[float]]: 右臂和左臂最终的逆运动学解。

    Example:
        >>> right_ik, left_ik = wipe_circular(
        ...     botId, 0.05, data, steps=24, primitive_name="wipe_c")
    """
    import math

    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, RIGHT_GRIPPER_LINK)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, LEFT_GRIPPER_LINK)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)

    center_right_y = rightGripperPosition[1]  # 右夹爪圆弧圆心 Y 坐标
    center_right_z = rightGripperPosition[2]  # 右夹爪圆弧圆心 Z 坐标
    center_left_y = leftGripperPosition[1]    # 左夹爪圆弧圆心 Y 坐标
    center_left_z = leftGripperPosition[2]    # 左夹爪圆弧圆心 Z 坐标

    for i in range(steps):
        angle = 2 * math.pi * i / steps  # 当前步对应的角度（弧度），从 0 到 2π

        rightGripperPosition[1] = center_right_y + radius * math.cos(angle)  # Y 分量：圆心 + r·cos(θ)
        rightGripperPosition[2] = center_right_z + radius * math.sin(angle)  # Z 分量：圆心 + r·sin(θ)
        leftGripperPosition[1] = center_left_y + radius * math.cos(angle)
        leftGripperPosition[2] = center_left_z + radius * math.sin(angle)

        gripperPosition1 = p.calculateInverseKinematics(botId, RIGHT_GRIPPER_LINK, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, LEFT_GRIPPER_LINK, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[0:10], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition1[0:10], forces=[200]*10)
        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[10:], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition2[10:], forces=[200]*9)

        for _ in range(10):
            p.stepSimulation()

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

    return gripperPosition1, gripperPosition2


def press_down(botId, press_depth, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, is_front_grasp=False, min_z=None):
    """执行向下按压动作，双臂沿 Z 轴负方向逐步下降。

    夹爪沿 Z 轴负方向移动指定深度，模拟按压操作。
    可选地通过 min_z 参数限制夹爪最低高度，防止穿透桌面。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        press_depth (float): 按压深度（单位：米），即 Z 轴负方向的移动距离。
        data (dict[str, list]): 数据采集字典。
        steps (int, optional): 插值步数。默认为 10。
        tableId1 (int | None, optional): 第一张桌面的模型标识符。默认为 None。
        tableId2 (int | None, optional): 第二张桌面的模型标识符。默认为 None。
        both (bool | None, optional): 是否同时记录两张桌面数据。默认为 None。
        primitive_name (str | None, optional): 运动原语名称标识。默认为 None。
        is_front_grasp (bool, optional): 是否为前向抓取模式。默认为 False。
        min_z (float | None, optional): 夹爪允许的最低 Z 轴高度（单位：米）。
            若为 None 则不进行高度钳位。默认为 None。

    Returns:
        tuple[list[float], list[float]]: 右臂和左臂最终的逆运动学解。

    Example:
        >>> right_ik, left_ik = press_down(
        ...     botId, 0.05, data, steps=10, min_z=0.80,
        ...     primitive_name="press")
    """
    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, RIGHT_GRIPPER_LINK)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, LEFT_GRIPPER_LINK)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)
    iter_z = press_depth / steps  # Z 轴每步下降量

    for _ in range(steps):
        rightGripperPosition[2] -= iter_z  # 右夹爪 Z 递减（向下按压）
        leftGripperPosition[2] -= iter_z   # 左夹爪 Z 递减（向下按压）

        if min_z is not None:  # 若指定了最低高度，进行钳位防止穿透
            rightGripperPosition[2] = clamp_gripper_z(rightGripperPosition[2], min_z)
            leftGripperPosition[2] = clamp_gripper_z(leftGripperPosition[2], min_z)

        gripperPosition1 = p.calculateInverseKinematics(botId, RIGHT_GRIPPER_LINK, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, LEFT_GRIPPER_LINK, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[0:10], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition1[0:10], forces=[300]*10)  # 按压时施加 300N 力
        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[10:], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition2[10:], forces=[300]*9)

        for _ in range(10):
            p.stepSimulation()

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

    return gripperPosition1, gripperPosition2


def lift_off(botId, lift_height, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, is_front_grasp=False):
    """执行抬升离面动作，双臂沿 Z 轴正方向逐步上升。

    夹爪沿 Z 轴正方向移动指定高度，模拟从表面抬起物体的操作。
    使用较小的关节力（100N）以实现柔和的抬升效果。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        lift_height (float): 抬升高度（单位：米），即 Z 轴正方向的移动距离。
        data (dict[str, list]): 数据采集字典。
        steps (int, optional): 插值步数。默认为 10。
        tableId1 (int | None, optional): 第一张桌面的模型标识符。默认为 None。
        tableId2 (int | None, optional): 第二张桌面的模型标识符。默认为 None。
        both (bool | None, optional): 是否同时记录两张桌面数据。默认为 None。
        primitive_name (str | None, optional): 运动原语名称标识。默认为 None。
        is_front_grasp (bool, optional): 是否为前向抓取模式。默认为 False。

    Returns:
        tuple[list[float], list[float]]: 右臂和左臂最终的逆运动学解。

    Example:
        >>> right_ik, left_ik = lift_off(
        ...     botId, 0.15, data, steps=10, primitive_name="lift_off")
    """
    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, RIGHT_GRIPPER_LINK)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, LEFT_GRIPPER_LINK)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)
    iter_z = lift_height / steps  # Z 轴每步上升量

    for _ in range(steps):
        rightGripperPosition[2] += iter_z  # 右夹爪 Z 递增（向上抬升）
        leftGripperPosition[2] += iter_z   # 左夹爪 Z 递增（向上抬升）

        gripperPosition1 = p.calculateInverseKinematics(botId, RIGHT_GRIPPER_LINK, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, LEFT_GRIPPER_LINK, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[0:10], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition1[0:10], forces=[100]*10)  # 柔和抬升，施加 100N 力
        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[10:], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition2[10:], forces=[100]*9)

        for _ in range(10):
            p.stepSimulation()

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

    return gripperPosition1, gripperPosition2


def shift_position(botId, dx, dy, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, is_front_grasp=False):
    """执行水平位移动作，双臂在 X-Y 平面内平移指定偏移量。

    夹爪在 X-Y 平面内同时移动相同的偏移量，适用于需要水平调整夹爪位置的场景。
    Z 轴高度和夹爪姿态保持不变。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        dx (float): X 轴方向的偏移量（单位：米），正值表示沿 X 正方向移动。
        dy (float): Y 轴方向的偏移量（单位：米），正值表示沿 Y 正方向移动。
        data (dict[str, list]): 数据采集字典。
        steps (int, optional): 插值步数。默认为 10。
        tableId1 (int | None, optional): 第一张桌面的模型标识符。默认为 None。
        tableId2 (int | None, optional): 第二张桌面的模型标识符。默认为 None。
        both (bool | None, optional): 是否同时记录两张桌面数据。默认为 None。
        primitive_name (str | None, optional): 运动原语名称标识。默认为 None。
        is_front_grasp (bool, optional): 是否为前向抓取模式。默认为 False。

    Returns:
        tuple[list[float], list[float]]: 右臂和左臂最终的逆运动学解。

    Example:
        >>> right_ik, left_ik = shift_position(
        ...     botId, 0.05, -0.03, data, steps=10,
        ...     primitive_name="shift")
    """
    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, RIGHT_GRIPPER_LINK)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, LEFT_GRIPPER_LINK)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)
    iter_x = dx / steps  # X 轴每步增量
    iter_y = dy / steps  # Y 轴每步增量

    for _ in range(steps):
        rightGripperPosition[0] += iter_x  # 右夹爪 X 递增
        rightGripperPosition[1] += iter_y  # 右夹爪 Y 递增
        leftGripperPosition[0] += iter_x   # 左夹爪 X 递增
        leftGripperPosition[1] += iter_y   # 左夹爪 Y 递增

        gripperPosition1 = p.calculateInverseKinematics(botId, RIGHT_GRIPPER_LINK, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, LEFT_GRIPPER_LINK, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[0:10], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition1[0:10], forces=[200]*10)
        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[10:], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition2[10:], forces=[200]*9)

        for _ in range(10):
            p.stepSimulation()

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

    return gripperPosition1, gripperPosition2


def connect(botId, rightCoords, leftCoords, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, is_front_grasp=False):
    """执行连接动作，双臂沿 Y 轴方向逐步移动到目标位置。

    左右夹爪分别沿 Y 轴方向线性插值移动到各自的目标 Y 坐标，
    X 和 Z 坐标保持不变。适用于双臂需要沿侧向合拢或展开的场景。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        rightCoords (list[float]): 右夹爪目标位置 [x, y, z]（仅 y 分量生效）。
        leftCoords (list[float]): 左夹爪目标位置 [x, y, z]（仅 y 分量生效）。
        data (dict[str, list]): 数据采集字典。
        steps (int, optional): 插值步数。默认为 10。
        tableId1 (int | None, optional): 第一张桌面的模型标识符。默认为 None。
        tableId2 (int | None, optional): 第二张桌面的模型标识符。默认为 None。
        both (bool | None, optional): 是否同时记录两张桌面数据。默认为 None。
        primitive_name (str | None, optional): 运动原语名称标识。默认为 None。
        is_front_grasp (bool, optional): 是否为前向抓取模式。默认为 False。

    Returns:
        tuple[list[float], list[float]]: 右臂和左臂最终的逆运动学解。

    Example:
        >>> right_ik, left_ik = connect(
        ...     botId, [0.6, -0.1, 0.9], [0.6, 0.1, 0.9],
        ...     data, steps=10, primitive_name="connect")
    """
    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, RIGHT_GRIPPER_LINK)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, LEFT_GRIPPER_LINK)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)

    iter_right_y = (rightCoords[1] - rightGripperPosition[1]) / steps  # 右夹爪 Y 方向每步增量
    iter_left_y = (leftCoords[1] - leftGripperPosition[1]) / steps     # 左夹爪 Y 方向每步增量

    for _ in range(steps):
        rightGripperPosition[1] += iter_right_y  # 右夹爪 Y 递增
        leftGripperPosition[1] += iter_left_y    # 左夹爪 Y 递增

        gripperPosition1 = p.calculateInverseKinematics(botId, RIGHT_GRIPPER_LINK, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, LEFT_GRIPPER_LINK, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)
        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[0:10], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition1[0:10])
        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[10:], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition2[10:])

        for _ in range(10):
            p.stepSimulation()

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

    return gripperPosition1, gripperPosition2


def lift(botId, z, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, is_front_grasp=False):
    """执行提升动作，双臂沿 Z 轴正方向上升并同时闭合夹爪。

    与 lift_off 不同，此原语在抬升过程中会同时控制夹爪关节闭合
    （目标角度 -0.75 弧度，施加 10000N 夹持力），模拟抓取物体后提升的操作。
    每步结束后还会微调手指关节（目标角度 -0.45 弧度）。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        z (float): 提升高度（单位：米），即 Z 轴正方向的移动距离。
        data (dict[str, list]): 数据采集字典。
        steps (int, optional): 插值步数。默认为 10。
        tableId1 (int | None, optional): 第一张桌面的模型标识符。默认为 None。
        tableId2 (int | None, optional): 第二张桌面的模型标识符。默认为 None。
        both (bool | None, optional): 是否同时记录两张桌面数据。默认为 None。
        primitive_name (str | None, optional): 运动原语名称标识。默认为 None。
        is_front_grasp (bool, optional): 是否为前向抓取模式。默认为 False。

    Returns:
        tuple[list[float], list[float]]: 右臂和左臂最终的逆运动学解。

    Example:
        >>> right_ik, left_ik = lift(
        ...     botId, 0.20, data, steps=15, primitive_name="lift")
    """
    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, RIGHT_GRIPPER_LINK)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, LEFT_GRIPPER_LINK)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)
    iter_z = z / steps  # Z 轴每步上升量

    for _ in range(steps):
        rightGripperPosition[2] += iter_z  # 右夹爪 Z 递增（向上提升）
        leftGripperPosition[2] += iter_z   # 左夹爪 Z 递增（向上提升）

        gripperPosition1 = p.calculateInverseKinematics(botId, RIGHT_GRIPPER_LINK, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, LEFT_GRIPPER_LINK, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[0:10], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition1[0:10], forces=[1000]*10)  # 提升时施加 1000N 力
        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[10:], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition2[10:], forces=[1000]*9)
        p.setJointMotorControlArray(botId, jointIndices=[29, 31, 52, 54], controlMode=p.POSITION_CONTROL, targetPositions=[-0.75]*4, forces=[10000]*4)  # 控制夹爪关节闭合，关节 29/31 为右夹爪，52/54 为左夹爪

        for _ in range(10):
            p.stepSimulation()

        p.setJointMotorControlArray(botId, jointIndices=[19, 42], controlMode=p.POSITION_CONTROL, targetPositions=[-0.45]*2, forces=[100]*2)  # 微调手指关节，关节 19 为右手指，42 为左手指

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

    return gripperPosition1, gripperPosition2


def move_sideways(botId, rightCoords, leftCoords, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, is_front_grasp=False):
    """执行侧向移动动作，双臂沿 Y 轴方向移动到目标位置并闭合夹爪。

    左右夹爪分别沿 Y 轴方向线性插值移动到目标 Y 坐标，
    同时控制夹爪关节闭合以保持抓取状态。X 和 Z 坐标使用目标值。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        rightCoords (list[float]): 右夹爪目标位置 [x, y, z]。
        leftCoords (list[float]): 左夹爪目标位置 [x, y, z]。
        data (dict[str, list]): 数据采集字典。
        steps (int, optional): 插值步数。默认为 10。
        tableId1 (int | None, optional): 第一张桌面的模型标识符。默认为 None。
        tableId2 (int | None, optional): 第二张桌面的模型标识符。默认为 None。
        both (bool | None, optional): 是否同时记录两张桌面数据。默认为 None。
        primitive_name (str | None, optional): 运动原语名称标识。默认为 None。
        is_front_grasp (bool, optional): 是否为前向抓取模式。默认为 False。

    Returns:
        tuple[list[float], list[float]]: 右臂和左臂最终的逆运动学解。

    Example:
        >>> right_ik, left_ik = move_sideways(
        ...     botId, [0.6, -0.1, 0.9], [0.6, 0.1, 0.9],
        ...     data, steps=10, primitive_name="move_side")
    """
    revoluteJoints = getRevoluteJoints(botId)

    y_prev_right = p.getLinkState(botId, RIGHT_GRIPPER_LINK)[0][1]  # 右夹爪当前 Y 坐标
    iter_right = (rightCoords[1] - y_prev_right) / steps  # 右夹爪 Y 方向每步增量

    y_prev_left = p.getLinkState(botId, LEFT_GRIPPER_LINK)[0][1]    # 左夹爪当前 Y 坐标
    iter_left = (leftCoords[1] - y_prev_left) / steps  # 左夹爪 Y 方向每步增量

    for _ in range(steps):
        y_prev_right += iter_right  # 右夹爪 Y 递增
        y_prev_left += iter_left    # 左夹爪 Y 递增

        gripperPosition1 = p.calculateInverseKinematics(botId, RIGHT_GRIPPER_LINK, [rightCoords[0], y_prev_right, rightCoords[2]], maxNumIterations=1000)  # X/Z 使用目标值，Y 逐步插值
        gripperPosition2 = p.calculateInverseKinematics(botId, LEFT_GRIPPER_LINK, [leftCoords[0], y_prev_left, leftCoords[2]], maxNumIterations=1000)

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[0:10], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition1[0:10])
        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[10:], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition2[10:])
        p.setJointMotorControlArray(botId, jointIndices=[29, 31, 52, 54], controlMode=p.POSITION_CONTROL, targetPositions=[-0.75]*4, forces=[10000]*4)  # 闭合夹爪以保持抓取

        for _ in range(10):
            p.stepSimulation()

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

    return gripperPosition1, gripperPosition2


def retract(botId, iter1, iter2, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, is_front_grasp=False):
    """执行回撤动作，双臂沿 X 轴负方向和 Y 轴正方向同时移动。

    夹爪每步沿 X 轴负方向移动 iter1、沿 Y 轴正方向移动 iter2，
    模拟从操作区域向后退出的动作。使用较小的关节力（50N）实现柔和回撤。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        iter1 (float): 每步沿 X 轴负方向的移动量（单位：米）。
        iter2 (float): 每步沿 Y 轴正方向的移动量（单位：米）。
        data (dict[str, list]): 数据采集字典。
        steps (int, optional): 插值步数。默认为 10。
        tableId1 (int | None, optional): 第一张桌面的模型标识符。默认为 None。
        tableId2 (int | None, optional): 第二张桌面的模型标识符。默认为 None。
        both (bool | None, optional): 是否同时记录两张桌面数据。默认为 None。
        primitive_name (str | None, optional): 运动原语名称标识。默认为 None。
        is_front_grasp (bool, optional): 是否为前向抓取模式。默认为 False。

    Returns:
        tuple[list[float], list[float]]: 右臂和左臂最终的逆运动学解。

    Example:
        >>> right_ik, left_ik = retract(
        ...     botId, 0.01, 0.005, data, steps=10,
        ...     primitive_name="retract")
    """
    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, RIGHT_GRIPPER_LINK)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, LEFT_GRIPPER_LINK)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)

    for _ in range(steps):
        rightGripperPosition[0] -= iter1  # 右夹爪 X 递减（后退）
        leftGripperPosition[0] -= iter1   # 左夹爪 X 递减（后退）
        rightGripperPosition[1] += iter2  # 右夹爪 Y 递增（侧移）
        leftGripperPosition[1] += iter2   # 左夹爪 Y 递增（侧移）

        gripperPosition1 = p.calculateInverseKinematics(botId, RIGHT_GRIPPER_LINK, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, LEFT_GRIPPER_LINK, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[0:10], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition1[0:10], forces=[50]*10)  # 柔和回撤，施加 50N 力
        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[10:], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition2[10:], forces=[50]*9)

        for _ in range(10):
            p.stepSimulation()

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

    return gripperPosition1, gripperPosition2


def lower(botId, iter1, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, is_front_grasp=False, min_z=None):
    """执行下降动作，双臂沿 Z 轴负方向逐步下降。

    夹爪每步沿 Z 轴负方向移动 iter1，模拟将物体放置到较低位置的操作。
    可选地通过 min_z 参数限制夹爪最低高度，防止穿透桌面。

    Args:
        botId (int): PyBullet 中加载的机器人模型唯一标识符。
        iter1 (float): 每步沿 Z 轴负方向的移动量（单位：米）。
        data (dict[str, list]): 数据采集字典。
        steps (int, optional): 插值步数。默认为 10。
        tableId1 (int | None, optional): 第一张桌面的模型标识符。默认为 None。
        tableId2 (int | None, optional): 第二张桌面的模型标识符。默认为 None。
        both (bool | None, optional): 是否同时记录两张桌面数据。默认为 None。
        primitive_name (str | None, optional): 运动原语名称标识。默认为 None。
        is_front_grasp (bool, optional): 是否为前向抓取模式。默认为 False。
        min_z (float | None, optional): 夹爪允许的最低 Z 轴高度（单位：米）。
            若为 None 则不进行高度钳位。默认为 None。

    Returns:
        tuple[list[float], list[float]]: 右臂和左臂最终的逆运动学解。

    Example:
        >>> right_ik, left_ik = lower(
        ...     botId, 0.01, data, steps=10, min_z=0.80,
        ...     primitive_name="lower")
    """
    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, RIGHT_GRIPPER_LINK)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, LEFT_GRIPPER_LINK)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)

    for _ in range(steps):
        rightGripperPosition[2] -= iter1  # 右夹爪 Z 递减（向下）
        leftGripperPosition[2] -= iter1   # 左夹爪 Z 递减（向下）

        if min_z is not None:  # 若指定了最低高度，进行钳位防止穿透
            rightGripperPosition[2] = clamp_gripper_z(rightGripperPosition[2], min_z)
            leftGripperPosition[2] = clamp_gripper_z(leftGripperPosition[2], min_z)

        gripperPosition1 = p.calculateInverseKinematics(botId, RIGHT_GRIPPER_LINK, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, LEFT_GRIPPER_LINK, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[0:10], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition1[0:10], forces=[100]*10)
        p.setJointMotorControlArray(botId, jointIndices=revoluteJoints[10:], controlMode=p.POSITION_CONTROL, targetPositions=gripperPosition2[10:], forces=[100]*9)

        for _ in range(10):
            p.stepSimulation()

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, is_front_grasp=is_front_grasp, both=both)

    return gripperPosition1, gripperPosition2
