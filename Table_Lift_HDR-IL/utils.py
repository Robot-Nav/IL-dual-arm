"""工具函数与数据集模块

本模块提供HDR-IL框架的基础工具函数，包括数据集加载、特征选择、
动作原语标签映射、结果可视化以及可微DTW损失函数的实现。

核心组件:
    selectColumnLabels: 从原始数据中选择21维特征列和动作原语标签
    outputHeaders: 返回特征列名列表，用于CSV输出
    label_primitive: 将字符串动作原语名称映射为整数索引
    BaxterDataset: Baxter机器人数据集类，继承PyTorch Dataset
    plotresults: 绘制预测轨迹与真实轨迹的对比图（含不确定性区间）
    analyzeErrors: 分析两组投影之间的欧氏误差
    dtw: 标准动态时间规整算法（不可微）
    diff_dtw_loss: 可微DTW损失，使用smooth-min替代argmin实现可微性
    DTW_Loss: PyTorch损失函数模块，封装可微DTW用于模型训练
"""

from __future__ import unicode_literals, print_function, division
from io import open

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import isinf
from numpy import array, zeros, full, argmin, inf, ndim
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F



def selectColumnLabels(data):
    """从原始数据中选择21维特征列和动作原语标签

    提取Baxter机器人的双臂夹爪位姿（各7维：3平移+4四元数）
    和桌面位姿（7维：3平移+4四元数），共21维特征。

    Args:
        data: pandas DataFrame，包含原始CSV数据的所有列

    Returns:
        tuple: (features, labels)
            - features: 21维特征DataFrame，包含双臂夹爪和桌面位姿
            - labels: 动作原语标签Series，值为原语名称字符串
    """
    features = data[{

        'right_gripper_pole_x_1',
         'right_gripper_pole_y_1',
         'right_gripper_pole_z_1',

         'right_gripper_pole_q_11',
         'right_gripper_pole_q_12',
         'right_gripper_pole_q_13',
         'right_gripper_pole_q_14',

         'left_gripper_pole_x_1',
         'left_gripper_pole_y_1',
         'left_gripper_pole_z_1',
         'left_gripper_pole_q_11',
         'left_gripper_pole_q_12',
         'left_gripper_pole_q_13',
         'left_gripper_pole_q_14',


         'x_1',
         'y_1',
         'z_1',
         'quat1_1',
         'quat2_1',
         'quat3_1',
         'quat4_1',
    }]

    labels = data[{

        'Prim'

    }]

    return features, labels



def outputHeaders():
    """返回特征列名列表

    生成与selectColumnLabels所选特征对应的列名列表，
    用于CSV文件输出时的表头。

    Returns:
        list: 21个特征列名的列表，顺序为右夹爪(7) + 左夹爪(7) + 桌面(7)
    """
    headers = [

        'right_gripper_pole_x_1',
         'right_gripper_pole_y_1',
         'right_gripper_pole_z_1',

         'right_gripper_pole_q_11',
         'right_gripper_pole_q_12',
         'right_gripper_pole_q_13',
         'right_gripper_pole_q_14',

         'left_gripper_pole_x_1',
         'left_gripper_pole_y_1',
         'left_gripper_pole_z_1',
         'left_gripper_pole_q_11',
         'left_gripper_pole_q_12',
         'left_gripper_pole_q_13',
         'left_gripper_pole_q_14',

         'x_1',
         'y_1',
         'z_1',
         'quat1_1',
         'quat2_1',
         'quat3_1',
         'quat4_1',


    ]

    return headers




def label_primitive (row):
    """将字符串动作原语名称映射为整数索引

    动作原语编码:
        0: Front_Grasp（前方抓取）
        1: MoveSideways（侧移）
        2: Lift（抬举）
        3: extend（伸展）
        4: Place（放置）
        5: Retract（收回）

    Args:
        row: pandas Series，包含'Primitive'列的数据行

    Returns:
        int: 动作原语的整数索引(0-5)，未知原语返回'Other'
    """

   if row['Primitive'] == "Front_Grasp" :
      return 0
   if row['Primitive'] == "MoveSideways" :
      return 1
   if row['Primitive'] == "Lift" :
      return 2
   if row['Primitive'] == "extend" :
      return 3
   if row['Primitive'] == "Place" :
      return 4
   if row['Primitive'] == "Retract" :
      return 5

   return 'Other'



class BaxterDataset(Dataset):
    """Baxter机器人数据集

    从CSV文件加载Baxter机器人的演示数据，提取21维特征和动作原语标签。
    特征包括双臂夹爪位姿（各7维）和桌面位姿（7维）。

    Args:
        无构造参数，数据路径硬编码为'../Simulation Data/primitive data filtering 625 L.csv'

    Attributes:
        coordinates: 原始CSV数据DataFrame
        columns: 21维特征张量，形状为 (N, 21)
        labels: 动作原语标签张量，形状为 (N,)
    """

    def __init__(self):

        #filePathTrain = '../Simulation Data/Replay_VLOOKUP.csv'
        filePathTrain = '../Simulation Data/primitive data filtering 625 L.csv'

        self.coordinates = pd.read_csv(filePathTrain)
        # 将字符串原语名称转换为整数索引
        self.coordinates['Prim'] = self.coordinates.apply(lambda row: label_primitive(row), axis = 1)
        self.columns, self.labels= selectColumnLabels(self.coordinates)

        # 转换为PyTorch张量
        self.columns = torch.tensor(self.columns.values)
        self.labels = torch.tensor(self.labels.values)


    def __getitem__(self, index):
        """获取指定索引的特征和标签

        Args:
            index: 数据索引

        Returns:
            tuple: (columns[index], labels[index])
        """
        return self.columns[index], self.labels[index]

    def __len__(self):
        """返回数据集大小

        Returns:
            int: 样本总数
        """
        return len(self.columns)



def plotresults(variances, a, a1):
    """绘制预测轨迹与真实轨迹的对比图（含不确定性区间）

    生成3张图（X/Y/Z坐标），每张图包含：
    - 左右夹爪的预测轨迹（填充区域表示±2σ不确定性区间）
    - 左右夹爪的真实演示轨迹（虚线）
    - 动作原语分界线（垂直线）

    Args:
        variances: 预测方差张量，形状为 (seq_len, 1, features)
        a: 预测轨迹张量，形状为 (seq_len, 1, features)
        a1: 真实演示轨迹张量，形状为 (seq_len, 1, features)
    """

    trainsize = len(a)
    time = np.arange(trainsize)[0:trainsize]

    # 方差取平方根并乘以2，得到95%置信区间宽度
    variances = np.array(variances.tolist())
    variances = np.sqrt(variances)*2


    # 右夹爪X坐标及不确定性区间
    rx = np.array(a[:, 0, 0].tolist())
    x1 = np.array(a1[:, 0, 0].tolist())
    rxvar = np.array(variances[:, 0, 0])
    rxupper =np.add(rx, rxvar)
    rxlower = np.subtract(rx, rxvar)

    # 右夹爪Y坐标及不确定性区间
    ry = np.array(a[:, 0, 1].tolist())
    y1 = np.array(a1[:, 0, 1].tolist())
    ryvar = np.array(variances[:, 0, 1].tolist())
    ryupper =np.add(ry, ryvar)
    rylower = np.subtract(ry, ryvar)

    # 右夹爪Z坐标及不确定性区间
    rz = np.array(a[:, 0, 2].tolist())
    z1 = np.array(a1[:, 0, 2].tolist())
    rzvar = np.array(variances[:, 0, 2].tolist())
    rzupper =np.add(rz, rzvar)
    rzlower = np.subtract(rz, rzvar)

    # 左夹爪X坐标及不确定性区间
    lx = np.array(a[:, 0, 7].tolist())
    x2 = np.array(a1[:, 0, 7].tolist())
    lxvar = np.array(variances[:, 0, 7].tolist())
    lxupper =np.add(lx, lxvar)
    lxlower = np.subtract(lx, lxvar)

    # 左夹爪Y坐标及不确定性区间
    ly = np.array(a[:, 0, 8].tolist())
    y2 = np.array(a1[:, 0, 8].tolist())
    lyvar = np.array(variances[:, 0, 8].tolist())
    lyupper =np.add(ly, lyvar)
    lylower = np.subtract(ly, lyvar)

    # 左夹爪Z坐标及不确定性区间
    lz = np.array(a[:, 0, 9].tolist())
    z2 = np.array(a1[:, 0, 9].tolist())
    lzvar = np.array(variances[:, 0, 9].tolist())
    lzupper =np.add(lz, lzvar)
    lzlower = np.subtract(lz, lzvar)



    #plt.plot(time, rx)
    f1 = plt.figure(figsize=(6, 4))
    # 绘制预测的不确定性区间（±2σ）
    plt.fill_between(time, rxupper, rxlower, alpha=0.8, label = "Right Predictions")
    plt.fill_between(time, lxupper, lxlower, alpha=0.8, label = "Left Predictions")

    # 绘制真实演示轨迹
    plt.plot(time, x1, 'k--', alpha = .6, label = "Right Demo")
    plt.plot(time, x2, 'r--', alpha = .6, label = "Left Demo")

    # 绘制动作原语分界线
    plt.axvline(x=10, color='k')
    plt.axvline(x=22, color='k')
    plt.axvline(x=34, color='k')
    plt.axvline(x=46, color='k')
    plt.axvline(x=58, color='k')

    plt.title("Multi-Model X Coordinates", fontsize=18)
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Meters", fontsize=14)

    #axs[0].plot(time, rx)
    ac = plt.gca()
    # 设置X轴刻度标签为动作原语名称
    ac.set_xticks([5, 17, 29, 41, 53, 65])
    ac.set_xticklabels(["grasp", "move", "lift", "extend", "place", "retract"])

    plt.legend(loc = 'upper left')
    plt.tight_layout()
    plt.savefig('x gnnrnn.png')


    f2 = plt.figure(figsize=(6, 4))
    plt.fill_between(time, ryupper, rylower, alpha=0.8)
    plt.fill_between(time, lyupper, lylower, alpha=0.8)

    plt.plot(time, y1, 'k--', alpha = .6)
    plt.plot(time, y2, 'r--', alpha = .6)

    plt.axvline(x=10, color='k')
    plt.axvline(x=22, color='k')
    plt.axvline(x=34, color='k')
    plt.axvline(x=46, color='k')
    plt.axvline(x=58, color='k')

    plt.title("Multi-Model Y Coordinates", fontsize=18)
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Meters", fontsize=14)

    # axs[0].plot(time, rx)

    ac = plt.gca()
    ac.set_xticks([5, 17, 29, 41, 53, 65])
    ac.set_xticklabels(["grasp", "move", "lift", "extend", "place", "retract"])

    plt.tight_layout()
    plt.savefig('y gnnrnn.png')


    f3 = plt.figure(figsize=(6, 4))
    plt.fill_between(time, rzupper, rzlower, alpha=0.8)
    plt.fill_between(time, lzupper, lzlower, alpha=0.8)

    plt.plot(time, z1, 'k--', alpha = .6)
    plt.plot(time, z2, 'r--', alpha = .6)

    plt.axvline(x=10, color='k')
    plt.axvline(x=22, color='k')
    plt.axvline(x=34, color='k')
    plt.axvline(x=46, color='k')
    plt.axvline(x=58, color='k')

    plt.title("Multi-Model Z Coordinates", fontsize=18)
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Meters", fontsize=14)

    # axs[0].plot(time, rx)

    ac = plt.gca()
    ac.set_xticks([5, 17, 29, 41, 53, 65])
    ac.set_xticklabels(["grasp", "move", "lift", "extend", "place", "retract"])
    plt.tight_layout()
    plt.savefig('z gnnrnn.png')



    plt.show()



def analyzeErrors():
    """分析两组投影之间的欧氏误差

    读取两组CSV数据，计算左右夹爪位置的欧氏距离误差，
    绘制50次演示的平均误差曲线及95%置信区间。

    注意：此函数中的文件路径为硬编码的本地路径，需根据实际环境修改。
    """
    df1 = pd.read_csv(r'C:\Users\jxtxw\Desktop\rl_course\official3\official\projection_data\3_3_V19.csv')

    df2 = pd.read_csv(r'C:\Users\jxtxw\Desktop\rl_course\il\official\simulator_data\lift_place_testing_xy_yaw_size.csv')

    def selectRows(data):
        """选择6维位置特征（左右夹爪各3维XYZ）"""
        features = data[[

            'left_gripper_x',
            'left_gripper_y',
            'left_gripper_z',
            'right_gripper_x',
            'right_gripper_y',
            'right_gripper_z',
        ]]

        return features

    feature1 = selectRows(df1)
    feature2 = selectRows(df2)

    f1 = feature1.to_numpy()
    f2 = feature2.to_numpy()
    f2 = f2[:len(feature1), :].copy()
    print(f1.shape, f2.shape)

    # 按61步一段分割，计算50次演示的误差
    e1_50 = np.zeros((50, 61))
    e2_50 = np.zeros((50, 61))
    for i in range(50):
        start_index = i * 61
        end_index = (i + 1) * 61
        # 右夹爪欧氏误差
        right_gripper_error = f1[start_index:end_index, 4:6] - f2[start_index:end_index, 4:6]
        e1 = np.sqrt(np.power(right_gripper_error, 2).sum(axis=1))
        # 左夹爪欧氏误差
        left_gripper_error = f1[start_index:end_index, 0:3] - f2[start_index:end_index, 0:3]
        e2 = np.sqrt(np.power(left_gripper_error, 2).sum(axis=1))

        e1_50[i] = e1
        e2_50[i] = e2

    # 计算平均误差和标准误差
    average1 = e1_50.mean(axis=0)
    average2 = e2_50.mean(axis=0)

    std_1 = e1_50.std(axis=0) / np.sqrt(e1_50.shape[0])
    std_2 = e2_50.std(axis=0) / np.sqrt(e2_50.shape[0])

    fig, ax = plt.subplots(1, 1)
    ax.set_title('Euclidean error for the gripper position')

    color1 = np.random.rand(3)
    color2 = np.random.rand(3)

    line1, = ax.plot(average1, color='r')
    line2, = ax.plot(average2, color='b')
    # 绘制95%置信区间（±1.98σ）
    ax.fill_between(np.arange(0, 61, 1), average1 - 1.98 * std_1, average1 + 1.98 * std_1, alpha=0.4, facecolor='g')
    ax.fill_between(np.arange(0, 61, 1), average2 - 1.98 * std_2, average2 + 1.98 * std_2, alpha=0.4, facecolor='g')
    ax.legend([line1, line2], ['right gripper', 'left gripper'])
    plt.savefig('3_3')
    plt.show()



def _traceback(D):
    """DTW最优路径回溯

    从累积代价矩阵的右下角回溯到左上角，找到最优对齐路径。

    Args:
        D: 累积代价矩阵，形状为 (r+2, c+2)

    Returns:
        tuple: (p, q) 两个索引数组，表示最优对齐路径
    """
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        # 选择代价最小的前驱方向：对角线、上方、左方
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """标准动态时间规整（DTW）算法

    计算两个序列之间的DTW距离。DTW通过动态规划找到两个序列之间的
    最优时间对齐路径，允许时间轴上的弹性伸缩。

    算法流程:
        1. 构建代价矩阵C[i,j] = dist(x[i], y[j])
        2. 构建累积代价矩阵D[i,j] = C[i,j] + min(D[i-1,j-1], D[i-1,j], D[i,j-1])
        3. 回溯最优路径

    Args:
        x: 第一个序列，形状为 (N1, M)
        y: 第二个序列，形状为 (N2, M)
        dist: 距离函数，接受两个向量返回标量距离
        warp: 允许的平移步数（默认1）
        w: 窗口大小，限制匹配索引之间的最大距离 |i-j|（默认无限制）
        s: 非对角线移动的权重，s越大路径越偏向对角线（默认1.0）

    Returns:
        tuple: (distance, cost_matrix, acc_cost_matrix, path)
            - distance: DTW距离（累积代价矩阵右下角值）
            - cost_matrix: 代价矩阵C
            - acc_cost_matrix: 累积代价矩阵D
            - path: 最优对齐路径 (p, q)
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        # 有窗口约束：初始化为inf，窗口内设为0
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        # 无窗口约束：标准初始化
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    # 计算局部代价矩阵
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    # 动态规划填充累积代价矩阵
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i+k, r)
                j_k = min(j+k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def smooth_min(x, rho=10):
    """平滑最小值函数（LogSumExp变体）

    使用LogSumExp技巧实现可微的近似最小值操作。
    smooth_min(x) ≈ min(x)，且关于x可微。

    公式: smooth_min(x) = -1/ρ * log(mean(exp(-ρ * x)))

    Args:
        x: 输入张量
        rho: 温度参数，ρ越大越接近真实的min操作

    Returns:
        Tensor: 平滑最小值
    """
    eps = 1e-12
    # 数值稳定：将指数参数裁剪到[-50, 50]避免溢出
    x_clamped = torch.clamp(x * (-rho), min=-50, max=50)
    val = -1/rho * T.log(T.mean(T.exp(x_clamped)) + eps)
    return val


def _traceback(D):
    """DTW最优路径回溯（可微版本使用）

    Args:
        D: 累积代价矩阵

    Returns:
        tuple: (p, q) 最优对齐路径索引数组
    """
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)


def diff_dtw_loss(x, y, dist, warp=1, w=inf, s=1.0, rho=40):
    """可微动态时间规整（Differentiable DTW）损失

    与标准DTW算法结构相同，但使用smooth_min替代不可微的min操作，
    使得DTW距离可以通过反向传播计算梯度，从而作为神经网络的损失函数。

    关键改进:
        标准DTW中 D1[i,j] += min(D0[i,j], D0[i_k,j]*s, D0[i,j_k]*s)
        可微DTW中 D1[i,j] += smooth_min([D0[i,j], D0[i_k,j]*s, D0[i,j_k]*s], rho)

    Args:
        x: 第一个序列张量，形状为 (seq_len, dim)
        y: 第二个序列张量，形状为 (seq_len, dim)
        dist: 可微距离函数，接受两个向量返回标量距离
        warp: 允许的平移步数（默认1）
        w: 窗口大小（默认无限制）
        s: 非对角线移动权重（默认1.0）
        rho: smooth_min的温度参数（默认40），越大越接近真实min

    Returns:
        tuple: (distance, cost_matrix, acc_cost_matrix, path)
            - distance: 可微DTW距离
            - cost_matrix: 代价矩阵
            - acc_cost_matrix: 累积代价矩阵
            - path: 最优对齐路径
    """

    assert x.shape[0]
    assert y.shape[0]
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0

    MAX_VAL = 1e2

    r, c = x.shape[0], y.shape[0]
    if not isinf(w):
        D0 = T.full((r + 1, c + 1), MAX_VAL)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = T.zeros(r + 1, c + 1)
        D0[0, 1:] = MAX_VAL
        D0[1:, 0] = MAX_VAL
    D1 = D0[1:, 1:]  # view
    # 计算局部代价矩阵（使用可微距离函数）
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[0][i], y[0][j])
    C = D1.clone()
    # 动态规划填充累积代价矩阵（使用smooth_min替代min）
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = D0[i, j]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                # 使用torch.stack替代Python列表，以支持自动微分
                min_list = T.stack([min_list, D0[i_k, j] * s, D0[i, j_k] * s])
            # 关键：用smooth_min替代不可微的min操作
            min_val = smooth_min(min_list, rho)
            D1[i, j] = D1[i, j] + min_val
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


# 欧氏范数距离函数（逐元素绝对差，用于DTW计算）
euclidean_norm = lambda x, y: T.abs(x - y)


from torch.nn.modules import Module

class _Loss(Module):
    """损失函数基类

    继承PyTorch的Module类，提供损失函数的通用接口，
    支持reduction参数控制输出的归约方式。

    Args:
        size_average: 已弃用参数，是否对损失取平均
        reduce: 已弃用参数，是否进行归约
        reduction: 归约方式，'mean'或'sum'
        _Reduction: 归约枚举类
    """
    def __init__(self, size_average=None, reduce=None, reduction='mean', _Reduction=None):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class DTW_Loss(_Loss):
    """可微DTW损失函数模块

    将可微动态时间规整（Differentiable DTW）封装为PyTorch损失函数模块，
    可直接用于模型训练。支持3D批量输入（batch × seq_len × dim）。

    核心思想:
        使用smooth_min替代标准DTW中的argmin操作，使整个DTW计算图可微，
        从而允许梯度通过DTW距离反向传播到模型参数。

    Args:
        rho: smooth_min的温度参数（默认10），越大越接近真实min
        size_average: 已弃用参数
        reduce: 已弃用参数
        reduction: 归约方式（默认'mean'）
    """

    def __init__(self, rho=10, size_average=None, reduce=None, reduction='mean'):
        super(DTW_Loss, self).__init__(size_average, reduce, reduction)
        self.rho = rho

    def forward(self, output, target):
        """计算可微DTW损失

        Args:
            output: 模型输出序列，形状为 (batch, seq_len, dim) 或 (seq_len, dim)
            target: 目标序列，形状与output相同

        Returns:
            Tensor: DTW损失值（标量）
        """
        # batch x seq_len x dim
        if ndim(output)==3:
            # 批量模式：对每个batch计算DTW距离后取平均
            dist = []
            for b in range(output.size(0)):
                d_b, cost_matrix, diff_acc_cost_matrix, diff_path = diff_dtw_loss(output[b], target[b], dist=euclidean_norm, rho=self.rho)
                dist.append(d_b)
            d = T.mean(T.stack(dist))
        else:
            # 单序列模式
            d, cost_matrix, diff_acc_cost_matrix, diff_path = diff_dtw_loss(output, target, dist=euclidean_norm, rho=self.rho)
        loss_val =  d
        return loss_val
