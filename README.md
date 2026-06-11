# IL-dual-arm: 基于分层解耦表示的双臂清洁操作模仿学习

<p align="center">
  <strong>Imitation Learning for Dual-Arm Surface Cleaning via Hierarchical Disentangled Representation</strong>
</p>

---

## 项目简介

HDR-IL 是一个面向**双臂协同清洁操作**的模仿学习框架。项目使用 Baxter 双臂机器人，在 PyBullet 物理仿真环境中执行**擦桌子**和**擦玻璃**两项清洁任务，通过分层解耦的方式将复杂操作分解为多个运动基元，利用图注意力网络（GAT）和变分自编码器（VAE）学习并生成多样化的清洁轨迹。

### 核心特点

- **分层解耦表示**：将清洁任务分解为高层规划（选择基元）和底层执行（生成轨迹）两个层级
- **图注意力交互建模**：基于 DGL 构建全连接图，捕捉双臂与清洁表面之间的空间交互关系
- **目标位姿条件增强**：在 VAE 编码器中融合目标物体位姿，增强轨迹生成的目标导向性
- **可微 DTW 对齐**：使用 smooth-min 实现可微 DTW，支持端到端时序对齐优化
- **末端执行器优化**：夹爪碰撞体替换为扁平抹布垫形状，海绵通过固定约束附着，z 轴安全限位防止穿入表面

---

## 项目结构

```
IL-dual-arm/
├── Simulation Code/                        # 仿真数据采集模块
│   ├── primitives.py                       # 运动基元函数库（含 z 轴安全限位、碰撞检测）
│   ├── clean_table.py                      # 擦桌子任务仿真（含海绵约束）
│   ├── clean_glass.py                      # 擦玻璃任务仿真
│   ├── baxter.xml                          # Baxter 机器人 URDF（碰撞体已优化）
│   ├── cleaning_table.xml                  # 清洁任务桌面 URDF
│   ├── glass_panel.xml                     # 竖直玻璃面板 URDF
│   ├── sponge.xml                          # 海绵/抹布 URDF
│   ├── plane.xml                           # 自定义地平面 URDF
│   ├── l_finger.stl / l_finger_tip.stl     # 夹爪网格文件
│   └── {base,head,torso,...}/              # Baxter 机器人各部件网格
│
├── Table_Lift_HDR-IL/                      # 模型训练与推理模块
│   ├── Models/
│   │   └── HDRIL_Models.py                 # 模型定义（GAT, VAE, Decoder, Planning）
│   ├── TrainDynamicModels.py               # 训练基元动态模型
│   ├── TrainPlanningModel.py               # 训练规划模型
│   ├── GenerateProjections.py              # 生成轨迹投影
│   └── utils.py                            # 数据集定义、DTW 损失、可视化
│
└── README.md                               # 本文档
```

---

## 任务定义

### 任务一：擦桌子（Clean Table）

Baxter 双臂夹持海绵/抹布，在**水平桌面**上执行清洁操作：

| 步骤 | 基元名称 | 功能描述 |
|------|---------|---------|
| 1 | Approach | 双臂从初始位置直线运动接近桌面（保持安全高度） |
| 2 | Press_Down | 双臂竖直下压，使海绵接触桌面表面 |
| 3 | Wipe_Horizontal | 双臂沿 y 轴水平擦拭（0.4m） |
| 4 | Shift_Position | 双臂沿 y 轴负方向平移（0.2m），为下一行擦拭定位 |
| 5 | Wipe_Horizontal_2 | 第二次水平擦拭（0.4m） |
| 6 | Shift_Position_2 | 第二次 y 轴负方向平移（0.2m） |
| 7 | Wipe_Circular | 双臂做圆周擦拭运动（半径 0.05m），清除局部污渍 |
| 8 | Lift_Off | 双臂竖直抬起，脱离桌面 |
| 9 | Retract | 双臂后退回收至初始区域 |

**关键实现**：
- 海绵通过 `JOINT_FIXED` 约束附着到夹爪末端
- z 轴安全限位机制：`min_z = table_top_z + 0.01`，防止垫体穿入桌面
- 双臂间距 0.9m（y = ±0.45），避免双臂重叠/碰撞

### 任务二：擦玻璃（Clean Glass）

Baxter 双臂在**竖直玻璃面板**上执行清洁操作：如下：


https://github.com/user-attachments/assets/94b4e737-82cc-4103-956f-8eb269328c14



| 步骤 | 基元名称 | 功能描述 |
|------|---------|---------|
| 1 | Approach | 双臂接近玻璃面板正面 |
| 2 | Press_Down | 双臂向前按压，使夹爪接触玻璃表面 |
| 3 | Wipe_Vertical | 双臂沿 z 轴竖直擦拭（0.3m） |
| 4 | Shift_Position | 双臂沿 y 轴负方向平移（0.15m） |
| 5 | Wipe_Vertical_2 | 第二次竖直擦拭（0.3m） |
| 6 | Shift_Position_2 | 第二次 y 轴负方向平移（0.15m） |
| 7 | Wipe_Circular | 双臂做圆周擦拭运动（半径 0.06m） |
| 8 | Lift_Off | 双臂抬起脱离玻璃表面 |
| 9 | Retract | 双臂后退回收 |

**关键实现**：
- 玻璃面板竖直放置（frame 0.8×0.05×1.0m + glass 0.7×0.01×0.9m）
- 擦拭方向为竖直（z 轴），与擦桌子的水平（y 轴）方向不同
- 玻璃面板中心高度 z = 0.5m

---

## 算法原理

### 整体架构

框架采用**分层结构**，将清洁任务分解为高层规划与底层轨迹生成两个层级：

```
┌─────────────────────────────────────────────────┐
│              高层：规划模型 (Planning Model)       │
│   PlanningGAT + PlanningDecoder → 基元序列预测    │
│   输入：观测序列 → 输出：基元类型（分类问题）       │
└──────────────────────┬──────────────────────────┘
                       │ 基元标签
┌──────────────────────▼──────────────────────────┐
│           底层：动态模型 (Dynamic Models)          │
│   每个基元一个 GAT-VAE → 轨迹生成                  │
│   输入：起始状态 + 目标位姿 → 输出：轨迹序列         │
└─────────────────────────────────────────────────┘
```

### 图注意力网络编码器（GAT Encoder）

基于 DGL 库构建**全连接交互图**，将双臂机器人的状态特征建模为图节点：

$$G = (V, E), \quad |V| = d_{input}, \quad E = V \times V$$

其中 $d_{input} = 21$ 为输入特征维度。

**图注意力层计算流程**：

1. **线性变换**：$z_i = W \cdot h_i$

2. **注意力系数**：$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T [z_i \| z_j]\right)$

3. **Softmax 归一化**：$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$

4. **加权聚合**：$h_i' = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \cdot z_j$

**GAT 编码器完整前向过程**：

$$\mu, \log\sigma^2 = f_{\theta}\left(\text{GRU}(\text{GAT}(x_{1:T})), \text{MLP}(p_{target})\right)$$

### VAE 轨迹生成模型

每个操作基元对应一个独立的 VAE 模型：

**编码过程**：

$$q_\phi(z | x_{1:T}) = \mathcal{N}(\mu_\phi(x_{1:T}), \sigma^2_\phi(x_{1:T}))$$

**重参数化技巧**：

$$z = \mu + \epsilon \cdot \sigma, \quad \epsilon \sim \mathcal{N}(0, I)$$

**解码过程**（GRU 自回归）：

$$\hat{x}_t = f_\psi(z, \hat{x}_{t-1}), \quad \hat{x}_0 = \mathbf{0}$$

**损失函数**：

$$\mathcal{L}_{recon} = \frac{1}{T} \sum_{t=1}^{T} \|x_t - \hat{x}_t\|^2$$

### 可微 DTW 损失

用 smooth-min 替代 hard-min 实现可微性：

$$\text{smooth-min}_\rho(\mathbf{x}) = -\frac{1}{\rho} \log\left(\frac{1}{n}\sum_{i=1}^{n} e^{-\rho x_i} + \epsilon\right)$$

$$D(i,j) = d(x_i, y_j) + \text{smooth-min}_\rho\{D(i\!-\!1,j), D(i,j\!-\!1), D(i\!-\!1,j\!-\!1)\}$$

其中 $\rho$ 控制平滑程度，$\rho \to \infty$ 时退化为标准 min。

### 规划模型

- **编码器**：PlanningGAT（不融合目标位姿）
- **解码器**：PlanningDecoder（GRU + Softmax）
- **损失函数**：交叉熵 $\mathcal{L}_{CE} = -\sum_{c} y_c \log \hat{y}_c$

### 输入特征

模型输入为 21 维特征向量：

| 特征组 | 维度 | 描述 |
|--------|------|------|
| 右夹爪位置 | 3 | $(x, y, z)$ |
| 右夹爪姿态 | 4 | 四元数 $(q_1, q_2, q_3, q_4)$ |
| 左夹爪位置 | 3 | $(x, y, z)$ |
| 左夹爪姿态 | 4 | 四元数 $(q_1, q_2, q_3, q_4)$ |
| 目标物体位置 | 3 | $(x, y, z)$ |
| 目标物体姿态 | 4 | 四元数 $(q_1, q_2, q_3, q_4)$ |

---

## 仿真环境

### 平台配置

| 项目 | 配置 |
|------|------|
| 物理引擎 | PyBullet |
| 机器人 | Baxter 双臂机器人（14 个旋转关节 + 夹爪） |
| 控制方式 | 逆运动学（IK）+ 关节位置控制 |
| 仿真步长 | 0.01s |
| 重力加速度 | -9.81 m/s² |
| 迭代次数 | 5000 次/任务 |

### URDF 模型

| 文件 | 描述 |
|------|------|
| `baxter.xml` | Baxter 双臂机器人（碰撞体已优化为扁平抹布垫形状） |
| `cleaning_table.xml` | 水平清洁桌面（固定基座，桌面高度 0.39m） |
| `glass_panel.xml` | 竖直玻璃面板（frame + glass + 4 个标记点，固定基座） |
| `sponge.xml` | 海绵/抹布（0.08×0.12×0.04m，非固定基座） |
| `plane.xml` | 自定义地平面（lateral_friction=0.25） |

### 末端执行器碰撞体优化

为解决夹爪穿入清洁表面的问题，对 `baxter.xml` 中的夹爪碰撞体进行了优化：

| 链接名称 | 原碰撞体 | 优化后碰撞体 | 优化原因 |
|----------|---------|------------|---------|
| `gripper_pole` / `_2` | 细圆柱 (r=0.01, L=0.1) | 扁平盒子 (0.10×0.08×0.02) | 模拟抹布/海绵垫的扁平形状 |
| `gripper_body` / `_2` | 高窄盒子 (0.02×0.02×0.08) | 扁平垫 (0.06×0.04×0.015) | 避免穿入表面 |
| `tip` / `_2` | 盒子 (0.02×0.02×0.04) | 极小盒子 (0.01×0.01×0.01) | 最小化指尖碰撞体积 |

> 仅修改碰撞体，视觉外观保持不变。

### 海绵约束与 z 轴安全限位

擦桌子任务中的安全机制：

```
z 高度计算链：
  table_top_z = 0.39          （桌面上表面）
  pad_bottom_offset = 0.08    （夹爪极点到垫体底部的距离）
  sponge_height = 0.04        （海绵厚度）
  approach_z = 0.61           （接近高度 = 0.39 + 0.08 + 0.04 + 0.10）
  wipe_z = 0.51               （擦拭高度 = 0.39 + 0.08 + 0.04）
  min_z = 0.40                （夹爪最低允许高度，确保垫体不低于桌面）
```

### 域随机化

每次迭代添加随机偏移，增强数据多样性和模型鲁棒性：

| 任务 | 随机参数 | 范围 |
|------|---------|------|
| 擦桌子 | 桌面 x/y 位移 | uniform(0, 0.1) |
| 擦玻璃 | 面板 x/y 位移 | uniform(0, 0.1) / uniform(0, -0.1) |

---

## 运行步骤

### 环境配置

```bash
# 创建 conda 环境
conda create -n hdril python=3.11
conda activate hdril

# 安装依赖
pip install pybullet pybullet_data
pip install torch dgl pandas numpy matplotlib
```

### 数据采集

```bash
cd "Simulation Code"

# 擦桌子任务（生成 clean_table_primitive_data.csv）
python clean_table.py

# 擦玻璃任务（生成 clean_glass_primitive_data.csv）
python clean_glass.py
```

> 仿真会运行 5000 次迭代，每次迭代执行完整的清洁操作序列并记录数据。

### 模型训练

```bash
cd "../Table_Lift_HDR-IL"

# 训练动态模型（每个基元一个 GAT-VAE）
python TrainDynamicModels.py

# 训练规划模型（基元序列预测）
python TrainPlanningModel.py
```

### 轨迹生成与评估

```bash
# 生成轨迹投影
python GenerateProjections.py -trainiters 1 -startindex 0 -datasize 55 -features 21
```

---

## 关键超参数

| 参数 | 值 | 描述 |
|------|-----|------|
| `n_latent` | 512 | VAE 潜在空间维度 |
| `lr_encoder` | 5e-5 | 编码器学习率 |
| `lr_decoder` | 5e-5 | 解码器学习率 |
| `n_epochs` | 1-2 | 训练轮数 |
| `runs` | 10-2500 | 演示数据量 |
| `runsize` | 70 | 每条演示的序列长度 |
| `sequencelength` | 10-12 | 基元序列长度 |
| `rho` (DTW) | 10-40 | smooth-min 温度参数 |
| `timeStep` | 0.01 | 仿真时间步长 |

---

## 数据格式

采集的 CSV 文件包含以下字段：

| 字段 | 描述 |
|------|------|
| `Primitive` | 基元名称（如 Approach, Press_Down, Wipe_Horizontal 等） |
| `right/left_gripper_pole_x/y/z_{1,2}` | 夹爪执行前(1)和后(2)的位置坐标 |
| `right/left_gripper_pole_q_{1-4}_{1,2}` | 夹爪执行前(1)和后(2)的四元数姿态 |
| `table1_x/y/z_{1,2}` | 目标物体执行前(1)和后(2)的位置坐标 |
| `table1_quat{1-4}_{1,2}` | 目标物体执行前(1)和后(2)的四元数姿态 |
| `x/y_displacement{1,2}` | 域随机化偏移量 |
| `grippers_open` | 夹爪状态（1=张开, 0=闭合） |
| `label` | 任务标签（正确示范=1） |

---

## 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│  阶段一：数据采集（PyBullet）                                  │
│  clean_table.py / clean_glass.py → 生成专家示范 CSV          │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段二：训练（纯 PyTorch，无物理仿真）                        │
│  TrainDynamicModels.py → 基元动态 VAE 模型                   │
│  TrainPlanningModel.py → 规划模型                             │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段三：推理生成轨迹（纯 PyTorch）                            │
│  GenerateProjections.py → 输出预测轨迹 CSV                   │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段四：可视化验证（PyBullet）                                │
│  读取预测 CSV → IK 驱动机器人 → 验证物理合理性 → 判定成功      │
└─────────────────────────────────────────────────────────────┘
```

---

## 依赖

| 库 | 版本要求 | 用途 |
|----|---------|------|
| Python | >= 3.11 | 运行环境 |
| PyBullet | >= 3.2.5 | 物理仿真与可视化 |
| PyTorch | >= 1.12 | 模型训练与推理 |
| DGL | >= 0.9 | 图注意力网络构建 |
| Pandas | >= 1.5 | 数据处理 |
| NumPy | >= 1.23 | 数值计算 |
| Matplotlib | >= 3.6 | 结果可视化 |

---

## 致谢

开源项目：https://github.com/Rose-STL-Lab/HDR-IL
