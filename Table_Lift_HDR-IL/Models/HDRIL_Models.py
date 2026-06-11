"""HDR-IL 模型定义模块

本模块实现了 HDR-IL（Hierarchical Dynamic Reasoning via Imitation Learning）框架中的核心神经网络模型，
用于机器人桌面抬举任务的运动轨迹学习与生成。

核心类:
    GATLayer: 图注意力网络单头层，实现节点间的注意力权重计算
    MultiHeadGATLayer: 多头图注意力层，聚合多个注意力头的输出
    GAT: 基于GAT+GRU的VAE编码器（含桌面特征融合），输出潜在分布参数
    GAT2: 基于GAT+GRU的VAE编码器（不含桌面特征融合），输出潜在分布参数
    Decoder: VAE解码器，基于GRU自回归地生成轨迹序列
    VAE: 变分自编码器，封装编码器-解码器架构，支持训练、推理与不确定性估计
    PlanningGAT: 规划模块的GAT编码器，输出节点特征与全局隐状态
    PlanningDecoder: 规划模块的解码器，输出动作原语的Softmax概率分布
    PlanningVAE: 规划模块的VAE，用于动作原语序列的预测与决策
"""

import math

from torch.nn import functional as F

import torch
import torch.nn as nn
import utils
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class GATLayer(nn.Module):
    """图注意力网络单头层

    实现标准的图注意力机制（Graph Attention Network），通过学习节点对之间的
    注意力权重来聚合邻居节点信息。基于DGL框架的消息传递接口实现。

    架构设计:
        1. 线性变换层(fc): 将输入特征映射到输出维度
        2. 注意力计算层(attn_fc): 对拼接的源-目标节点特征计算注意力分数
        3. 消息传递: 通过DGL的update_all接口实现高效的消息聚合

    Args:
        g: DGL图对象，定义了节点与边的拓扑结构
        in_dim: 输入特征维度
        out_dim: 输出特征维度
    """

    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()
        self.alphalist = []

    def reset_parameters(self):
        """使用Xavier初始化重置模型参数

        根据ReLU激活函数计算增益值，对线性变换层和注意力层进行Xavier正态分布初始化，
        以保证前向传播时信号的方差稳定。
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        """计算边级注意力分数

        将源节点和目标节点的变换后特征拼接，通过注意力层计算原始注意力分数，
        再经过LeakyReLU激活函数引入非线性。

        Args:
            edges: DGL边对象，包含源节点和目标节点特征

        Returns:
            dict: 包含注意力分数 'e' 的字典
        """
        # 拼接源节点和目标节点的变换特征，形成 [z_src || z_dst]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        # 通过注意力层计算原始分数
        a = self.attn_fc(z2)

        # LeakyReLU激活，负斜率默认为0.01
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        """消息函数：收集源节点特征和边注意力分数

        Args:
            edges: DGL边对象

        Returns:
            dict: 包含源节点变换特征 'z' 和边注意力分数 'e' 的字典
        """
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        """归约函数：通过注意力加权聚合邻居节点信息

        对接收到的消息执行Softmax归一化得到注意力权重，再对邻居节点特征
        进行加权求和，实现注意力聚合。

        Args:
            nodes: DGL节点对象，包含邮箱中的消息

        Returns:
            dict: 包含聚合后节点特征 'h' 的字典
        """
        # 对所有邻居的注意力分数执行Softmax归一化
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # 注意力加权求和聚合邻居特征
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def getalpha(self, nodes):
        """获取注意力权重分布

        Args:
            nodes: DGL节点对象

        Returns:
            Tensor: Softmax归一化后的注意力权重
        """
        return F.softmax(nodes.mailbox['e'], dim=1)

    def forward(self, h):
        """GAT层前向传播

        执行完整的图注意力计算流程：线性变换 → 边注意力计算 → 消息传递 → 注意力聚合。

        Args:
            h: 输入节点特征矩阵，形状为 (N, in_dim)

        Returns:
            Tensor: 输出节点特征矩阵，形状为 (N, out_dim)
        """
        # 线性变换：将节点特征映射到输出空间
        z = self.fc(h)
        # 将变换后的特征存储到图节点数据中
        self.g.ndata['z'] = z
        # 计算所有边的注意力分数
        self.g.apply_edges(self.edge_attention)
        # 执行消息传递和注意力聚合
        self.g.update_all(self.message_func, self.reduce_func)

        return self.g.ndata.pop('h')



class MultiHeadGATLayer(nn.Module):
    """多头图注意力层

    将多个独立的GATLayer并行运行，并将各头的输出进行聚合。
    多头机制允许模型同时关注不同子空间中的特征表示，增强表达能力。

    Args:
        g: DGL图对象
        in_dim: 输入特征维度
        out_dim: 每个注意力头的输出维度
        num_heads: 注意力头数量
        merge: 多头输出聚合方式，'cat'为拼接，'avg'为平均
    """

    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        """多头注意力前向传播

        Args:
            h: 输入节点特征矩阵，形状为 (N, in_dim)

        Returns:
            Tensor: 聚合后的输出特征，'cat'模式下形状为 (N, out_dim*num_heads)，
                    'avg'模式下形状为 (N, out_dim)
        """
        # 各注意力头独立计算
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # 拼接模式：将各头输出在特征维度拼接
            return torch.cat(head_outs, dim=1)
        else:
            # 平均模式：对各头输出取均值
            return torch.mean(torch.stack(head_outs))

    def getalpha(self):
        """获取所有注意力头的注意力权重"""
        for h in self.heads:
            h.getalpha()




class GAT(nn.Module):
    """基于GAT+GRU的VAE编码器（含桌面特征融合）

    该编码器是HDR-IL动力学模型的核心组件，采用图注意力网络处理机器人关节图结构，
    结合GRU捕捉时序依赖，最终输出VAE潜在空间的均值和对数方差。
    特别地，该变体将桌面位姿特征与GRU隐状态融合后再映射到潜在分布参数。

    架构设计:
        1. 两层多头GAT: 提取图结构的空间特征
        2. GRU循环网络: 建模时序动态演化
        3. 桌面特征融合: 将桌面位姿(7维: 3平移+4旋转)映射到隐空间并拼接
        4. VAE参数输出: 全连接层输出潜在分布的均值和对数方差

    Args:
        g: DGL图对象
        in_dim: 输入特征维度（每个节点的特征数）
        hidden_dim: 隐层维度，同时也是VAE潜在空间维度的一半
        latent_dim: GAT输出/GRU隐状态的维度
        num_heads: 第一层GAT的注意力头数
    """

    def __init__(self, g, in_dim, hidden_dim, latent_dim, num_heads):
        super(GAT, self).__init__()
        # 第一层：多头GAT，输出维度为 latent_dim * num_heads
        self.layer1 = MultiHeadGATLayer(g, in_dim, latent_dim, num_heads)
        # 第二层：单头GAT，将多头输出融合为 latent_dim 维
        self.layer2 = MultiHeadGATLayer(g, latent_dim * num_heads, latent_dim, 1)

        self.input_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.nodes = len(g.nodes())

        # 桌面位姿特征映射：7维(3平移+4四元数) → hidden_dim
        self.table2hid = nn.Linear(7, hidden_dim)

        # GRU输入维度 = 节点数 × 每节点特征维度
        self.gru_size = self.nodes * latent_dim

        # 将GAT输出映射回输入空间，用于自回归输入
        self.hid2in = nn.Linear(latent_dim, in_dim)

        # GRU循环网络：建模时序动态
        self.rnn = nn.GRU(self.gru_size, self.gru_size)

        # VAE参数映射层
        self.hid1 = nn.Linear(self.gru_size, hidden_dim)
        # 融合GRU隐状态与桌面特征：2*hidden_dim → hidden_dim
        self.hid2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.hid3 = nn.Linear(hidden_dim, hidden_dim)
        # 输出VAE潜在分布参数：hidden_dim → 2*hidden_dim (均值+对数方差)
        self.hid2hid = nn.Linear(hidden_dim, 2 * hidden_dim)



    def forward(self, h, target):
        """编码器前向传播

        逐步处理输入序列，通过GAT提取空间特征、GRU建模时序动态，
        最终融合桌面特征输出VAE潜在分布参数。

        Args:
            h: 输入序列，形状为 (seq_len, num_nodes, in_dim)
            target: 目标时间步张量，用于确定循环步数

        Returns:
            tuple: (zmean, zlogvar)
                - zmean: VAE潜在空间均值，形状为 (1, hidden_dim)
                - zlogvar: VAE潜在空间对数方差，形状为 (1, hidden_dim)
        """
        # 初始化输入为序列第一个时间步
        temp = h[0]
        # 提取桌面位姿特征（7维：3平移+4四元数），来自第7-13行节点
        t = h[0, 7:14, :].reshape(1, 7)
        table = self.table2hid(t).unsqueeze(0)

        # 初始化GRU隐状态为零
        hidden = torch.zeros((1, 1, self.gru_size)).to(device)

        for i in range(0, target.size()[0]):

            # 自回归：在输入序列范围内使用真实数据，超出后使用上一步的预测
            if i < h.size()[0]:
                temp = h[i]

            # 第一层多头GAT：提取空间特征
            h1 = self.layer1(temp)

            # 展平所有节点特征作为GRU输入
            rnn_inp = h1.flatten()

            # GRU前向：更新时序隐状态
            out, hid = self.rnn(rnn_inp.unsqueeze(0).unsqueeze(0), hidden)

            # 重塑GRU输出和隐状态为图节点形式
            out = out.reshape(self.nodes, self.latent_dim)
            hid = hid.reshape(self.nodes, self.latent_dim)
            # 第二层GAT：融合多头特征并更新隐状态
            hidden = self.layer2(hid)
            hidden = hidden.flatten().unsqueeze(0).unsqueeze(0)

            # 将GRU输出映射回输入空间，作为下一步的自回归输入
            temp = self.hid2in(out).squeeze(0)


        # 将GRU最终隐状态映射到VAE参数空间
        h0 = self.hid1(hidden.float())

        # 拼接GRU隐状态与桌面特征，实现条件生成
        h0 = self.hid2(torch.cat((h0, table), 2))
        h0 = self.hid3(h0)
        # 映射到2*hidden_dim，前半为均值，后半为对数方差
        h0 = self.hid2hid(h0).squeeze(0)

        # 分离VAE潜在分布参数：均值和对数方差
        zmean = h0[:, :self.hidden_dim]
        zlogvar = h0[:, self.hidden_dim:]

        return zmean, zlogvar


class GAT2(nn.Module):
    """基于GAT+GRU的VAE编码器（不含桌面特征融合）

    与GAT编码器结构类似，但去除了桌面位姿特征的融合路径。
    适用于不需要桌面条件信息的动力学建模场景。

    架构设计:
        1. 两层多头GAT + GRU: 与GAT相同的空间-时序特征提取
        2. 直接映射: GRU隐状态直接映射到VAE参数，不融合桌面特征

    Args:
        g: DGL图对象
        in_dim: 输入特征维度
        hidden_dim: 隐层维度
        latent_dim: GAT输出/GRU隐状态维度
        num_heads: 第一层GAT的注意力头数
    """

    def __init__(self, g, in_dim, hidden_dim, latent_dim, num_heads):
        super(GAT2, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, latent_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(g, latent_dim * num_heads, latent_dim, 1)

        self.input_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.nodes = len(g.nodes())


        self.table2hid = nn.Linear(7, hidden_dim)

        self.gru_size = self.nodes * latent_dim
        self.hid2in = nn.Linear(latent_dim, in_dim)
        self.rnn = nn.GRU(self.gru_size, self.gru_size)
        self.hid1 = nn.Linear(self.gru_size, hidden_dim)
        self.hid2 = nn.Linear(hidden_dim, hidden_dim)
        self.hid3 = nn.Linear(hidden_dim, hidden_dim)

        # 输出VAE潜在分布参数：hidden_dim → 2*hidden_dim
        self.hid2hid = nn.Linear(hidden_dim, 2 * hidden_dim)

    def forward(self, h, target):
        """编码器前向传播（无桌面特征融合）

        Args:
            h: 输入序列，形状为 (seq_len, num_nodes, in_dim)
            target: 目标时间步张量

        Returns:
            tuple: (zmean, zlogvar)
                - zmean: VAE潜在空间均值，形状为 (1, hidden_dim)
                - zlogvar: VAE潜在空间对数方差，形状为 (1, hidden_dim)
        """
        temp = h[0]

        hidden = torch.zeros((1, 1, self.gru_size)).to(device)

        for i in range(0, target.size()[0]):

            h1 = self.layer1(temp)

            rnn_inp = h1.flatten()

            out, hid = self.rnn(rnn_inp.unsqueeze(0).unsqueeze(0), hidden)

            out = out.reshape(self.nodes, self.latent_dim)
            hid = hid.reshape(self.nodes, self.latent_dim)
            hidden = self.layer2(hid)
            hidden = hidden.flatten().unsqueeze(0).unsqueeze(0)
            temp = self.hid2in(out).squeeze(0)

        # 直接从GRU隐状态映射到VAE参数，不融合桌面特征
        h0 = self.hid1(hidden.float())

        h0 = self.hid2(h0)
        h0 = self.hid3(h0)

        h0 = self.hid2hid(h0).squeeze(0)
        zmean = h0[:, :self.hidden_dim]
        zlogvar = h0[:, self.hidden_dim:]

        return zmean, zlogvar



class Decoder(nn.Module):
    """VAE解码器

    基于GRU的自回归解码器，从VAE潜在空间的采样点出发，
    逐步生成轨迹序列。每一步的GRU输出经过多层全连接映射到输出空间。

    架构设计:
        1. GRU: 以潜在向量为初始隐状态，自回归地生成序列
        2. 多层全连接: 将GRU输出逐步映射到目标维度
        3. 自回归机制: 上一步的GRU输出作为下一步的输入

    Args:
        input_dim: 输入维度（与编码器输出对齐）
        output_dim: 输出维度（与原始特征维度对齐）
        hidden_dim: 隐层维度
    """

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim

        self.output_dim = int(output_dim)
        self.hidden_dim = hidden_dim


        self.gru = nn.GRU(hidden_dim, hidden_dim)

        # 多层全连接映射：GRU隐层 → 输出空间
        self.l2h = nn.Linear(hidden_dim, hidden_dim)
        self.l2h2 = nn.Linear(hidden_dim, hidden_dim)
        self.l2h3 = nn.Linear(hidden_dim, hidden_dim)
        self.l2h4 = nn.Linear(hidden_dim, hidden_dim)
        self.l2o2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden, target):
        """解码器前向传播

        从VAE潜在向量出发，自回归地生成目标长度的轨迹序列。

        Args:
            hidden: VAE潜在向量（编码器输出），形状为 (1, 1, hidden_dim)
            target: 目标时间步张量，用于确定生成长度

        Returns:
            Tensor: 生成的轨迹序列，形状为 (seq_len, 1, output_dim)
        """
        # 将潜在向量作为GRU初始隐状态
        hiddeninp = hidden.unsqueeze(0)

        # 初始化GRU输入为零向量（自回归起始信号）
        input = torch.zeros(1, 1, hiddeninp.size()[2]).to(device)
        # 初始化输出张量
        output = torch.zeros(1, target.size()[1], target.size()[2]).to(device)

        for i in range(0, target.size()[0]):
            # GRU单步前向：输入当前token，更新隐状态
            zs, h = self.gru(input, hiddeninp)

            # 自回归：当前步输出作为下一步输入
            input = zs
            # 多层全连接映射到输出空间
            hs = self.l2h(zs)
            hs = self.l2h2(hs)
            hs = self.l2h3(hs)
            hs = self.l2h4(hs)
            hs = self.l2o2(hs)
            hiddeninp = h

            # 拼接当前步输出到结果序列
            output = torch.cat((output, hs))

        # 去除初始化的零向量行
        output = torch.cat((output[1:, :, :],))

        return output


class VAE(nn.Module):
    """变分自编码器（VAE）

    封装GAT编码器和GRU解码器，实现完整的VAE训练与推理流程。
    支持重参数化技巧采样、均值-方差估计以及KL散度正则化。

    架构设计:
        1. 编码器(GAT/GAT2): 将输入轨迹编码为潜在分布参数(均值+对数方差)
        2. 解码器(Decoder): 从潜在空间采样点生成轨迹
        3. 重参数化技巧: z = μ + ε * exp(0.5 * log σ²)，保证梯度可传播
        4. KL散度正则化: 约束潜在分布接近标准正态分布

    Args:
        input_size: 输入维度（节点数）
        target_size: 输出维度
        latent_size: 潜在空间维度
        encoder: 编码器网络（GAT或GAT2实例）
        decoder: 解码器网络（Decoder实例）
        encoder_optimizer: 编码器优化器
        decoder_optimizer: 解码器优化器
        loss: 损失函数
    """

    def __init__(self, input_size, target_size, latent_size, encoder, decoder, encoder_optimizer, decoder_optimizer,
                 loss):
        super(VAE, self).__init__()
        self.input_dim = input_size
        self.output_dim = target_size
        self.hidden_dim = latent_size


        self.encoder = encoder
        self.decoder = decoder
        self.nodes = input_size


        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.criterion = loss

    def forward(self, input_tensor, target_tensor):
        """VAE前向传播（确定性推理）

        编码输入序列后直接使用编码器隐状态作为解码器输入，
        不经过采样过程，用于确定性推理。

        Args:
            input_tensor: 输入张量
            target_tensor: 目标张量

        Returns:
            Tensor: 解码器输出序列
        """
        max_length = 500000

        rows = input_tensor.size()[0]

        dim0 = target_tensor.size()[0]
        dim1 = target_tensor.size()[1]
        # 构建时间步索引表：每行重复对应的时间步编号
        timetable = []
        for i in range(1, dim0 + 1):
            for j in range(1, dim1 + 1):
                timetable.append(i)

        time = torch.tensor(timetable).view(dim0, dim1).unsqueeze(-1)
        # 编码器：提取潜在分布参数
        encoder_output, encoder_hidden = self.encoder(input_tensor, time)

        # 直接使用编码器隐状态（不采样）作为解码器初始状态
        decoder_hidden = encoder_hidden
        decoder_output = self.decoder(decoder_hidden, time)

        return decoder_output

    def generate_with_seed(self, seed_x, size):
        """基于种子序列生成轨迹（含重参数化采样）

        使用重参数化技巧从编码器输出的潜在分布中采样，
        再通过解码器生成轨迹序列。

        Args:
            seed_x: 种子输入序列，形状为 (seq_len, batch, nodes*features)
            size: 目标时间步张量，用于确定生成长度

        Returns:
            tuple: (x_p, zmean, zlogvar)
                - x_p: 生成的轨迹序列
                - zmean: 潜在空间均值
                - zlogvar: 潜在空间对数方差
        """
        seed_t_len = seed_x.shape[0]

        # 重塑输入为 (seq_len, num_nodes, features_per_node) 格式
        input = seed_x.reshape(seed_x.size()[0], seed_x.size()[1], self.nodes,
                               int(seed_x.size(2) / self.nodes)).squeeze(1)


        # 编码器输出潜在分布参数
        zmean, zlogvar = self.encoder(input, size)

        # 重参数化技巧：z = μ + ε * exp(0.5 * log σ²)
        # 使得采样操作可微，梯度可通过μ和σ传播
        z = zmean + torch.randn_like(zmean) * torch.exp(0.5 * zlogvar)
        # 解码器从采样点生成轨迹
        x_p = self.decoder(z, size)

        return x_p, zmean, zlogvar


    def generate_mean_variance(self, seed_x, time, size):
        """通过多次采样估计生成轨迹的均值和方差

        对同一输入执行5次随机采样生成，统计输出轨迹的均值和方差，
        用于量化模型的不确定性。

        Args:
            seed_x: 种子输入序列
            time: 目标时间步张量
            size: 尺寸张量

        Returns:
            tuple: (mean, var)
                - mean: 生成轨迹的均值，形状为 (dim0, 1, dim2)
                - var: 生成轨迹的方差，形状为 (dim0, 1, dim2)
        """
        seed_t_len = seed_x.shape[0]

        dim0 = size.size()[0]
        dim1 = size.size()[1]
        dim2 = size.size()[2]

        flat = torch.zeros([1, 1, dim0*dim2]).to(device)

        # 多次采样生成，估计输出分布
        for i in range(0, 5):

            output, _, _ = self.generate_with_seed(seed_x, time)

            output = output.to(device)

            output = output.view(1, 1, dim0*dim2)
            flat = torch.cat((flat, output), 0)


        # 去除初始零行
        length = flat.size()[0]
        flat = flat[1:length, :, :]

        # 计算多次采样的均值和方差
        mean = flat.mean(dim = 0)
        var = flat.var(dim = 0)
        mean = mean.view(dim0, 1, dim2)
        var = var.view(dim0, 1, dim2)

        return mean, var

    def train(self, input_tensor, target_tensor):
        """VAE训练步骤

        执行一次完整的训练迭代：前向传播 → 计算重构损失 + KL散度 → 反向传播 → 参数更新。

        损失函数 = 重构损失(MSE) + KL散度
        KL散度 = -0.5 * Σ(1 + log σ² - μ² - exp(log σ²))

        Args:
            input_tensor: 输入张量
            target_tensor: 目标张量

        Returns:
            float: 本次训练的总损失值（重构损失 + KL散度）
        """
        max_length = 500

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        rows = target_tensor.size()[1]
        trials = target_tensor.size()[0]

        # 带重参数化采样的前向传播
        output, zmean, zlogvar = self.generate_with_seed(input_tensor.float(), target_tensor)

        # 计算重构损失（MSE）
        loss = self.criterion(output.float(), target_tensor.float())
        # 计算KL散度：KL(q(z|x) || p(z))，其中p(z) = N(0, I)
        kl_loss = -0.5 * torch.sum(1 + zlogvar - zmean.pow(2) - zlogvar.exp())
        # 总损失 = 重构损失 + KL散度
        loss = loss + kl_loss

        loss.backward()

        params = list(self.parameters())


        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def evaluate(self, input_tensor, target_tensor):
        """使用DTW损失评估模型性能

        使用动态时间规整（DTW）距离作为评估指标，
        相比MSE更能容忍时间轴上的微小偏移。

        Args:
            input_tensor: 输入张量
            target_tensor: 目标张量

        Returns:
            float: DTW损失值
        """
        max_length = 500

        # 使用可微DTW损失进行评估
        criterion = utils.DTW_Loss()

        rows = target_tensor.size()[1]
        trials = target_tensor.size()[0]

        output, _, _ = self.generate_with_seed(input_tensor.float(), target_tensor)

        loss = criterion(output.float(), target_tensor.float())



        return loss.item()




class PlanningGAT(nn.Module):
    """规划模块的GAT编码器

    与动力学模型的GAT编码器结构类似，但不输出VAE潜在分布参数，
    而是输出节点级特征和全局隐状态，供规划解码器预测动作原语序列。

    架构设计:
        1. 两层多头GAT + GRU: 提取空间-时序特征
        2. 全局隐状态映射: 将GRU最终隐状态映射到规划隐空间

    Args:
        g: DGL图对象
        in_dim: 输入特征维度
        hidden_dim: 规划隐层维度
        latent_dim: GAT输出/GRU隐状态维度
        num_heads: 注意力头数
    """

    def __init__(self, g, in_dim, hidden_dim, latent_dim, num_heads):
        super(PlanningGAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, latent_dim, num_heads)

        self.layer2 = MultiHeadGATLayer(g, latent_dim * num_heads, latent_dim, 1)

        self.input_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.nodes = len(g.nodes())



        self.gru_size = self.nodes * latent_dim
        self.hid2in = nn.Linear(latent_dim, in_dim)
        self.rnn = nn.GRU(self.gru_size, self.gru_size)
        # 规划隐状态映射层
        self.hid1 = nn.Linear(self.gru_size, hidden_dim)
        self.hid2 = nn.Linear(hidden_dim, hidden_dim)
        self.hid3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, target):
        """规划编码器前向传播

        Args:
            h: 输入序列，形状为 (seq_len, num_nodes, in_dim)
            target: 目标时间步张量

        Returns:
            tuple: (out, h0)
                - out: 最后一步的节点级特征，形状为 (num_nodes, latent_dim)
                - h0: 全局规划隐状态，形状为 (1, 1, hidden_dim)
        """
        temp = h[0]

        hidden = torch.zeros((1, 1, self.gru_size)).to(device)

        for i in range(0, target.size()[0]):

            h1 = self.layer1(temp)

            rnn_inp = h1.flatten()

            out, hid = self.rnn(rnn_inp.unsqueeze(0).unsqueeze(0), hidden)

            out = out.reshape(self.nodes, self.latent_dim)
            hid = hid.reshape(self.nodes, self.latent_dim)
            hidden = self.layer2(hid)
            hidden = hidden.flatten().unsqueeze(0).unsqueeze(0)
            temp = self.hid2in(out).squeeze(0)

        # 映射GRU最终隐状态到规划隐空间
        h0 = self.hid1(hidden.float())
        h0 = self.hid2(h0)
        h0 = self.hid3(h0)

        return out, h0



class PlanningDecoder(nn.Module):
    """规划模块的解码器

    基于GRU的自回归解码器，从规划隐状态出发逐步预测动作原语的概率分布。
    输出经过Softmax归一化，表示每个时间步选择各动作原语的概率。

    架构设计:
        1. GRU: 自回归地生成序列
        2. 多层全连接 + Softmax: 输出动作原语概率分布

    Args:
        input_dim: 输入维度
        output_dim: 输出维度（动作原语类别数）
        hidden_dim: 隐层维度
    """

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PlanningDecoder, self).__init__()
        self.input_dim = input_dim

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim


        self.gru = nn.GRU(hidden_dim, hidden_dim)

        self.l2h = nn.Linear(hidden_dim, hidden_dim)
        self.l2h2 = nn.Linear(hidden_dim, hidden_dim)
        self.l2h3 = nn.Linear(hidden_dim, hidden_dim)
        # 最终映射到动作原语类别数
        self.l2h4 = nn.Linear(hidden_dim, output_dim)
        # Softmax归一化，输出概率分布
        self.soft = nn.Softmax(2)

    def forward(self, hidden, target):
        """规划解码器前向传播

        Args:
            hidden: 规划隐状态，形状为 (1, 1, hidden_dim)
            target: 目标时间步张量，用于确定生成长度

        Returns:
            Tensor: 动作原语概率分布序列，形状为 (seq_len, 1, output_dim)
        """
        # 初始化GRU输入为零向量
        input = torch.zeros(1, 1, hidden.size()[2]).to(device)
        output = torch.zeros(1, target.size()[1], target.size()[2]).to(device)

        for i in range(0, target.size()[0]):
            zs, h = self.gru(input, hidden)

            input = zs

            # 多层全连接映射
            zs = self.l2h(zs)
            zs = self.l2h2(zs)
            zs = self.l2h3(zs)
            hs = self.l2h4(zs)

            hidden = h

            # Softmax归一化得到动作原语概率分布
            softmax = self.soft(hs)

            output = torch.cat((output, softmax))

        # 去除初始零行
        output = torch.cat((output[1:, :, :],))

        return output

class PlanningVAE(nn.Module):
    """规划模块的变分自编码器

    用于动作原语序列的预测与决策。编码器将轨迹编码为全局隐状态，
    解码器自回归地预测每个时间步的动作原语概率分布。
    与动力学VAE不同，此处使用交叉熵损失进行分类训练。

    架构设计:
        1. PlanningGAT编码器: 提取轨迹的空间-时序特征
        2. PlanningDecoder解码器: 预测动作原语概率分布
        3. 交叉熵损失: 用于多分类训练

    Args:
        input_size: 输入维度（节点数）
        target_size: 输出维度
        latent_size: 潜在空间维度
        encoder: PlanningGAT编码器实例
        decoder: PlanningDecoder解码器实例
        encoder_optimizer: 编码器优化器
        decoder_optimizer: 解码器优化器
        loss: 损失函数（交叉熵）
    """

    def __init__(self, input_size, target_size, latent_size, encoder, decoder, encoder_optimizer, decoder_optimizer,
                 loss):

        super(PlanningVAE, self).__init__()
        self.input_dim = input_size
        self.output_dim = target_size
        self.hidden_dim = latent_size
        self.nodes = input_size

        self.encoder = encoder
        self.decoder = decoder

        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.criterion = loss

    def forward(self, input_tensor, target_tensor):
        """规划VAE前向传播

        Args:
            input_tensor: 输入轨迹张量
            target_tensor: 目标动作原语张量

        Returns:
            Tensor: 动作原语概率分布序列
        """
        # 重塑输入为图节点格式
        input = input_tensor.reshape(input_tensor.size()[0], input_tensor.size()[1], self.nodes,
                                     int(input_tensor.size(2) / self.nodes)).squeeze(1)

        dim0 = input_tensor.size()[0]
        dim1 = input_tensor.size()[1]
        # 构建时间步索引表
        timetable = []
        for i in range(1, dim0 + 1):
            for j in range(1, dim1 + 1):
                timetable.append(i)

        time = torch.tensor(timetable).view(dim0, dim1).unsqueeze(-1)

        # 编码器：提取全局隐状态
        encoder_output, encoder_hidden = self.encoder(input, time)

        decoder_hidden = encoder_hidden


        # 解码器：预测动作原语概率分布
        decoder_output = self.decoder(decoder_hidden, target_tensor)



        return decoder_output

    def generate_with_seed(self, seed_x, t):
        """基于种子序列生成动作原语预测

        Args:
            seed_x: 种子输入序列
            t: 时间步张量

        Returns:
            Tensor: 预测的动作原语概率分布序列
        """
        # 重塑输入为图节点格式
        input = seed_x.reshape(seed_x.size()[0], seed_x.size()[1], self.nodes,
                               int(seed_x.size(2) / self.nodes)).squeeze(1)

        seed_t_len = seed_x.shape[0]
        # 编码器提取隐状态，解码器生成动作原语预测
        z_mean, z_log_var = self.encoder(input, t[:seed_t_len])
        x_p = self.decoder(z_mean, t)
        return x_p

    def getPrimitive(self, x, t):
        """获取完整动作原语序列

        从模型输出中提取6个关键时间步的动作原语预测，
        对应6个动作阶段（抓取、侧移、抬举、伸展、放置、收回）。

        Args:
            x: 输入轨迹张量
            t: 时间步张量

        Returns:
            list: 6个动作原语的索引列表，顺序为 [grasp, side, lift, extend, place, retract]
        """
        output = self.forward(x, t)
        # 从6个关键时间步提取动作原语：每个原语间隔12步
        prim1 = np.argmax(output[9][-1].detach().cpu())
        prim2 = np.argmax(output[21][-1].detach().cpu())
        prim3 = np.argmax(output[33][-1].detach().cpu())
        prim4 = np.argmax(output[45][-1].detach().cpu())
        prim5 = np.argmax(output[57][-1].detach().cpu())
        prim6 = np.argmax(output[69][-1].detach().cpu())

        output = []
        output.append(prim1.tolist())
        output.append(prim2.tolist())
        output.append(prim3.tolist())
        output.append(prim4.tolist())
        output.append(prim5.tolist())
        output.append(prim6.tolist())

        return output

    def getCurrentPrimitive(self, x, t):
        """获取当前时间步的动作原语

        从模型输出的最后一个时间步提取当前动作原语预测。

        Args:
            x: 输入轨迹张量
            t: 时间步张量

        Returns:
            int: 当前动作原语的索引
        """


        output = self.forward(x, t)

        # 取最后时间步的最大概率类别作为当前原语
        return np.argmax(output[-1][-1].detach().cpu())

    def train(self, input_tensor, target_tensor):
        """规划VAE训练步骤

        使用交叉熵损失训练动作原语分类器。

        Args:
            input_tensor: 输入轨迹张量
            target_tensor: 目标动作原语标签张量

        Returns:
            float: 本次训练的交叉熵损失值
        """
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()


        # 重塑输入为图节点格式
        input = input_tensor.reshape(input_tensor.size()[0], input_tensor.size()[1], self.nodes,
                                     int(input_tensor.size(2) / self.nodes)).squeeze(1)



        dim0 = input_tensor.size()[0]
        dim1 = input_tensor.size()[1]
        # 构建时间步索引表
        timetable = []
        for i in range(1, dim0 + 1):
            for j in range(1, dim1 + 1):
                timetable.append(i)

        time = torch.tensor(timetable).view(dim0, dim1).unsqueeze(-1)

        # 编码器前向传播
        encoder_output, encoder_hidden = self.encoder(input, time)

        decoder_hidden = encoder_hidden

        # 解码器前向传播
        decoder_output = self.decoder(decoder_hidden, target_tensor)

        # 去除batch维度
        decoder_output = decoder_output.squeeze(1)
        target_tensor = target_tensor.squeeze(1)

        # 将目标标签从one-hot转为类别索引
        indices = []
        for i in range(0, target_tensor.size()[0]):
            indices.append(np.argmax(target_tensor[i].cpu()))

        indicestensor = torch.tensor(indices).to(device)

        # 计算交叉熵损失
        loss = self.criterion(decoder_output.float(), indicestensor)

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()
