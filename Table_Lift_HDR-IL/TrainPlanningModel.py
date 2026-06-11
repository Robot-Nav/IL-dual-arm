"""规划模型训练模块

本模块负责训练HDR-IL框架中的规划模型（PlanningVAE），该模型用于预测
动作原语序列。与动力学模型不同，规划模型将轨迹编码后解码为动作原语的
概率分布（Softmax输出），使用交叉熵损失进行分类训练。

核心流程:
    1. 构建全连接DGL图
    2. 初始化PlanningGAT编码器和PlanningDecoder解码器
    3. 组装PlanningVAE模型
    4. 加载预训练模型（可选）
    5. 使用交叉熵损失训练规划模型
    6. 保存训练后的模型参数

关键全局变量:
    seqmodel: 规划VAE模型实例
    params: 训练超参数字典
"""

import random

import utils
from Models import HDRIL_Models
from torch import optim
import torch
import torch.nn as nn

from torch.utils import data
import dgl

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {

    "runs" : 2500,
    "runsize" : 70,
    "load_model" : False,
    "train_model" : True,
    "n_epochs" : 2,
    "sequencelength" : 70,
    "n_hidden" :  21,
    "n_latent" : 512,
    "n_iters" : 20,
    "lr_encoder": .00005,
    "lr_decoder": .00005

}

# 规划模型保存路径
PATH = 'planningmodel.pt'



# 训练集大小 = 运行次数 × 每次演示长度
trainsize = params["runs"]*params["runsize"]
train_set = utils.BaxterDataset()


# 输入维度由数据集特征列数决定
input_size = train_set.columns.size()[1]
maxsize = len(train_set)
sequencelength = params["sequencelength"]

# 划分训练集和测试集
trainingdata = data.Subset(train_set, indices=list(range(0, trainsize)))
testdata = data.Subset(train_set, indices=list(range(trainsize, maxsize)))



# 构建全连接图：每个特征维度作为节点，所有节点互相连接
g2 = dgl.DGLGraph().to(device)
nodes = input_size
g2.add_nodes(nodes)
for i in range(0, nodes):
    for j in range(0, nodes):
        g2.add_edge(i, j)


"""规划模型初始化"""

input_size = input_size
output_size = input_size
n_latent = params['n_latent']
n_iters = params['n_iters']

current_loss = 0
test_loss_total = 0
all_losses = []
test_losses = []
losses = []

force = []


# 规划GAT编码器：将轨迹编码为全局隐状态
gcn1 = HDRIL_Models.PlanningGAT(g2, int(input_size / nodes), n_latent, input_size, 1)

# 规划解码器：输出维度为1（动作原语类别概率）
decoder = HDRIL_Models.PlanningDecoder(input_size, 1, n_latent)

# Adam优化器
encoder_optimizer = optim.Adam(gcn1.parameters(), lr= params["lr_encoder"])
decoder_optimizer = optim.Adam(decoder.parameters(), lr= params["lr_decoder"])

# 交叉熵损失函数，用于动作原语分类
criterion = nn.CrossEntropyLoss()

# 组装规划VAE模型
seqmodel = HDRIL_Models.PlanningVAE(input_size, output_size, n_latent, gcn1, decoder, encoder_optimizer,
                                    decoder_optimizer,
                                    criterion)

# 将模型移至GPU
seqmodel = seqmodel.cuda()



"""模型训练"""

if (params["load_model"] == True):
    seqmodel.load_state_dict(torch.load(PATH))



if (params["train_model"] == True):



    for epoch_idx in range(params["n_epochs"]):

        print(epoch_idx)

        i = 0

        while i < trainsize:



            # 随机选择一个演示的起始位置
            s = random.randint(0, params["runs"] - 1) * params['runsize']
            # 获取该位置的动作原语标签
            primitive = train_set.labels[s]


            # 获取当前窗口的坐标和原语标签
            c, p = train_set[s:s + sequencelength]
            c1, _ = train_set[s + 1:s + 1 + sequencelength]
            i += sequencelength

            rows = int(c.size()[0] / sequencelength)

            # 重塑为 (seq_len, batch, features) 格式
            startcord = c.reshape(rows, sequencelength, input_size).transpose(0, 1).cuda()
            endcord = c1.reshape(rows, sequencelength, input_size).transpose(0, 1).cuda()
            target = p.reshape(rows, sequencelength, 1).transpose(0, 1).cuda()

            trainloss = 0
            testloss = 0

            # 训练规划模型：输入轨迹，预测动作原语序列
            avgloss = seqmodel.train(startcord.float(), target.float())

            trainavgloss = avgloss

            print(trainavgloss)

            if trainloss > 100000:
                break
            losses.append(trainavgloss)

            size = 0

            iteration = 0
            j = trainsize

    print(losses)
    print(test_losses)

    # 保存训练后的规划模型
    torch.save(seqmodel.state_dict(), PATH)
