"""动力学模型训练模块

本模块负责训练HDR-IL框架中的6个动力学VAE模型，每个模型对应一个动作原语
（抓取、侧移、抬举、伸展、放置、收回）。每个模型使用GAT+GRU编码器和GRU解码器
组成的VAE架构，学习对应动作原语的轨迹动态。

核心流程:
    1. 构建全连接DGL图，初始化6组编码器-解码器-优化器
    2. 加载预训练模型（可选）
    3. 按动作原语标签分发训练数据到对应模型
    4. 使用MSE重构损失 + KL散度训练各VAE
    5. 保存训练后的模型参数
    6. 使用DTW损失评估模型性能（可选）

关键全局变量:
    models: 6个VAE模型的字典，键为动作原语索引(0-5)
    params: 训练超参数字典
"""

import dgl
import utils
from Models import HDRIL_Models
from torch import optim
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils import data



params = {

    "runs": 10,
    "runsize": 70,
    "load_model": True,
    "train_model" : False,
    "evaluate" : False,
    "n_epochs" : 1,
    "n_latent" : 512,
    "lr_encoder": .00005,
    "lr_decoder": .00005

}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载Baxter机器人数据集
train_set = utils.BaxterDataset()


# 训练集大小 = 运行次数 × 每次演示长度
trainsize = params["runsize"] * params["runs"]
test_size = len(train_set)

# 按索引划分训练子集
trainingdata = data.Subset(train_set, indices=list(range(0, trainsize)))


# 输入/输出维度：由数据集特征列数决定
input_size = train_set.columns.size()[1]
output_size = input_size


"""模型保存路径"""

path1 = 'xy/graspmodel.pt'
path2 = 'xy/liftmodel.pt'
path3 = 'xy/extendmodel.pt'
path4 = 'xy/placemodel.pt'
path5 = 'xy/retractmodel.pt'
path6 = 'xy/sidemodel.pt'


"""初始化模型参数"""

n_latent = params['n_latent']

current_loss = 0
test_loss_total = 0
all_losses = []
dtw_errors = []
test_losses = []
losses = []

force = []

# 初始化6个编码器和6个解码器

headers = utils.outputHeaders()


"""构建全连接图

创建一个包含所有节点的全连接图，每个节点代表机器人的一个特征维度。
全连接结构确保所有特征之间可以互相传递信息。
"""
g2 = dgl.DGLGraph().to(device)
nodes = input_size
g2.add_nodes(nodes)
# 添加全连接边：每个节点与所有其他节点相连
for i in range(0, nodes):
    for j in range(0, nodes):
        g2.add_edge(i, j)



# MSE损失函数，用于轨迹重构
criterion = nn.MSELoss()

# 初始化6个GAT编码器，每个对应一个动作原语
gcn1 = HDRIL_Models.GAT(g2, int(input_size / nodes), n_latent, output_size, 1)
gcn2 = HDRIL_Models.GAT(g2, int(input_size / nodes), n_latent, output_size, 1)
gcn3 = HDRIL_Models.GAT(g2, int(input_size / nodes), n_latent, output_size, 1)
gcn4 = HDRIL_Models.GAT(g2, int(input_size / nodes), n_latent, output_size, 1)
gcn5 = HDRIL_Models.GAT(g2, int(input_size / nodes), n_latent, output_size, 1)
gcn6 = HDRIL_Models.GAT(g2, int(input_size / nodes), n_latent, output_size, 1)

# 初始化6个GRU解码器
p1decoder = HDRIL_Models.Decoder(input_size, output_size, n_latent)
p2decoder = HDRIL_Models.Decoder(input_size, output_size, n_latent)
p3decoder = HDRIL_Models.Decoder(input_size, output_size, n_latent)
p4decoder = HDRIL_Models.Decoder(input_size, output_size, n_latent)
p5decoder = HDRIL_Models.Decoder(input_size, output_size, n_latent)
p6decoder = HDRIL_Models.Decoder(input_size, output_size, n_latent)

# 初始化6组Adam优化器（编码器 + 解码器各一组）
p1encoder_optimizer = optim.Adam(gcn1.parameters(), lr= params["lr_encoder"])
p1decoder_optimizer = optim.Adam(p1decoder.parameters(), lr= params["lr_decoder"])
p2encoder_optimizer = optim.Adam(gcn2.parameters(), lr= params["lr_encoder"])
p2decoder_optimizer = optim.Adam(p2decoder.parameters(), lr= params["lr_decoder"])
p3encoder_optimizer = optim.Adam(gcn3.parameters(), lr= params["lr_encoder"])
p3decoder_optimizer = optim.Adam(p3decoder.parameters(), lr= params["lr_decoder"])
p4encoder_optimizer = optim.Adam(gcn4.parameters(), lr= params["lr_encoder"])
p4decoder_optimizer = optim.Adam(p4decoder.parameters(), lr= params["lr_decoder"])
p5encoder_optimizer = optim.Adam(gcn5.parameters(), lr= params["lr_encoder"])
p5decoder_optimizer = optim.Adam(p5decoder.parameters(), lr= params["lr_decoder"])
p6encoder_optimizer = optim.Adam(gcn6.parameters(), lr= params["lr_encoder"])
p6decoder_optimizer = optim.Adam(p6decoder.parameters(), lr= params["lr_decoder"])

# 组装6个VAE模型：编码器 + 解码器 + 优化器 + 损失函数
graspmodel = HDRIL_Models.VAE(input_size, output_size, n_latent, gcn1, p1decoder, p1encoder_optimizer,
                              p1decoder_optimizer, criterion)
sidemodel = HDRIL_Models.VAE(input_size, output_size, n_latent, gcn2, p2decoder, p2encoder_optimizer,
                             p2decoder_optimizer, criterion)
liftmodel = HDRIL_Models.VAE(input_size, output_size, n_latent, gcn3, p3decoder, p3encoder_optimizer,
                             p3decoder_optimizer, criterion)
extendmodel = HDRIL_Models.VAE(input_size, output_size, n_latent, gcn4, p4decoder, p4encoder_optimizer,
                               p4decoder_optimizer, criterion)
placemodel = HDRIL_Models.VAE(input_size, output_size, n_latent, gcn5, p5decoder, p5encoder_optimizer,
                              p5decoder_optimizer, criterion)
retractmodel = HDRIL_Models.VAE(input_size, output_size, n_latent, gcn6, p6decoder, p6encoder_optimizer,
                                p6decoder_optimizer, criterion)



# 模型字典：键为动作原语索引，值为对应VAE模型
models = {

    0: graspmodel,
    1: sidemodel,
    2: liftmodel,
    3: extendmodel,
    4: placemodel,
    5: retractmodel

}

# 将所有模型移至GPU
for key in models.keys():
    models[key] = models[key].cuda()



if (params["load_model"] == True):
    # 加载预训练的抓取模型参数
    graspmodel.load_state_dict(torch.load(path1))


if (params["train_model"] == True):
    for epoch_idx in range(params["n_epochs"]):

        for i in range(0, params["runs"]*6):

            # 每6步重置一次演示起始位置（对应6个动作原语）
            if i % 6 == 0:
                s = random.randint(0, params['runs'] - 1) * params['runsize']

            # 防止索引越界
            if s + sequencelength + 1 >= len(train_set):
                s = random.randint(0, params['runs'] - 1) * params['runsize']

            print("epoch", epoch_idx, "i", i, "s", s)

            # 获取当前样本的动作原语标签
            primitive = train_set.labels[s]

            # 第一个原语序列长度为10，后续为12
            if i % 6 == 0:
                sequencelength = 10

            else:
                sequencelength = 12


            print(sequencelength, "=============================")
            length = int(sequencelength)
            # 获取当前窗口和下一窗口的数据
            c, l = train_set[s:s + sequencelength]
            c1, l1 = train_set[s + 1:s + 1 + sequencelength]
            s += sequencelength

            rows = int(c.size()[0] / length)

            # 重塑为 (seq_len, batch, features) 格式并转置
            startcord = c.reshape(rows, length, input_size).transpose(0, 1).cuda()
            endcord = c1.reshape(rows, length, input_size).transpose(0, 1).cuda()


            trainloss = 0
            testloss = 0

            print("primitive", s, primitive[0].item())

            # 根据动作原语标签选择对应模型进行训练
            avgloss = models[primitive[0].item()].train(startcord, endcord)


            print("avg loss" , avgloss)

            if trainloss > 100000:
                break
            losses.append(avgloss)

            size = 0
            magnitude = 0
            iteration = 0

    # 保存所有训练后的模型参数
    torch.save(graspmodel.state_dict(), path1)
    torch.save(liftmodel.state_dict(), path2)
    torch.save(extendmodel.state_dict(), path3)
    torch.save(placemodel.state_dict(), path4)
    torch.save(retractmodel.state_dict(), path5)
    torch.save(sidemodel.state_dict(), path6)

    print(losses)
    print(test_losses)

    # 输出结果

    csvlist = []




if params["evaluate"] == True:
    s = 0

    for i in range(0, params["runs"]):

        sequencelength = params["runsize"]

        length = int(sequencelength)
        c = train_set[s:s + sequencelength]
        c1 = train_set[s + 1:s + 1 + sequencelength]

        rows = int(c.size()[0] / length)
        features1 = int(input_size / nodes)
        output1 = int(output_size / nodes)

        startcord = c.reshape(rows, length, input_size).transpose(0, 1).cuda()
        endcord = c1.reshape(rows, length, input_size).transpose(0, 1).cuda()

        trainloss = 0
        testloss = 0


        # 使用DTW损失评估抓取模型
        avgloss = graspmodel.evaluate(startcord, endcord)

        print(avgloss)

        if trainloss > 100000:
            break
        dtw_errors.append(avgloss)

        size = 0
        magnitude = 0
        iteration = 0

    # 输出平均DTW误差
    print(np.mean(dtw_errors), "dtw errors")
