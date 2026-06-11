"""轨迹投影生成模块

本模块使用训练好的6个动力学VAE模型，依次对每个动作原语生成轨迹预测，
并将所有原语的预测拼接为完整的演示轨迹。同时估计每个时间步的方差，
用于量化模型的不确定性。最终将预测结果和真实演示写入CSV文件。

核心流程:
    1. 加载6个训练好的动力学模型（从TrainDynamicModels模块）
    2. 对每个演示，依次用6个模型生成各动作原语的轨迹
    3. 通过5次采样估计均值和方差（不确定性量化）
    4. 拼接所有原语预测，形成完整轨迹
    5. 将预测轨迹和真实演示写入CSV文件

命令行参数:
    -trainiters: 训练迭代次数（默认1）
    -startindex: 生成起始行索引（默认0）
    -datasize: 每个演示的数据行数（默认55）
    -features: 特征维度数（默认21）
"""

#from tqdm import tqdm_notebook as tqdm

#import seaborn as sns

#import A1PrimitiveData
#import A3TrainSoftmax
import TrainDynamicModels as B3TrainODE
import utils

#sns.color_palette("bright")

import csv
import torch
from matplotlib import pyplot as plt
import numpy as np
import argparse



# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('-trainiters', default=1, help='Number of training iterations to output')
parser.add_argument('-startindex', default=0, help='Row to start generation')
parser.add_argument('-datasize', default=55, help='Number of rows in each demonstration')
parser.add_argument('-features', default=21, help='Number of features')
args = parser.parse_args()



device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

"""加载所有模型"""

datasize = args.datasize
trainsize = args.datasize*args.trainiters
start = args.startindex

features = args.features

# 从TrainDynamicModels模块获取6个动力学模型
models = {

0: B3TrainODE.graspmodel,
1: B3TrainODE.liftmodel,
2: B3TrainODE.extendmodel,
3: B3TrainODE.placemodel,
4: B3TrainODE.retractmodel,
5: B3TrainODE.sidemodel,

}

#seqmodel = A3TrainSoftmax.seqmodel


# 获取模型保存路径
path1 = B3TrainODE.path1
path2 = B3TrainODE.path2
path3 = B3TrainODE.path3
path4 = B3TrainODE.path4
path5 = B3TrainODE.path5
path6 = B3TrainODE.path6


# 加载各模型的预训练参数
models[0].load_state_dict(torch.load(path1))
models[1].load_state_dict(torch.load(path2))
models[2].load_state_dict(torch.load(path3))
models[3].load_state_dict(torch.load(path4))
models[4].load_state_dict(torch.load(path5))
models[5].load_state_dict(torch.load(path6))


# 加载数据集
train_set = utils.BaxterDataset()


with torch.no_grad():


    # 初始化输出张量，形状为 (datasize, 1, features)
    outputsize = torch.zeros([args.datasize, 1, features])

    # 初始化累积张量：预测轨迹、真实轨迹、方差
    visited = torch.zeros([1, 1, features]).cuda()
    truth = torch.zeros([1, 1, features]).cuda()
    varianceslist = torch.zeros([1, 1, features]).cuda()


    while start < trainsize:

        print(datasize, "datasize", start, trainsize)

        # 获取起始点数据作为种子
        a, _ = train_set[start:start + 1]
        a = a.unsqueeze(1).cuda()


        print(a)
        print(trainsize)

        # 获取完整演示的真实数据
        a1, _ = train_set[start:start + datasize]
        a1 = a1.unsqueeze(1).cuda()

        start = start + datasize
        print(start, "s")

        variances = torch.zeros([1, 1, features]).cuda()


        # 依次用6个动力学模型生成各动作原语的轨迹
        test = [0,1,2,3,4,5]


        for primitive in test:


            print("size", a.size())
            # 使用当前原语模型生成轨迹均值和方差（5次采样估计）
            mean, var = models[primitive].generate_mean_variance(a[-1].unsqueeze(1).float(), outputsize, outputsize)
            print(var)
            print(a.size(), mean.size())
            # 将生成的轨迹拼接到已有轨迹后面
            a = torch.cat((a.float(), mean.float()), 0)
            # 累积方差
            variances = torch.cat((variances.float(), var.float()), 0)
            print("grasp")



        # 累积本次演示的预测、真实和方差数据
        visited = torch.cat((visited.float(), a[0:datasize, :, :].float()), 0)
        truth = torch.cat((truth.float(), a1[0:datasize, :, :].float()), 0)
        varianceslist = torch.cat((varianceslist.float(), variances[0:datasize, :, :].float()), 0)

        headers = utils.outputHeaders()


# 截断多余数据
a = a[0:trainsize - start, :, :]
a1 = a1[0:trainsize - start, :, :]
variances = variances[0:trainsize - start, :, :]



"""将结果写入CSV文件

每行包含预测轨迹特征和真实演示特征，列名为对应的特征名称。
"""
with open('projection_data/out.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(headers)

    for i in range(1, visited.size()[0]):

        thewriter.writerow(visited[i][0].tolist() + truth[i][0].tolist())
