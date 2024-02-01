# 合金生产问题的未知参数版本

# 合金生产问题：家合金生产工厂需要生产一定量的特定合金。
# 这需要混合M种金属，为此，它必须至少获得m∈[M]中的每种金属req_m吨
# 原材料将从K个供应商那里获得，每个供应商提供一种不同类型的矿石。
# 工厂计划从矿点购买矿石，然后自行提取金属。
# 第k个矿点（k∈[K]）提供的矿石含有m（m∈[0, 1]）的con_{km}分数的材料
# 每吨成本为costk。
# 工厂的目标是以最低成本满足每种金属的需求。

# 未知量是不同矿点的矿石中的金属含量
# 工厂会根据历史购买记录(矿石产地,矿石报告)估计金属浓度
# 矿点的数量会发生变化,给定的特征集是一个矿点的7000(时间步长)*4096(特征)的训练数据
# 以及一个7000(时间步长)的验证标签
# 

import numpy as np
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import gurobipy as gp
import logging
import gurobipy as gp
from gurobipy import GRB

import os, sys
os.chdir(sys.path[0])

print(torch.__version__)

# 定义自己的输出格式
def my_print(description: str, x):
    if x is not None:
        print(f'---------------------------- {description} -----------------------------------')
        print(x)
        print('----------------------------- end ----------------------------------------')
    else:
        print(f'Start: ---------------------------- {description} -----------------------------------')
    print()

# 2种元素
rowSizeG = 2
colSizeG = 10
varNum = colSizeG
featureNum = 4096
trainSize = 350
cap = [627.54, 369.72]
# cap = [0.8, 60, 40, 2.5]
penaltyTerm = 0.25

def actual_obj(cTemp, GTemp, hTemp, n_instance):
    obj_list = []
    for num in range(n_instance):
        c = np.zeros((colSizeG))
        cntC = num * colSizeG
        for i in range(colSizeG):
            c[i] = cTemp[cntC]
            cntC = cntC + 1
        c = c.tolist()
        h = np.zeros((rowSizeG))
        cntH = num * rowSizeG
        for i in range(rowSizeG):
            h[i] = hTemp[cntH]
            cntH = cntH + 1
        h = h.tolist()
        
        G = np.zeros((rowSizeG, colSizeG))
        cnt = num * rowSizeG * colSizeG
        for i in range(rowSizeG):
            for j in range(colSizeG):
                G[i][j] = GTemp[cnt]
                cnt = cnt + 1
        G = G.tolist()
        
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(varNum, vtype=GRB.CONTINUOUS, name='x')
        m.setObjective(x.prod(c), GRB.MINIMIZE)
        for i in range(rowSizeG):
            m.addConstr((x.prod(G[i])) >= h[i])

        m.optimize()
        sol = []
        for i in range(varNum):
            sol.append(x[i].x)
        
        objective = m.objVal
        obj_list.append(objective)
        
    return np.array(obj_list)
    
def correction_single_obj(c, realG, preG, h, penalty):
    rowSizeG = realG.shape[0]
    if preG.all() >= 0:
        realG = realG.tolist()
        preG = preG.tolist()
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(varNum, vtype=GRB.CONTINUOUS, name='x')
        m.setObjective(x.prod(c), GRB.MINIMIZE)
        for i in range(rowSizeG):
            m.addConstr((x.prod(preG[i])) >= h[i])

        m.optimize()
        predSol = []
        try:
            for i in range(varNum):
                predSol.append(x[i].x)

            objective = m.objVal
        except:
            for i in range(varNum):
                predSol.append(0)
            objective = 0
        
        # Stage 2:
        m2 = gp.Model()
        m2.setParam('OutputFlag', 0)
        x = m2.addVars(varNum, vtype=GRB.CONTINUOUS, name='x')
        sigma = m2.addVars(varNum, vtype=GRB.CONTINUOUS, name='sigma')

        OBJ = objective
        for i in range(varNum):
            OBJ = OBJ + (1 + penalty[i]) * c[i] * sigma[i]
        m2.setObjective(OBJ, GRB.MINIMIZE)

        for i in range(rowSizeG):
            m2.addConstr((x.prod(realG[i]) + sigma.prod(realG[i])) >= h[i])
        for i in range(varNum):
            m2.addConstr(x[i] == predSol[i])

        m2.optimize()
        objective = m2.objVal
        sol = []
        for i in range(varNum):
            sol.append(sigma[i].x)
        
    return objective

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# 纯傻逼代码,batchsize大小表示con的维度...batchsize=colSizeG * rowSizeG
def make_fc(num_layers, num_features, num_targets=1,
            activation_fn = nn.ReLU,intermediate_size=512, regularizers = True):
    net_layers = [nn.Linear(num_features, intermediate_size),activation_fn()]
    for hidden in range(num_layers-2):
        net_layers.append(nn.Linear(intermediate_size, intermediate_size))
        net_layers.append(activation_fn())
    net_layers.append(nn.Linear(intermediate_size, num_targets))
    net_layers.append(activation_fn())
    return nn.Sequential(*net_layers)
        

class MyCustomDataset(Dataset):
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

    def __len__(self):
        return len(self.value)

    def __getitem__(self, idx):
        return self.feature[idx], self.value[idx]


import sys
import ip_model_whole as ip_model_wholeFile
from ip_model_whole import IPOfunc

# penalty是给定值，但实际中penalty应该是随决策变量、下一次决策变量和时间发生改变的
class Intopt:
    def __init__(self, c, h, A, b, penalty, n_features, num_layers=5, smoothing=False, thr=0.1, max_iter=None, method=1, mu0=None,
                 damping=0.5, target_size=1, epochs=8, optimizer=optim.Adam,
                 batch_size=rowSizeG*colSizeG, **hyperparams):
        self.c = c
        self.h = h
        self.A = A
        self.b = b

        self.penalty = penalty
        self.target_size = target_size
        self.n_features = n_features
        self.damping = damping
        self.num_layers = num_layers

        self.smoothing = smoothing
        self.thr = thr
        self.max_iter = max_iter
        self.method = method
        self.mu0 = mu0

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.hyperparams = hyperparams
        self.epochs = epochs

        self.model = make_fc(num_layers=self.num_layers,num_features=n_features)

        self.optimizer = optimizer(self.model.parameters(), **hyperparams)

    def fit(self, feature, value):
        my_print("Intopt", None)
        train_df = MyCustomDataset(feature, value)

        criterion = nn.L1Loss(reduction='mean')  # nn.MSELoss(reduction='mean')
        grad_list = np.zeros(self.epochs)
        for e in range(self.epochs):
            total_loss = 0
            if e < 1:
                lr = 1e-1
                train_dl = DataLoader(train_df, batch_size=self.batch_size, shuffle=False)

                # 傻逼一样的代码，传入参数feature value先构成一下数据集然后在这里再解构成feature和label
                for feature, value in train_dl:
                    self.optimizer.zero_grad()
                    op = self.model(feature).squeeze()
                    
                    loss = criterion(op, value)
                    # 这里的criterion仍然用的是预测值和真实值相减这样的loss
                    # 这里用item是因为loss是只包含一个值的tensor类型变量
                    total_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()

            else:
                train_dl = DataLoader(train_df, batch_size=self.batch_size, shuffle=False)
                
                corr_obj_list = []
                num = 0
                batchCnt = 0
                # Variable已被弃用，Tensor已经可以实现求导功能
                # loss = Variable(torch.tensor(0.0, dtype=torch.double), requires_grad=True)
                loss = torch.tensor(0.0, dtype=torch.double, requires_grad=True)
                for feature, value in train_dl:
                    self.optimizer.zero_grad()
                    op = self.model(feature).squeeze()
                    while torch.min(op) <= 0 or torch.isnan(op).any() or torch.isinf(op).any():
                        self.optimizer.zero_grad()
                        self.model = make_fc(num_layers=self.num_layers,num_features=self.n_features)
                        op = self.model(feature).squeeze()
    
                    price = np.zeros(colSizeG)
                    cap = np.zeros(rowSizeG)
                    penaltyVector = np.zeros(colSizeG)
                    for i in range(colSizeG):
                        price[i] = self.c[i+num*colSizeG]
                        penaltyVector[i] = self.penalty[i+num*colSizeG]
                    for j in range(rowSizeG):
                        cap[j] = self.h[j+num*rowSizeG]
                    
                    c_torch = torch.from_numpy(price).float()
                    h_torch = torch.from_numpy(cap).float()
                    A_torch = torch.from_numpy(self.A).float()
                    b_torch = torch.from_numpy(self.b).float()
                    penalty_torch = torch.from_numpy(penaltyVector).float()
                    
                    cntG = 0
                    G_torch = torch.zeros((rowSizeG, colSizeG))
                    for i in range(rowSizeG):
                        for j in range(colSizeG):
                            G_torch[i][j] = value[cntG]
                            cntG = cntG + 1
                    
                    cntOp = 0
                    op_torch = torch.zeros((rowSizeG, colSizeG))
                    for i in range(rowSizeG):
                        for j in range(colSizeG):
                            op_torch[i][j] = op[cntOp]
                            cntOp = cntOp + 1
                    x_s2 = IPOfunc(A=A_torch, b=b_torch, h=-h_torch, c=c_torch, GTrue=-G_torch, penalty=penalty_torch, max_iter=self.max_iter, thr=self.thr, damping=self.damping,
                                smoothing=self.smoothing)(-op_torch)
                    x_s1 = ip_model_wholeFile.x_s1
                    G_real_numpy = value.numpy()
                    newLoss = torch.dot(penalty_torch * c_torch, abs(x_s2-x_s1).float()) + (x_s2 * c_torch).sum()
                    corr_obj_list.append(newLoss.item())
                    loss = loss + newLoss
                    batchCnt = batchCnt + 1
                    total_loss += newLoss.item()
                    
                    newLoss.backward()
                    self.optimizer.step()
                    
                    num = num + 1
                total_loss = total_loss/trainSize
            logging.info("EPOCH Ends")

            print("Epoch{} ::loss {} ->".format(e,total_loss))
            grad_list[e] = total_loss

            if e > 0 and grad_list[e] >= grad_list[e-1]:
                break

    def val_loss(self, feature, value):
        valueTemp = value.numpy()
        test_instance = len(valueTemp) / self.batch_size
        real_obj = actual_obj(self.c, value, self.h, n_instance=int(test_instance))

        self.model.eval()
        criterion = nn.L1Loss(reduction='mean')  # nn.MSELoss(reduction='sum')
        valid_df = MyCustomDataset(feature, value)
        valid_dl = DataLoader(valid_df, batch_size=self.batch_size, shuffle=False)

        obj_list = []
        corr_obj_list = []
        predVal = torch.zeros(len(valueTemp))
        
        num = 0
        for feature, value in valid_dl:
            op = self.model(feature).squeeze()
            for i in range(rowSizeG*colSizeG):
                predVal[i+num*rowSizeG*colSizeG] = op[i]
            loss = criterion(op, value)

            price = np.zeros(colSizeG)
            cap = np.zeros(rowSizeG)
            penaltyVector = np.zeros(colSizeG)
            for i in range(colSizeG):
                price[i] = self.c[i+num*colSizeG]
                penaltyVector[i] = self.penalty[i+num*colSizeG]
            for j in range(rowSizeG):
                cap[j] = self.h[j+num*rowSizeG]
            
            cntG = 0
            realG = np.zeros((rowSizeG, colSizeG))
            for i in range(rowSizeG):
                for j in range(colSizeG):
                    realG[i][j] = value[cntG]
                    cntG = cntG + 1
            
            cntOp = 0
            predG = np.zeros((rowSizeG, colSizeG))
            for i in range(rowSizeG):
                for j in range(colSizeG):
                    predG[i][j] = op[cntOp]
                    cntOp = cntOp + 1
            
            price = price.tolist()
            cap = cap.tolist()
            
            corrrlst = correction_single_obj(price, realG, predG, cap, penaltyVector)
            corr_obj_list.append(corrrlst)
            num = num + 1

        self.model.train()
        return abs(np.array(corr_obj_list) - real_obj), predVal

def main():

    A_data = np.zeros((2, colSizeG))
    b_data = np.zeros(2)

    h_train = np.zeros(trainSize*rowSizeG)
    h_test = np.zeros(trainSize*rowSizeG)
    for i in range(trainSize*rowSizeG):
        h_train[i] = cap[i%rowSizeG]
        h_test[i] = cap[i%rowSizeG]

    startmark = 0
    endmark = startmark + 1

    print("*** HSD ****")

    testTime = 10
    recordBest = np.zeros((1, testTime))

    stopCriterion = 0
    if penaltyTerm == 0.25:
        stopCriterion = 50
    elif penaltyTerm == 0.5:
        stopCriterion = 65
    elif penaltyTerm == 1:
        stopCriterion = 90
    elif penaltyTerm == 2:
        stopCriterion = 120
    elif penaltyTerm == 4:
        stopCriterion = 160
    elif penaltyTerm == 8:
        stopCriterion = 180
        

    for testi in range(startmark, endmark):
        
        c_train = np.loadtxt('./Testdata/AlloyProduction/train_prices/train_prices(' + str(testi) + ').txt')
        x_train = np.loadtxt('./Testdata/AlloyProduction/train_features/train_features(' + str(testi) + ').txt')
        y_train = np.loadtxt('./Testdata/AlloyProduction/train_weights/train_weights(' + str(testi) + ').txt')
        penalty_train = np.loadtxt('./Testdata/AlloyProduction/train_penalty' + str(penaltyTerm) + '/train_penalty(' + str(testi) + ').txt')
        feature_train = torch.from_numpy(x_train).float()
        value_train = torch.from_numpy(y_train).float()
        meanVal = np.mean(y_train)

        c_test = np.loadtxt('./Testdata/AlloyProduction/test_prices/test_prices(' + str(testi) + ').txt')
        x_test = np.loadtxt('./Testdata/AlloyProduction/test_features/test_features(' + str(testi) + ').txt')
        y_test = np.loadtxt('./Testdata/AlloyProduction/test_weights/test_weights(' + str(testi) + ').txt')
        penalty_test = np.loadtxt('./Testdata/AlloyProduction/test_penalty' + str(penaltyTerm) + '/test_penalty(' + str(testi) + ').txt')
        feature_test = torch.from_numpy(x_test).float()
        value_test = torch.from_numpy(y_test).float()

        start = time.time()
        damping = 1e-2
        thr = 1e-3
        lr = 5e-7
        bestTrainCorrReg = float("inf")
        for _ in range(3):
            # 这里batchsize是rowSizeG * colSizeG = 20, 声明中称350个样本,也就是说这段代码将Intopt
            # 对于这个问题,特征4096*7000,对于这个问题,有2个金属成分,10个矿点(提供商),
            clf = Intopt(c_train, h_train,  A_data, b_data, penalty_train, damping=damping, lr=lr, n_features=featureNum, thr=thr, epochs=40)
            clf.fit(feature_train, value_train)
            train_rslt, predTrainVal = clf.val_loss(feature_train, value_train)
            avgTrainCorrReg = np.mean(train_rslt)
            trainHSD_rslt = ' train: ' + str(np.mean(train_rslt))

            if avgTrainCorrReg < bestTrainCorrReg:
                bestTrainCorrReg = avgTrainCorrReg
                torch.save(clf.model.state_dict(), 'model.pkl')
            print(trainHSD_rslt)
            
            if avgTrainCorrReg < stopCriterion:
                break


        clfBest = Intopt(c_test, h_test, A_data, b_data, penalty_test, damping=damping, lr=lr, n_features=featureNum, thr=thr, epochs=20)
        clfBest.model.load_state_dict(torch.load('model.pkl'))

        val_rslt, predTestVal = clfBest.val_loss(feature_test, value_test)
        end = time.time()
        HSD_rslt = ' test: ' + str(np.mean(val_rslt))
        print(HSD_rslt)
        print ('Elapsed time: ' + str(end-start))
        recordBest[0][testi] = np.sum(val_rslt)
        
        predTestVal = predTestVal.detach().numpy()
        predValue = np.zeros((predTestVal.size, 2))
        for i in range(predTestVal.size):
            predValue[i][0] = value_test[i]
            predValue[i][1] = predTestVal[i]
            
        np.savetxt('./Testdata/AlloyProduction/2S_weights/2S_weights' + str(penaltyTerm) + '(' + str(testi) + ').txt', predValue, fmt="%.2f")

    print(recordBest)

if __name__ == '__main__':
    main()
