import numpy as np
import random


from datadeal import *
from GAT import *
from utils import *
from sklearn.metrics import roc_curve
import pandas as pd
from RF import *
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

A=np.loadtxt("./data/MDAD/drug_microbe_matrix.txt")  #邻接矩阵
A1=np.loadtxt("./data/MDAD/drug_dmicrobe_matrix.txt") #用drug-microbe-disease填补了的关联矩阵
known=np.loadtxt("./data/MDAD/known.txt") #已知关联索引
unknown=np.loadtxt("./data/MDAD/unknown.txt")
known1=np.loadtxt("./data/MDAD/known1.txt")
unknown1=np.loadtxt("./data/MDAD/unknown1.txt")
Smf=np.loadtxt("./data/MDAD/microbe_function_sim.txt")
Src=np.loadtxt("./data/MDAD/drug_structure_sim.txt")
dd=pd.read_excel("./data/MDAD/drug_drug_interactions.xlsx")
dd=dd.values
mm=pd.read_excel("./data/MDAD/microbe_microbe_interactions.xlsx")
mm=mm.values

def kflod_5(known,unknown,A):
    scores = []
    tlabels = []
    k = []
    unk = []
    b=np.zeros((1,2))
    lk = len(known)  # 已知关联数
    luk = len(unknown)  # 未知关联数
    for j in range(lk):
        k.append(j)
    for j in range(luk):
        unk.append(j)
    random.shuffle(k)  # 打乱顺序
    random.shuffle(unk)
    for cv in range(1, 6):
        interaction = np.array(list(A))
        if cv < 5:
            B1 = known[k[(cv - 1) * (lk // 5):(lk // 5) * cv], :]  # 1/5的1的索引
            B2 = unknown[unk[(cv - 1) * (luk // 5):(luk // 5) * cv], :]  # 1/5的0的索引
            for i in range(lk // 5):
                interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
        else:
            B1 = known[k[(cv - 1) * (lk // 5):lk], :]
            B2 = unknown[unk[(cv - 1) * (luk // 5):luk], :]
            for i in range(lk - (lk // 5) * 4):
                interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
        B=np.vstack((B1,B2))
        b=np.vstack((b,B))
        Srg=GIP_Calculate1(interaction)
        Smg=GIP_Calculate(interaction)
        Srh=HIP_Calculate(interaction)
        Smh=HIP_Calculate(interaction.T)
        Sr=(Srg+Srh)/2
        Sm=(Smg+Smh)/2
        for i in range(len(dd)):
            Sr[int(dd[i][0])-1][int(dd[i][2])-1]=1
        for i in range(len(mm)):
            Sm[int(mm[i][0])-1][int(mm[i][2])-1]=1
        N1 = np.hstack((Sr, interaction))
        N2 = np.hstack((interaction.T, Sm))
        Net = np.vstack((N1, N2))  # 异构网络1
        Srr1 = RWR(Sr)
        Smm1 = RWR(Sm)
        Srr1 = np.array(Srr1)
        Smm1 = np.array(Smm1)
        return Srr1.shape,Smm1.shape
c,b=kflod_5(known, unknown, A)
print(c,b)