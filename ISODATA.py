# -*- coding: utf-8 -*-
"""
Created on Wed May  2 20:45:17 2018

@author: th732
"""

import numpy as np


def Setparam(K, setaN, setaS, setaC, L, nIter):
    return K, setaN, setaS, setaC, L, nIter


# 给样本i分类
def BelongTo(i):
    global X, MeanTr, NTr, XClass
    temp = MeanTr - X[i] # 中间变量
    temp = np.dot(temp, temp.T)
    dist = np.diag(temp)  # dist[r]表示当前样本到目前第r个中心的距离的距离
        
    # 求与当前样本距离最小的类编号
    jdg = np.argmin(dist)
    # 将样本i加入到最近的类中
    XClass[i] = jdg + 1
    NTr[jdg] += 1
    # 更新该类的中心
    MeanTr[jdg] = (NTr[jdg] - 1)/NTr[jdg]*MeanTr[jdg]+ 1.0/NTr[jdg]*X[i]
    
    
# 给所有样本分类
def Classify():
    global X, MeanTr, NTr, XClass, setaN, C
    for i in range(X.shape[0]):
        BelongTo(i)
    # 样本数量不足的类别
    unesaryclass = np.where(NTr < setaN)
    MeanTr = np.delete(MeanTr, unesaryclass, axis=0)
    NTr = np.delete(NTr, unesaryclass)
    # 将分解后类中的元素重新分到其它类
    if unesaryclass[0].shape[0] != 0:
        for r in unesaryclass:
            for h in np.where(XClass == r + 1):
                BelongTo(h)
    C -= unesaryclass[0].shape[0]
        

# 分裂
def Divide(d, d_):
    # d[i]表示第i类所有样本到样本中心的距离， 
    # d_表示所有样本到其相应的聚类中心的距离的平均值
    global MeanTr, XClass, X, C, setaS, setaN
     
    sigmoidTr = np.zeros((C, X.shape[1]))
    for i in range(C):
        # 提取出第i类的样本矩阵xi
        xi = X[np.where(XClass  == i + 1)]
        # 计算第i类的方差向量
        temp = xi - MeanTr[i]
        temp = np.power(temp, 2)
        sigmoidTr[i] = np.sum(temp, axis=0) / xi.shape[0]
        # 分裂
        if((np.max(sigmoidTr[i]) > setaS) and (((d[i] > d_) and (xi.shape[0] > 2*setaS + 1)) or C <= K/2)):
            temp = MeanTr[i].copy()
            MeanTr = np.delete(MeanTr, i, axis=0)
            temp[np.argmax(sigmoidTr[i])] -= 0.3 * np.max(sigmoidTr[i])
            MeanTr = np.r_[MeanTr, temp.reshape(1,temp.shape[0])]
            temp[np.argmax(sigmoidTr[i])] += 2* 0.3 * np.max(sigmoidTr[i])
            MeanTr = np.r_[MeanTr, temp.reshape(1,temp.shape[0])]
            C += 1

            
# 求两类间的距离矩阵
def Distance(Xn):
    ClassDisTr = np.zeros((Xn.shape[0],Xn.shape[0]))
    n = 0
    for i in range(Xn.shape[0]):
        n += 1
        for j in range(n):
            ClassDisTr[j][i] = ClassDisTr[i][j] = np.sum(np.power(Xn[i] - Xn[j], 2))
    return ClassDisTr + np.eye(Xn.shape[0]) * 10000000

# 合并
def Combine():
    global MeanTr, X, setaC, NTr, C
    
    d = Distance(MeanTr)
    index = np.where(d < setaC)
    index = list(index)
    tempd = d[index]
    
    while(index[0].shape[0] != 0):
        i = index[0][np.argmin(tempd)]
        j = index[1][np.argmin(tempd)]
        temp = (NTr[i] * MeanTr[i] + NTr[j] * MeanTr[j]) / (NTr[i] + NTr[j])
        MeanTr = np.delete(MeanTr, np.array([i, j]), axis=0)
        MeanTr = np.r_[MeanTr, temp.reshape(1, temp.shape[0])]
        # 更新
        d = Distance(MeanTr)
        index = np.where(d < setaC)
        index = list(index)
        tempd = d[index]
        # 由于每个类只能合并一次，故需要删除已经合并过的类
        r = 0
        while(r < tempd.shape[0]):
            if(index[0][r] == i or index[0][r] == j or index[1][r] == i or index[1][r] == j):
                index[0] = np.delete(index[0], r)
                index[1] = np.delete(index[1], r)
                tempd = np.delete(tempd, r)
            r += 1
        C -= 1
        

def IsoDataMain(X, K, setaN, setaS, setaC, L, nIter):
    global XClass, MeanTr, NTr
    
    C = min(K, X.shape[0])
    # MeanTr[i]表示第i类的聚类中心
    MeanTr = X[:C,:]
    MeanTr = MeanTr.astype("float64")
    # Ntr[i]代表第i类的样本数
    NTr = np.full((C,), 1)
    # XClass[i]表示第i个样本属于的类别
    XClass = np.full((X.shape[0],),0)
    
    # 迭代
    for i in range(nIter):
        # 重新更新NTr和XClass
        NTr = np.full((C,), 1)
        XClass = np.full((X.shape[0],), 0)
        
        Classify()
        
        d = np.zeros(C,)
        d_ = 0
        for r in range(C):
            indeXr = np.where(XClass == r + 1)
            Xr = X[indeXr]
            temp = Xr - MeanTr[r]
            temp = np.dot(temp, temp.T)
            d[r] = np.sum(np.diag(temp)) / Xr.shape[0]
            d_ += d[r] * Xr.shape[0]
        d_ /= X.shape[0]
        if(C <= K/2):
            Divide(d, d_)
        elif(C < 2*K and (i // 2)):
            Divide(d, d_)
        else:
            pass
            
    Combine()
    return XClass

# 通过最后的XClass得到样本类别分组
def GetClassGroup(XClass):
    Class = []
    for i in range(np.max(XClass)):
        temp = np.where(XClass == i + 1)[0]
        Class.append(temp.tolist())
    return Class

# main
if __name__ == "__main__":
    
    X = np.array([[0,0],[3,8],[2,2],
                  [1,1],[5,3],[4,8],
                  [6,3],[5,4],[6,4],
                  [7,5]])
    # 参数设置
    XClass = IsoDataMain(X, 4, 1, 1, 4, 1, 4)
    print("分类结果(输出的第i个数字代表第i个样本所属的类别编号)：")
    print(GetClassGroup(XClass))
    
    
    