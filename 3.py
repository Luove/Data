# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 20:13:46 2018

@author: Luove
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datetime import date
# from apriori import *
import time  # 计算耗时
import os

os.chdir('D:/Analyze/Python Matlab/Python/BookCodes/Python数据分析与挖掘实战/图书配套数据、代码/chapter8/demo/code')

input = '../data/data.xls'
data = pd.read_excel(input)

# from __future__ import print_function
typelabel = {u'肝气郁结证型系数': 'A', u'热毒蕴结证型系数': 'B', u'冲任失调证型系数': 'C', u'气血两虚证型系数': 'D', u'脾胃虚弱证型系数': 'E', u'肝肾阴虚证型系数': 'F'}
k = 4

keys = list(typelabel.keys())
values = list(typelabel.values())
result = pd.DataFrame()

# if __name__ == '__main__':  #作为模块导入不运行此代码块，作为函数运行则运行代码块（此时__name__等于__main__）
for i in range(len(keys)):
    print(u'Hi~Man 我正在进行 "%s"的聚类...' % keys[i])
    kmodel = KMeans(n_clusters=k, n_jobs=4)
    kmodel.fit(data[[keys[i]]].as_matrix())
    #         r1=pd.DataFrame(kmodel.cluster_centers_,columns=[typelabel[keys[i]]])
    r1 = pd.DataFrame(kmodel.cluster_centers_, columns=[values[i]])  # 聚类中心
    r2 = pd.Series(kmodel.labels_).value_counts()  # 各类含样本量
    r2 = pd.DataFrame(r2, columns=[values[i] + 'n'])  # 转DataFrame 且修改列名
    r = pd.concat([r1, r2], axis=1).sort_values(values[i])
    r.index = [1, 2, 3, 4]
    r[values[i]] = r[values[i]].rolling(2).mean()  # 滚动计算两个聚类中心均值
    r[values[i]][1] = 0.0  # 将第一个空位补上0
    result = result.append(r.T)
result = result.sort_index()

data_ = data[[keys[i] for i in range(len(keys))]]  # 提取要建模的各证型
# 选择去掉 r2，方便处理
# result_=result[0:len(result):2]
# result_count=result[1:len(result):2]
result_ = result.iloc[::2, :]
result_count = result.iloc[1::2, :]
# data_.iloc[:,1]


# 聚类结果 指标值A1、A2、。。。
strabc = pd.DataFrame()
for i in range(len(keys)):
    strabcd = [values[i] + '%s' % (j + 1) for j in range(k)]
    strabcd = pd.DataFrame(strabcd, columns=[values[i]])  # columns=[values[i]],columns须是list，要转化加[],[values[]]
    strabc = strabc.append(strabcd.T)
''' strabc
    0   1   2   3
A  A1  A2  A3  A4
B  B1  B2  B3  B4
C  C1  C2  C3  C4
D  D1  D2  D3  D4
E  E1  E2  E3  E4
F  F1  F2  F3  F4
'''
''' result_ 
     1         2         3         4
A  0.0  0.167995  0.246969  0.339837
B  0.0  0.153543  0.298217  0.489954
C  0.0  0.202149  0.289061  0.423537
D  0.0  0.172049  0.251583  0.359353
E  0.0  0.153398  0.258200  0.376062
F  0.0  0.179143  0.261386  0.354643  
'''
# 将数值转化为类别指标,用到的数据
data_.shape  # (930, 6)
result_.shape  # (6, 4)
strabc.shape  # (6, 4)
len(keys)  # 6

# 转换值到指标,为避免潜在错误，新建一个的DataFrame接收转换后的指标矩阵
# data_new=pd.DataFrame(columns=[keys[i]+'new' for i in range(data_.shape[1])])
data_new = pd.DataFrame()
# i=0,1,2,3,4,5,6个，从result_行取比较值，和data_/data列值比较，确定值在strabc中找
# j=0,1,2,3，4个，A类中比较时result_行固定i
for i in range(len(result_)):
    index1 = data[keys[i]] < result_.iloc[i, 1]
    index2 = (result_.iloc[i, 1] < data[keys[i]]) & (data[keys[i]] < result_.iloc[i, 2])
    index3 = (result_.iloc[i, 2] < data[keys[i]]) & (data[keys[i]] < result_.iloc[i, 3])
    index4 = result_.iloc[i, 3] < data[keys[i]]
    # index0=pd.DataFrame()

    # len(index1)
    data_n = index1.copy()  # 仅为生成data_n
    data_n[index1 == True] = strabc.iloc[i, 0]
    data_n[index2 == True] = strabc.iloc[i, 1]
    data_n[index3 == True] = strabc.iloc[i, 2]
    data_n[index4 == True] = strabc.iloc[i, 3]
    # set(data_new)
    data_new = pd.concat([data_new, data_n], axis=1)
'''
更好的实现划分方法
pd.cut
'''
# data_.iloc[:,data_.shape[1]-1] #最后一列
# data_['n']=0
# del(data_['n'])#删去列名为'n'的列
# del(data_[data_.columns[data_.shape[1]-1]])#删去最后一列

# 至此，由值转指标类别工作完毕，下面开始建模apriori
data_new.head(5)

data_model = pd.concat([data_new, data['TNM分期']], axis=1)

start = time.clock()
print(u'\n转换原始矩阵至0-1矩阵...')
'''
b是ct函数作用到data_model.as_matrix() (930, 7) 矩阵的结果
按行展开排成一序列，作为新建Series的index 对应values为1
ct函数就是将x按行排成序列作为index 将x值作为index 

'''
ct = lambda x: pd.Series(1, index=x[pd.notnull(x)])
b = map(ct, data_model.as_matrix())  # 依次作用于matrix的每行
c = list(b)  # 必须转成可迭代的list
# len(c)    #930
# len(c[0]) #7
data_model_ = pd.DataFrame(c).fillna(0)  # 将c list转化为 DataFrame，所有list的index作为列，每行对应index对应值是对应list[i]
# DataFrame 每行对应原一条记录中各证型的出现与否，是就是1，否为0
type(data_model_)
data_model_.shape  # (930, 28)
# 每行是个事务集
end = time.clock()
print(u'\n耗时%0.2f ' % (end - start))
# del b


support = 0.06
confidence = 0.75
ms = '--'
start = time.clock()
print(u'\n开始搜索关联规则...')


# 自定义连接函数，用于实现L_{k-1}到C_k的连接
def connect_string(x, ms):
    x = list(map(lambda i: sorted(i.split(ms)), x))
    l = len(x[0])
    r = []
    # 生成二项集
    for i in range(len(x)):
        for j in range(i, len(x)):
            #      if x[i][l-1] != x[j][l-1]:
            if x[i][:l - 1] == x[j][:l - 1] and x[i][l - 1] != x[j][
                l - 1]:  # 判断数字和字母异同，初次取字母数字不全相同（即不同症状（字母），或同一证型程度不同（数字））
                r.append(x[i][:l - 1] + sorted([x[j][l - 1], x[i][l - 1]]))
    return r


# 寻找关联规则的函数
def find_rule(d, support, confidence, ms=u'--'):
    result = pd.DataFrame(index=['support', 'confidence'])  # 定义输出结果

    support_series = 1.0 * d.sum() / len(d)  # 支持度序列
    column = list(support_series[support_series > support].index)  # 初步根据支持度筛选,符合条件支持度，共 276个index证型
    k = 0

    while len(column) > 1:  # 随着项集元素增多 可计算的column（满足条件支持度的index）会被穷尽，随着证型增多，之间的关系会越来越不明显，（同时发生可能性是小概率了）
        k = k + 1
        print(u'\n正在进行第%s次搜索...' % k)
        column = connect_string(column, ms)
        print(u'数目：%s...' % len(column))
        sf = lambda i: d[i].prod(axis=1, numeric_only=True)  # 新一批支持度的计算函数
        len(d)
        # 创建连接数据，这一步耗时、耗内存最严重。当数据集较大时，可以考虑并行运算优化。
        # 依次对column每个元素（如[['A1', 'A2'], ['A1', 'A3']]中的['A1', 'A2']）运算，计算data_model_中对应该行的乘积，930个，若['A1', 'A2']二者同时发生为1则此行积为1
        d_2 = pd.DataFrame(list(map(sf, column)),
                           index=[ms.join(i) for i in column]).T  # list(map(sf,column)) 276X930  index 276

        support_series_2 = 1.0 * d_2[[ms.join(i) for i in column]].sum() / len(d)  # 计算连接后的支持度
        column = list(support_series_2[support_series_2 > support].index)  # 新一轮支持度筛选
        support_series = support_series.append(support_series_2)
        column2 = []

        for i in column:  # 遍历可能的推理，如{A,B,C}究竟是A+B-->C还是B+C-->A还是C+A-->B？
            i = i.split(ms)  # 由'A1--B1' 转化为 ['A1', 'B1']
            for j in range(len(i)):  #
                column2.append(i[:j] + i[j + 1:] + i[j:j + 1])

        cofidence_series = pd.Series(index=[ms.join(i) for i in column2])  # 定义置信度序列

        for i in column2:  # 计算置信度序列  如i为['B1', 'A1']
            # i置信度计算：i的支持度除以第一个证型的支持度，表示第一个发生第二个发生的概率
            cofidence_series[ms.join(i)] = support_series[ms.join(sorted(i))] / support_series[ms.join(i[:len(i) - 1])]

        for i in cofidence_series[cofidence_series > confidence].index:  # 置信度筛选
            result[i] = 0.0  # B1--A1    0.330409  A1--B1    0.470833,绝大部分是要剔除掉的，初次全剔除
            result[i]['confidence'] = cofidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]

    result = result.T.sort_values(by=['confidence', 'support'],
                                  ascending=False)  # 结果整理，输出,先按confidence升序，再在confidence内部按support升序，默认升序，此处降序


'''举个虚假的例子
            F2--H2
support          0
confidence       0
   转化为
        support  confidence
F2--H2        0           0
'''

print(u'\n结果为：')
print(result)

return result

find_rule(data_model_, support, confidence, ms)
end = time.clock()
print(u'\搜索完毕，耗时%0.2f秒' % (end - start))
