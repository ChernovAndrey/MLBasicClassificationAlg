#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 20:02:36 2018

@author: andrey
"""
#%%
# все реализовано только для двух признаков.
import os
import numpy as np
from myUtils import readData
from create_datasets import LS, QS, path_save
import matplotlib.pyplot as plt
import math
begin_lr = 0.01 # begin learning rate
epochs = 10000
count_features = 2
eps = 1e-10
# функция стоимтости SSE
def adaline_step(x_train,y_train,w,lr): 
    
    # прямой проход и обратный проход делается по сути одновременно, но старые веса не меняются во время прохода
    cost = 0
    new_w = w.copy()
    for i,el in enumerate(x_train): 
        z = w[0] + w[1]*el[0] + w[2]*el[1]
        cost += (y_train[i] - z) ** 2
        new_w[0] -= lr*( (y_train[i] - z) * (-1) ) 
        new_w[1] -= lr*( (y_train[i] - z) * (-el[0]) ) 
        new_w[2] -= lr*( (y_train[i] - z) * (-el[1]) ) 
    cost = cost/2.0
    return cost,new_w


def sigmoid(x):
    return 1/(1+math.exp(-x))

def log_regres_step(x_train,y_train,w,lr): 
    
    # прямой проход и обратный проход делается по сути одновременно, но старые веса не меняются во время прохода
    cost = 0
    new_w = w.copy()
    for i,el in enumerate(x_train): 
        z = w[0] + w[1]*el[0] + w[2]*el[1]
        cost += -( y_train[i]*math.log(sigmoid(z)+eps) - (1-y_train[i])*math.log(1-sigmoid(z)+eps) )
        new_w[0] -= lr*( (y_train[i] - sigmoid(z)) * (-1) ) 
        new_w[1] -= lr*( (y_train[i] - sigmoid(z)) * (-el[0]) ) 
        new_w[2] -= lr*( (y_train[i] - sigmoid(z)) * (-el[1]) ) 
    cost = cost/2.0
    return cost,new_w


def merge_and_mix_train_data(data0,data1,label0,label1):
    data = np.concatenate( (data0,data1), axis = 0)
    label = np.concatenate( (label0,label1), axis = 0)
    ind = np.random.permutation(len(data))
    return data[ind], label[ind]

def train(x_train,y_train):
    w = np.random.uniform(-1,1,count_features + 1)
    lr = begin_lr
    for i in range(epochs):
        cost,w= log_regres_step(x_train,y_train,w,lr)
#        cost,w= adaline_step(x_train,y_train,w,lr)
        if (i+1)%100 == 0:
            print('step = ', i, '; cost function = ', cost)
#        if i == 200:
#            lr = 5e-5
#        if i == 9000:
#            lr = 1e-8
    return w

def evaluate(x_test,y_test,w):
    count_error = 0
    for i, el in enumerate(x_test):
        z = w[0] + w[1]*el[0] + w[2]*el[1]
        y_pred = 0
        if z >= 0.5:
            y_pred = 1
        if y_test[i] != y_pred:
            print('value pred error = ',z)
            count_error+=1
    print('count error samples = ', count_error)        
    print('accuracy = ', 1.0 - count_error/len(x_test))        


def show_results(data0,data1,w):
    plt.scatter(data0[:,0], data0[:,1], color= 'red', marker='o')
    plt.scatter(data1[:,0], data1[:,1], color='blue' , marker= 'x')
    x1_line_point = [-7,0,7] # для линии раздленеия
    x2_line_point = x1_line_point
    x2_line_point[0] = (-w[0]-w[1]*x1_line_point[0])/w[2]
    x2_line_point[1] = (-w[0]-w[1]*x1_line_point[1])/w[2]
    x2_line_point[2] = (-w[0]-w[1]*x1_line_point[2])/w[2]
    print(x2_line_point)
    plt.plot(x1_line_point, x2_line_point)
#    plt.plot([-10,10], [-25.23456990138067, 22.200020415768552])
    plt.show()


#%%
if __name__ == "__main__":
    data0 = readData(os.path.join(path_save, LS),'data0')
    data1 = readData(os.path.join(path_save, LS),'data1')
    label0 = np.full(len(data0),0) # -1 для adaline, у остальных - 0
    label1 = np.full(len(data0),1)
    data, label = merge_and_mix_train_data(data0,data1,label0,label1)        
    count_train = len(data)//2
    x_train, y_train, x_test, y_test = data[:count_train], label[:count_train], data[count_train:], label[count_train:]
    w = train(x_train,y_train)    
    evaluate(x_test,y_test,w)
    evaluate(data,label,w)
    show_results(data0,data1,w)
#%%
