#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 18:32:15 2018

@author: andrey
"""
#%%
#метки ноль и один; генерируем всего два признака.
import numpy as np
import matplotlib.pyplot as plt
from myUtils import saveData
import os
import math
count_data = 100
max_val = 10 # максимальное значение признаков по модулю.
count_features = 2
path_save = '/home/andrey/datasetsNN/MLBasicClassificationAlg'
LS = 'linear_separable.hdf5' # константа для сохранения.
QS = 'quadr_separable.hdf5' # константа для сохранения.
def create_linear_separable_data(): 
     k,b = np.random.uniform(-max_val//2,max_val//2,2) # генерация линии разделения      
     cur_count_zero_class = 0
     cur_count_first_class = 0
     data0 = np.empty( (count_data//2, count_features) ) # данные для нулевого класса
     data1 = np.empty( (count_data//2, count_features) ) # данные для первого класса
  
     while (cur_count_first_class+cur_count_zero_class<count_data):
        x1,x2 = np.random.uniform(-max_val,max_val,2)        
        if ( (k*x1+b <= x2) and (cur_count_zero_class<count_data//2) ):
            data0[cur_count_zero_class] = np.array([x1,x2])
            cur_count_zero_class += 1
        if ( (k*x1+b > x2) and (cur_count_first_class<count_data//2) ):
            data1[cur_count_first_class] = np.array([x1,x2])
            cur_count_first_class += 1
     return data0,data1
 
#Первый класс будет лежать внутри окружности, второй за ней.(Для удобства центр оркужности в нуле.)
def create_quadr_separable_data(): # квадратично разделимые. 
     r  = np.random.uniform(max_val//5,max_val//2,1) # генерация радиуса окружности
     cur_count_zero_class = 0
     cur_count_first_class = 0
     data0 = np.empty( (count_data//2, count_features) ) # данные для нулевого класса
     data1 = np.empty( (count_data//2, count_features) ) # данные для первого класса
  
     while (cur_count_first_class+cur_count_zero_class<count_data):
        x1,x2 = np.random.uniform(-max_val,max_val,2)        
        if ( ( math.sqrt(x1*x1 + x2*x2) <= r ) and (cur_count_zero_class<count_data//2) ):
            data0[cur_count_zero_class] = np.array([x1,x2])
            cur_count_zero_class += 1
        if ( ( math.sqrt(x1*x1 + x2*x2) > r ) and (cur_count_first_class<count_data//2) ):
            data1[cur_count_first_class] = np.array([x1,x2])
            cur_count_first_class += 1
     return data0,data1


def show_data(data0,data1):
    plt.scatter(data0[:,0], data0[:,1], color= 'red', marker='o')
    plt.scatter(data1[:,0], data1[:,1], color='blue' , marker= 'x')
    plt.show()
            

def execute():
    data0,data1 = create_linear_separable_data()
#    data0,data1 = create_quadr_separable_data()
    show_data(data0,data1)
#    saveData(os.path.join(path_save, LS),'data0',data0)
#    saveData(os.path.join(path_save, LS),'data1',data1)
#%%
if __name__ == "__main__":
    execute()