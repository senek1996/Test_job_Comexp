# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:22:33 2022

@author: a_bredihin
"""
import numpy as np

#работа с метками
with open('train-labels.idx1-ubyte','rb') as f:
    l = np.array(list(f.read()))

l = l[8:]

#работа с изображениями
with open('train-images-idx3-ubyte-32x32','rb') as f:
    s = np.array(list(f.read()))

s = s[16:]

data = np.zeros((60000,32,32), dtype=np.uint8)
start=0
step=1024

for i in range(60000):
    im = s[start:start+step]
    im = im.reshape((32,32))
    data[i,:,:] = im
    start += step
    


CNN_l1_filt = [(5,5),(7,7)]
CNN_l1_krnls = [2,4]
CNN_l2_filt = [(3,3),(5,5)]
CNN_l2_krnls = [4,6]
CNN_l3_filt = [(1,1),(3,3)]
CNN_l3_krnls = [10]

pool_opts = ['avg','max']

MLP_hid_sze = [80,40]

#таблица со всем результатами
res_NN = {
    'CNN_l1_filt': [],
    'CNN_l1_krnls': [],
    'CNN_l2_filt': [],
    'CNN_l2_krnls': [],
    'CNN_l3_filt': [],
    'CNN_l3_krnls': [],
    'pool_opt': [],
    'MLP_hid_sze': [],
    'F1_score': [],
    'CNN_model': []
}



import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize


inds_all = np.arange(60000)
inds_test = np.random.choice(inds_all, size=10000)

inds_all = set(inds_all)
inds_test = set(inds_test)
inds_train = inds_all - inds_test

print(len(inds_all), len(inds_test), len(inds_train))

inds_all = list(inds_all)
inds_test = list(inds_test)
inds_train = list(inds_train)

X_train = data[inds_train,:,:]/255
X_test = data[inds_test,:,:]/255
y_train = l[inds_train]
y_test = l[inds_test]
y_train_vec = label_binarize(y_train,classes = [x for x in range(10)])
y_test_vec = label_binarize(y_test,classes = [x for x in range(10)])

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)



i = 1
max_F1 = 0

for l1_f in CNN_l1_filt:
    for l1_k in CNN_l1_krnls:
        for l2_f in CNN_l2_filt:
            for l2_k in CNN_l2_krnls:
                for l3_f in CNN_l3_filt:
                    for l3_k in CNN_l3_krnls:
                        for p_opt in pool_opts:
                            for MLP_hid in MLP_hid_sze:
                                
                                print('Сеть №{}...'.format(i))
                                model = Sequential()
                                model.add(Conv2D(l1_k, l1_f, activation='relu', input_shape=(32,32,1))) #слой 1 свертки
                                
                                if p_opt=='avg':
                                    model.add(AveragePooling2D(pool_size=(2, 2)))
                                else:
                                    model.add(MaxPooling2D(pool_size=(2, 2)))
                                
                                model.add(Conv2D(l2_k, l2_f, activation='relu')) #слой 2 свертки
                                
                                if p_opt=='avg':
                                    model.add(AveragePooling2D(pool_size=(2, 2)))
                                else:
                                    model.add(MaxPooling2D(pool_size=(2, 2)))
                                    
                                model.add(Conv2D(l3_k, l3_f, activation='relu')) #слой 3 свертки
                                
                                if p_opt=='avg':
                                    model.add(AveragePooling2D(pool_size=(2, 2)))
                                else:
                                    model.add(MaxPooling2D(pool_size=(2, 2)))
                                
                                model.add(Flatten())
                                model.add(Dense(MLP_hid, activation='relu'))
                                model.add(Dense(10, activation='softmax'))
                                model.compile(optimizer='sgd', loss='categorical_crossentropy')
                                
                                model.fit(X_train, y_train_vec, batch_size=200, epochs=50)
                                
                                y_pred = model.predict(X_test)
                                y_pred = y_pred.argmax(axis=1)
                                F1 = f1_score(y_test,y_pred,average='micro')
                                
                                res_NN['CNN_l1_filt'].append(l1_f)
                                res_NN['CNN_l1_krnls'].append(l1_k)
                                res_NN['CNN_l2_filt'].append(l2_f)
                                res_NN['CNN_l2_krnls'].append(l2_k)
                                res_NN['CNN_l3_filt'].append(l3_f)
                                res_NN['CNN_l3_krnls'].append(l3_k)
                                res_NN['pool_opt'].append(p_opt)
                                res_NN['MLP_hid_sze'].append(MLP_hid)
                                res_NN['F1_score'].append(F1)
                                
                                if F1>max_F1: #сохраняем не все модели
                                    max_F1 = F1
                                    res_NN['CNN_model'].append(model)
                                else:
                                    res_NN['CNN_model'].append('')