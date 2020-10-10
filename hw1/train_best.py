import sys 
import numpy as np 
import pandas as pd 
from preprocess import *
from constant import *

train1 = sys.argv[1]
train2 = sys.argv[2]
df1 = pd.read_csv(train1)
df2 = pd.read_csv(train2)
train_data = pd.concat([df1,df2], ignore_index=True)
train_data, pm25 = Preprocessing(train_data)
train_data, pm25 = train_data.to_numpy(), pm25.to_numpy().reshape(-1,1)
x,y = MakeTrainData(train_data,pm25)
if KEYDAY:
    x = Keyday(x,squ_day=SQU_DAYS,zero_day=ZERO_DAYS)

n = len(x)
dim = len(x[0])+1
x = np.concatenate((x,np.ones([n,1], dtype=float)),axis=1)
w = np.zeros([dim,1],dtype=float) 
ada = np.zeros([dim,1],dtype=float)
if VAL:
    train_x = x[:n//10*9,:]
    train_y = y[:n//10*9,0].reshape(-1,1)
    val_x = x[n//10*9:,:]
    val_y = y[n//10*9:,:].reshape(-1,1)
    train_n = len(train_x)
    val_n = len(val_x)
else:
    train_x = x[:,:]
    train_y = y[:,0].reshape(-1,1)
    train_n = len(train_x)

for i in range(iter_time):
    p_y = np.dot(train_x,w).astype(float) # n*1
    loss = np.sqrt(np.sum(np.power(p_y-train_y,2))/train_n) # root mean square error
    if i%100 == 0:
        print('{}: Loss is {}.'.format(i/100,loss))
    gradient = 2*np.dot(train_x.transpose(), p_y-train_y) # dim*1
    ada += gradient ** 2 
    w = w - lr * gradient / np.sqrt(ada+eps) 

if VAL:
    p_y = np.dot(val_x,w).astype(float) # n*1
    loss = np.sqrt(np.sum(np.power(p_y-val_y,2))/val_n) # root mean square error
    print('Validation Loss is: {}.'.format(loss))
np.save('./model/weight.npy', w)

## 477485