import sys
import pandas as pd 
import numpy as np 

tf1 = sys.argv[1]
tf2 = sys.argv[2]
df1 = pd.read_csv(tf1)
df2 = pd.read_csv(tf2)
df1.dropna(axis=0,how='any',inplace=True)
df2.dropna(axis=0,how='any',inplace=True)
np1 = df1.to_numpy()   # 8762x15 
np2 = df2.to_numpy()   # 8806x15
#print(np1.shape, np2.shape)
## 把有不能轉成數字的標記起來
valid1 = [1 for i in range(len(np1))]
for i in range(len(np1)):
    for j in range(15):
        try:
            x = float(np1[i,j]) 
            np1[i,j] = float(np1[i,j])
        except:
            valid1[i] = 0 
            break 
valid2 = [1 for i in range(len(np2))]
for i in range(len(np2)):
    for j in range(15):
        try:
            x = float(np2[i,j])
            np2[i,j] = float(np2[i,j])
        except:
            valid2[i] = 0
            break
# 把離平均值跟標準差太遠的標記起來
count1 = 0
mean1 = np.zeros([1,15], dtype=float) 
std1 = np.zeros([1,15], dtype=float)
for i in range(len(np1)):
    if valid1[i] == 1:
        count1 += 1 
        mean1 = mean1 + np1[i]
        std1 = std1 + (np1[i]**2)
mean1 = mean1/count1
std1 = (std1/count1-(mean1**2))**(1/2)
for i in range(len(np1)):
    if valid1[i] == 1:
        count = 0
        for j in range(15):
            if (np1[i,j] < mean1[0,j]-3*std1[0,j]) or (np1[i,j] > mean1[0,j]+3*std1[0,j]):
                count += 1 
        if count > 5:
            valid1[i] = 0
count2 = 0
mean2 = np.zeros([1,15], dtype=float) 
std2 = np.zeros([1,15], dtype=float)
for i in range(len(np2)):
    if valid2[i] == 1:
        count2 += 1 
        mean2 = mean2 + np2[i]
        std2 = std2 + (np2[i]**2)
mean2 = mean2/count2
std2 = (std2/count2-(mean2**2))**(1/2)
for i in range(len(np2)):
    if valid2[i] == 1:
        count = 0
        for j in range(15):
            if (np2[i,j] < mean2[0,j]-3*std2[0,j]) or (np2[i,j] > mean2[0,j]+3*std2[0,j]):
                count += 1 
        if count > 5:
            valid2[i] = 0    
# count = 0 
# for i in range(len(np2)):
#     if valid2[i] == 1:
#         count += 1
# print(count)

x1 = []
y1 = []
for i in range(len(np1)-10):
    flag = True 
    for j in range(i,i+10):
        if valid1[j] == 0:
            flag = False
            break
    if flag == True:
        tmp = np.empty([1,15*9],dtype=float)
        tmp = np1[i:i+9,:].reshape(1,-1) 
        x1.append(tmp[0,:])
        y1.append([np1[i+9,10]])
x1 = np.array(x1) # 6262 * 135
y1 = np.array(y1)

x2 = []
y2 = []
for i in range(len(np2)-10):
    flag = True 
    for j in range(i,i+10):
        if valid2[j] == 0:
            flag = False
            break
    if flag == True:
        tmp = np.empty([1,15*9],dtype=float)
        tmp = np2[i:i+9,:].reshape(1,-1) 
        x2.append(tmp[0,:])
        y2.append([np2[i+9,10]])
x2 = np.array(x2)
y2 = np.array(y2)

x = np.concatenate((x1,x2), axis=0).astype(float)
y = np.concatenate((y1,y2), axis=0).astype(float)

n = len(x)
train_x = x[:n//10*9,:]
train_y = y[:n//10*9,:]
val_x = x[n//10*9:,:]
val_y = y[n//10*9:,:]
train_n = len(train_x)
val_n = len(val_x)

dim = 15*9+1
w = np.zeros([dim,1], dtype=float)
x = np.concatenate((train_x,np.ones([train_n,1],dtype=float)), axis=1).astype(float)
y = train_y[:,0].reshape(-1,1).astype(float)
lr = 0.01
ada = np.zeros([dim,1], dtype=float)
eps = 0.0000000001
iter_time = 1500

for i in range(iter_time):
    p_y = np.dot(x,w).astype(float) # n*1
    loss = np.sqrt(np.sum(np.power(p_y-y,2))/n) # root mean square error
    if i%100 == 0:
        print('{}: Loss is {}.'.format(i/100,loss))
    gradient = 2*np.dot(x.transpose(), p_y-y) # dim*1
    ada += gradient ** 2 
    w = w - lr * gradient / np.sqrt(ada+eps) 

x = np.concatenate((val_x,np.ones([val_n,1])), axis=1).astype(float)
y = val_y[:,0].reshape(-1,1).astype(float)
p_y = np.dot(x,w).astype(float) # n*1
loss = np.sqrt(np.sum(np.power(p_y-y,2))/val_n) # root mean square error
print('Validation Loss is: {}.'.format(loss))
np.save('weight.npy', w)


##  9/30 第四次submission
## 上傳的有把validation合併train過
