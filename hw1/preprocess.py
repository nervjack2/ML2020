import numpy as np 
import pandas as pd 
from scipy import stats
from constant import *

def Preprocessing(data, train=True):
    ## delete attribute
    if train:
        data = data.drop(TRAIN_DEL_ATTR, axis=1)
    else:
        data = data.drop(TEST_DEL_ATTR, axis=1)
    ## delete rows with too many non-numeric datas
    data[data == '-'] = np.nan  
    data = data.apply(pd.to_numeric, errors='coerce')
    del_nan = data.isnull()
    count_nan = del_nan.sum(axis=1)
    del_nan = (count_nan >= DEL_NAN) 
    data[del_nan] = np.nan 
    #print('Preprocessing: Delete {} rows of non-numeric data.'.format(del_nan.sum(axis=0)))
    ## delete rows with too many zeros 
    del_zero = (data == 0.0).sum(axis=1)
    del_zero = (del_zero >= DEL_ZERO)
    data[del_zero] = np.nan 
    #print('Preprocessing: Delete {} rows which has too many zeros.'.format(del_zero.sum(axis=0)))
    ## delete rows with too many outliers 
    mean = data.mean(axis=0,skipna=True)
    std = data.std(axis=0,skipna=True)
    del_out = data.copy()
    for i,x in del_out.iterrows():
        for j in range(len(del_out.columns)):
            if std[j] != 0 and x[j]:
                x[j] = (x[j]-mean[j])/std[j]
    del_out = (del_out > 3).sum(axis=1)
    del_out = (del_out >= DEL_OUT)
    data[del_out] = np.nan          
    #print('Preprocessing: Delete {} rows which has too many outliers.'.format(del_out.sum(axis=0)))
    ## interpolate nan 
    data = data.interpolate(axis=0)

    ## give some attributes square weight 
    pm25 = data['PM2.5'].copy()
    for attr in SQUARE_ATTR:
        data[attr] = data[attr].map(lambda x: x**2)
    if train:
        return data, pm25
    else:
        return data
    
def MakeTrainData(train_data,pm25):
    n = train_data.shape[0]
    m = train_data.shape[1]
    x = np.empty([n-9,9*m], dtype=float)
    for i in range(len(train_data)-9):
        x[i,:] = train_data[i:i+9,:].reshape(1,-1)
    return x,pm25[9:,0].reshape(-1,1)

def Keyday(data,squ_day,zero_day):
    dim = len(data[0])
    k = dim//9
    data[:,dim-k*squ_day:] = data[:,dim-k*squ_day:]**2
    data[:,:k*zero_day] = 0
    return data

## 477485