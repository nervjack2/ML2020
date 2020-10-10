
import numpy as np 
import sys
import pandas as pd
from preprocess import *

w = np.load('./model/hw1_best.npy')
input_file = sys.argv[1]
output_file = sys.argv[2]
test_data = pd.read_csv(input_file) 
test_data = Preprocessing(test_data,train=False).to_numpy()
dim = 9*test_data.shape[1]
x = np.empty([500,dim],dtype=float)
for i in range(500):
    x[i,:] = test_data[i*9:(i+1)*9,:].reshape(1,-1)
if KEYDAY:
    x = Keyday(x,squ_day=SQU_DAYS,zero_day=ZERO_DAYS)
x = np.concatenate((x,np.ones([500,1],dtype=float)),axis=1)
with open(output_file,'w') as fp:
    fp.write('id,value\n')
    y = np.dot(x,w)
    for i in range(500):
        fp.write('id_{},{}\n'.format(i,y[i,0]))

## 477485
