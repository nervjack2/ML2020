
import numpy as np 
import sys
import pandas as pd

w = np.load('./model/hw1.npy')
input_file = sys.argv[1]
output_file = sys.argv[2]
test_data = pd.read_csv(input_file) 
test_data = test_data.to_numpy().astype(float)

res = np.empty([500,15*9], dtype=float)
for i in range(500):
    res[i,:] = test_data[9*i:9*(i+1),:].reshape(1,-1)

res = np.concatenate((res,np.ones([500,1])), axis=1).astype(float)

with open(output_file, 'w') as fp:
    fp.write('id,value\n')
    ans = np.dot(res, w).astype(float)
    for i in range(500):
        fp.write('id_{},{}\n'.format(i,ans[i][0]))
    
