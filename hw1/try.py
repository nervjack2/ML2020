import pandas as pd 

x = pd.DataFrame({'A':[1,2,3,4],'B':[5,6,7,8]})
for i,row in x.iterrows():
    row[0] = 2 
print(len(x.columns))