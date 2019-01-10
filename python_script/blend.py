import pandas as pd
import numpy as np


ridge = pd.read_csv('/home/shubham/Desktop/ridge_30nov.csv')
logistic = pd.read_csv('/home/shubham/Desktop/logrec.csv')
randomforest= pd.read_csv('/home/shubham/Desktop/ML_PROJECT/randomforest.csv')
SGD=pd.read_csv('/home/shubham/Desktop/SGD.csv')
xgboost= pd.read_csv('/home/shubham/Desktop/ML_PROJECT/submissionxgboost.csv')


b1 = ridge.copy()
col = ridge.columns

col = col.tolist()
col.remove('id')
for i in col:
    b1[i] = (ridge[i] * 11 + logistic[i] * 7 + randomforest[i] * 2 + xgboost[i] * 3 + SGD[i] * 5 ) /  28

b1.to_csv('/home/shubham/Desktop/blend_result.csv', index = False)
