import pandas as pd
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn import cross_validation
import xgboost as xgb
import time
def loaddata(Filename):
    data = pd.read_csv(Filename,sep=',',header = None)
    return np.array(data)
# dataset
data1 = loaddata("data0_1.csv")
data2 = loaddata("data1_1.csv")
data3 = loaddata("data2_1.csv")
data4 = loaddata("data3_1.csv")
data5 = loaddata("data4_1.csv")

data_train = np.vstack((data1[:len(data1)-1],data2[:len(data1)]))
data_train = np.vstack((data_train,data3[:len(data1)]))
data_train = np.vstack((data_train,data4[:len(data1)]))
data_train = np.vstack((data_train,data5[:len(data1)]))

print('This is data_train',type(data_train),data_train.shape)
#label
data1 = loaddata("label0_1.csv")
data2 = loaddata("label1_1.csv")
data3 = loaddata("label2_1.csv")
data4 = loaddata("label3_1.csv")
data5 = loaddata("label4_1.csv")

label_train = np.vstack((data1[:len(data1)-1],data2[:len(data1)]))
label_train = np.vstack((label_train,data3[:len(data1)]))
label_train = np.vstack((label_train,data4[:len(data1)]))
label_train = np.vstack((label_train,data5[:len(data1)]))
#print(label_test[100:800])
train_X,test_X,train_Y,test_Y=cross_validation.train_test_split(data_train,label_train,test_size=0.1)

print('This is test_X',type(test_X),test_X.shape)
print('This is test_Y',type(test_Y),test_Y.shape)
print('This cell has done...')

################################################
xg_train = xgb.DMatrix( train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.3
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 10
param['num_class'] = 5

watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 5
bst = xgb.train(param, xg_train, num_round, watchlist );
# get prediction
pred = bst.predict( xg_test );
print(pred)

print ('predicting, classification error=%f' 
       % (sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))

# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist );
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
yprob = bst.predict( xg_test ).reshape( test_Y.shape[0], 5 )
print(yprob)
ylabel = np.argmax(yprob, axis=1)  # return the index of the biggest pro
print ('predicting, classification error=%f' 
       % (sum( int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
