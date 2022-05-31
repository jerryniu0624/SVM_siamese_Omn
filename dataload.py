import pandas as pd
import scipy
from scipy import io
# 文件路径
features_struct = scipy.io.loadmat(r'C:\\Users\\Jerry\\Desktop\\junior2\\PRML\\SVM\\data2\\Dataset.mat')
# matlab变量名
for i in range(200):
    for j in range(15):
        features_train = features_struct['train'][i,j,:,:]
        # print(features_struct['train'][i,j,:,:].shape)
        dfdata_train = pd.DataFrame(features_train)
        datapath1 = r'C:\\Users\\Jerry\\Desktop\\junior2\\PRML\\SVM\\dataset_train\\Dataset_train.csv'
        dfdata_train.to_csv(datapath1, index=False)
# 存储路径


for i in range(200):
    for j in range(5):
        features_test = features_struct['test'][i,j,:,:]
        dfdata_test = pd.DataFrame(features_test)
        datapath2 = r'C:\\Users\\Jerry\\Desktop\\junior2\\PRML\\SVM\\dataset_test\\Dataset_test.csv'
        dfdata_test.to_csv(datapath2, index=False)
# 存储路径

