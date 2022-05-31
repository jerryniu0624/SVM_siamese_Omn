import scipy.io as io
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import torchvision
import torchvision.transforms as transforms
import torch
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split as ts
import numpy as np

if __name__ == '__main__':
# import cv2
# 导入数据
    data=io.loadmat('./data2/Dataset.mat')
    # transform = transforms.Compose(
    #     [transforms.Resize([227,227]),
    #     transforms.ToTensor(),
    #     transforms.RandomCrop(224,padding = 2 ,pad_if_needed = True,fill = 0,padding_mode ='constant'),
    #     transforms.RandomHorizontalFlip(p=0.5), # 表示进行左右的翻转
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = torchvision.datasets.ImageFolder(root='./dataset_train',transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            #   shuffle=True, num_workers=2)

    # for data in enumerate(trainloader, 0):
    #     inputs, labels = data
        # print(inputs)
        # print("-----------------------")
        # print(labels)

    print(data.keys())
    print(data['train'].shape)
    print(data['test'].shape)
# plt.imshow(data['train'][0][0])
# 200类，每类15张训练集，5张测试集，图片size为28*28

# 把28*28 的矩阵resize为 784的一维向量作为SVM的输入

    data_train=data['train']
    X_train=[]
    for i in data_train:
        for j in i:
        # print(j)
            j=j.flatten()
        # print(j)
            X_train.append(j)
    y_train=[]
    for i in range(0,200):
        for j in range(0,15):
            y_train.append(i)

    data_test=data['test']
    X_test=[]
    for i in data_test:
        for j in i:
        # print(j.shape)
            j=j.flatten()
        # print(j.shape)
            X_test.append(j)
    y_test=[]
    for i in range(0,200):
        for j in range(0,5):
            y_test.append(i)

    X_train=np.array(X_train)
    X_test=np.array(X_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)



# trainset = transform(X_train)

    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# kernel = 'rbf'
    # list1 = np.arange(0.0001,0.001,0.0001) #C_best=3.1 0.426
    list2 = np.arange(2.5,3.5,0.1) #C_best=3.1 0.426
    # for gamma1 in list1:
    for Ci in list2:
        clf_rbf = svm.SVC(kernel='rbf', C=Ci)
        clf_rbf.fit(X_train,y_train)
        score_rbf = clf_rbf.score(X_test,y_test)
        print("The score of C= %f rbf  is : %f"%(Ci,score_rbf))

# # kernel = 'linear'
#     list2 = np.arange(2.5,3.5,0.1) 
#     for Ci in list2:
#         clf_linear = svm.SVC(kernel='linear',C=Ci)
#         clf_linear.fit(X_train,y_train)
#         score_linear = clf_linear.score(X_test,y_test)
#         print("The score of linear is : %f"%score_linear)

# kernel = 'poly'
    # list1 = np.arange(0,10,1) #C_best=3.1 0.426
    # list2 = np.arange(2.5,3.5,0.1) #C_best=3.1 0.426
    # for c1 in list1:#The score of degree= 3.000000 C=3.400000 poly is : 0.402000
    #     # for Ci in list2:
    #     clf_poly = svm.SVC(kernel='poly',C=3,coef0=c1)
    #     clf_poly.fit(X_train,y_train)
    #     score_poly = clf_poly.score(X_test,y_test)
    #     print("The score of coef0= %f C=3 poly is : %f"%(c1,score_poly))

# # kernel = 'sigmoid'
#     clf_rbf = svm.SVC(kernel='sigmoid')
#     clf_rbf.fit(X_train,y_train)
#     score_rbf = clf_rbf.score(X_test,y_test)
#     print("The score of sigmoid is : %f"%score_rbf)

# # kernel = 'precomputed'
# clf_rbf = svm.SVC(kernel='precomputed')
# clf_rbf.fit(X_train,y_train)
# score_rbf = clf_rbf.score(X_test,y_test)
# print("The score of precomputed is : %f"%score_rbf)