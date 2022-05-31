import torch
from sklearn import svm
from dataset import get_loaders
from model import SiameseSVMNet
from svmloss import SVMLoss, compute_accuracy
import random
from torch.autograd import Variable
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
import scipy.io as io

# constants
dim = 105


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese kernel SVM')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--C', type=float, default=0.2, metavar='C',
                        help='C regulation coefficients (default: 0.2)')
    parser.add_argument('--test-number', type=int, default=10, metavar='N',
                        help='number of different subset of test (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def main():
    args = get_args()
    model = SiameseSVMNet()
    if args.cuda:
        model = model.cuda()
    criterion = SVMLoss(args.C)
    optimizer = torch.optim.Adam(model.parameters())
    train_loader, validate_loader , test_data = get_loaders(args)

    def training(epoch):
        print('Epoch', epoch + 1)
        model.train()
        for batch_idx, (x0, x1, label) in enumerate(train_loader):
            if args.cuda:
                x0, x1, label = x0.cuda(), x1.cuda(), label.cuda()
            x0, x1, label = Variable(x0), Variable(x1), Variable(label)
            optimizer.zero_grad()
            output = model(x0, x1)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print("\n Batch:  ", batch_idx, " / ",
                      len(train_loader), " --- Loss: ", loss.data[0])

    def validate():
        model.eval()
        acc = 0
        for batch_idx, (x0, x1, label) in enumerate(validate_loader):
            if args.cuda:
                x0, x1, label = x0.cuda(), x1.cuda(), label.cuda()
            x0, x1, label = Variable(x0), Variable(x1), Variable(label)
            output = model(x0, x1)
            acc += compute_accuracy(output, label).numpy()[0]

        acc = 100.0 * acc / len(validate_loader.dataset)
        print('\nValidation set: Accuracy: {}%\n'.format(acc))
        return acc

    def test(n, k):
        model.eval()
        clf = svm.SVC(C=args.C, kernel='linear')
        featuremodel = model.get_FeatureNet()
        if args.cuda:
            featuremodel = featuremodel.cuda()
        # choose classes
        acc = 0
        for i in range(args.test_number):
            random.seed(i)
            temp_ = []
            for i in range(1623 - 1200):
                temp_.append(i)
            random.shuffle(temp_)
            choosen_classes = temp_[:n]

            data=io.loadmat('./data2/Dataset.mat')

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

            # calculate features
            train_features = []
            test_features = []
            for train_point in X_train:
                train_features.append(featuremodel(
                    Variable(train_point)).cpu().data.numpy())
            for test_point in X_test:
                test_features.append(featuremodel(
                    Variable(test_point)).cpu().data.numpy())

            # create features
            train_features = np.array(train_features)
            train_features = np.reshape(
                train_features, (train_features.shape[0], 4096))
            test_features = np.array(test_features)
            test_features = np.reshape(
                test_features, (test_features.shape[0], 4096))

            # predict with SVM
            clf.fit(train_features, y_train)
            pred = clf.predict(test_features)
            acc += accuracy_score(y_test, pred)

        acc = 100.0 * acc / args.test_number
        print('\nTest set: {} way {} shot Accuracy: {:.4f}%'.format(n, k, acc))
        return acc

    best_val = 0.0
    test_results = []
    for ep in range(args.epochs):
        training(ep)
        val = validate()
        if val > best_val:
            test_results = []
            test_results.append(test(5, 1))
            test_results.append(test(5, 5))
            test_results.append(test(20, 1))
            test_results.append(test(20, 5))

    # Print best results
    print('\nResult: 5 way 1 shot Accuracy: {:.4f}%'.format(
        test_results[0]))
    print('\nResult: 5 way 1 shot Accuracy: {:.4f}%'.format(
        test_results[1]))
    print('\nResult: 5 way 1 shot Accuracy: {:.4f}%'.format(
        test_results[2]))
    print('\nResult: 5 way 1 shot Accuracy: {:.4f}%\n'.format(
        test_results[3]))


if __name__ == '__main__':
    main()