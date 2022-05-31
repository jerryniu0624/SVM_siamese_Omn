import numpy as np
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model",type=int,default=8)
args = parser.parse_args()


def draw_loss(loss1,name):
    x=range(0,len(loss1))
    plt.title("loss of the Alexnet") 
    plt.xlabel("training times") 
    plt.ylabel("loss") 
    plt.plot(x,loss1)
    plt.legend() # 添加图例
    plt.savefig('./'+name+'.jpg')
    plt.show()
    plt.close()


def draw_acc(acc,name):
    x=range(0,len(acc))
    plt.title("accuracy of the Alexnet")
    plt.xlabel("training times")
    plt.ylabel("accuracy")
    plt.plot(x,acc)
    plt.legend() # 添加图例
    plt.savefig('./'+name+'.jpg')
    plt.show()
    plt.close()


if __name__ == '__main__':

    name1='loss'+'_model:'+str(args.model)
    name2='acc'+'_model:'+str(args.model)
    loss1=np.load(name1+'.npy')
    print(loss1)
    
    acc=np.load(name2+'.npy')
    print(acc)
    draw_loss(loss1,name1)
    draw_acc(acc,name2)
