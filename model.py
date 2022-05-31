import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # dropout
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc1 = nn.Linear(in_features=32 * 32 * 3, out_features=10)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        '''
        # try to define different layers 
        '''

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(-x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = self.fc1(x)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        '''
        # try to forward with different predefined layers 
        '''
        return x
        
class Net2(nn.Module):
	# init定义网络中的结构
    def __init__(self):
        super().__init__()
        # 3输入，16输出，卷积核(7, 7)，膨胀系数为2
        self.conv1 = nn.Conv2d(3,6,5,padding=2)  
        self.conv2 = nn.Conv2d(6,16,5,padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        # dropout
        self.conv2_drop = nn.Dropout2d()
        # 全连接层
        self.fc1 = nn.Linear(8*8*16,200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 10)
	
	# forward定义数据在网络中的流向
    def forward(self, x):
    	# 卷积之后做一个最大池化，然后RELU激活
        x = self.pool(F.relu(self.conv1(-x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # 整形
        #x = x.view(-1, 55696*2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = self.fc3(x)
        return x
        
        
class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,3)
        self.conv3=nn.Conv2d(16,32,3)
        self.fc1=nn.Linear(32*5*5,400)
        self.fc2=nn.Linear(400,200)
        self.fc3=nn.Linear(200,100)
        self.fc4=nn.Linear(100,10)
    def forward(self, x):
        x=self.pool(self.conv1(x))
        x=self.conv2(x)
        x=self.pool(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = self.fc3(x)
        x=self.fc4(x)
        return x
        
        

#net = Net1()
#net.to(device)
#print(net)
#!!!!!!!!
#net.cuda()
#print(net.cuda())
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 10)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size = 7)
        self.dropout2 = nn.Dropout(p=0.1)

        self.conv3 = nn.Conv2d(128, 128, kernel_size = 4)
        self.dropout3 = nn.Dropout(p=0.1)

        self.conv4 = nn.Conv2d(128, 256, kernel_size = 4)
        self.dropout4 = nn.Dropout(p=0.1)

        self.fc = nn.Linear(9216, 4096)
    

    def forward(self, x):
        block1 = self.dropout1(F.max_pool2d(F.relu(self.conv1(x)),2))
        block2 = self.dropout2(F.max_pool2d(F.relu(self.conv2(block1)),2))
        block3 = self.dropout3(F.max_pool2d(F.relu(self.conv3(block2)),2))
        block4 = self.dropout3(F.relu(self.conv4(block3)))
        flatten = block4.view(-1,9216)
        output = self.fc(flatten)
        return output


class SiameseSVMNet(nn.Module):
    def __init__(self):
        super(SiameseSVMNet, self).__init__()
        self.featureNet = FeatureNet()
        self.fc = nn.Linear(4096, 1)

    def forward(self, x1, x2):
        output1 = self.featureNet(x1)
        output2 = self.featureNet(x2)
        difference = torch.abs(output1 - output2)
        output = self.fc(difference)
        return output

    def get_FeatureNet(self):
        return self.featureNet
         
        