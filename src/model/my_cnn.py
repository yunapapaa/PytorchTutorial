import torch.nn as nn
import torch.nn.functional as F


class MyCNN(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_ave_pool = nn.AdaptiveAvgPool2d((1, 1))  
        
        self.fc = nn.Linear(64, n_class)


    def forward(self, x):
        # 入力 : [bs, 3, 32, 32]（RGB）

        # 畳み込み1 : [bs, 3, 32, 32] -> [bs, 32, 32, 32]
        # MaxPooling : [bs, 32, 32, 32] -> [bs, 32, 16, 16]   
        x = self.conv1(x)             
        x = F.relu(x)              
        x = self.max_pool(x)              

        # 畳み込み1 : [bs, 32, 32, 32] -> [bs, 64, 32, 32]
        # MaxPooling : [bs, 64, 16, 16] -> [bs, 64, 8, 8]   
        x = self.conv2(x)             
        x = F.relu(x)                 
        x = self.max_pool(x)          

        # global average pooling (特徴マップ -> スカラー値) : [bs, 64, 8, 8] -> [bs, 64, 1, 1]
        x = self.global_ave_pool(x)   
        # 一次元化 : [bs, 64, 1, 1] -> [bs, 64]
        x = x.view(x.size(0), -1)     
        # クラス分類 : [bs, 64] -> [bs, 10]
        x = self.fc(x)      

        return x

