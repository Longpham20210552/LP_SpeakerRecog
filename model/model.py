import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import model.resnet as resnet


class background_resnet(nn.Module):
    def __init__(self, embedding_size, num_classes, modell='resnet18'):
        super(background_resnet, self).__init__()
        self.modell = modell                                    # Chọn model resnet18
        self.pretrained = resnet.resnet18(pretrained=False)    
        self.fc0 = nn.Linear(128, embedding_size)               # Định nghĩa lớp Fully_connected là lớp tuyến tính 1 đầu vào vector (1,128) đầu ra (1,128) 
        self.bn0 = nn.BatchNorm1d(embedding_size)               # Định nghĩa lớp BatchNorm1d là lớp 
        self.relu = nn.ReLU()                                   # Hàm kích hoạt ReLU
        self.last = nn.Linear(embedding_size, num_classes)      # Định nghĩa lớp Fully_connected là lớp tuyến tính 1 đầu vào vector (1,128) đầu ra (1,n_classes = 233) 

    def forward(self, x):
        # input enroll test x: minibatch (1) x 1 x 40 x 100
        # input train x: minibatch (64) x 1 x height x width
        x = self.pretrained.conv1(x)        # Lớp tích chập 1, kernel 7x7, padding = 3, tăng số kênh từ 1 lên 16, kích thước đầu ra (batch_size, 16, 40, 100)                           
        x = self.pretrained.bn1(x)          # Lớp BatchNorm2d đầu tiên 
        x = self.pretrained.relu(x)         # Hàm kích hoạt ReLU
        x = self.pretrained.layer1(x)       # Khối makelayer 1, gồm 2 khối BasicBlock (4 tầng conv), (batch_size, 16, 40, 100)
        x = self.pretrained.layer2(x)       # Khối makelayer 2, gồm 2 khối BasicBlock (4 tầng conv), (batch_size, 32, 40/2 =20, 100/2 = 50)
        x = self.pretrained.layer3(x)       # Khối makelayer 3, gồm 2 khối BasicBlock (4 tầng conv), (batch_size, 64, 20/2 = 10, 50/2 = 25)
        x = self.pretrained.layer4(x)       # Khối makelayer 4, gồm 2 khối BasicBlock (4 tầng conv), (batch_size, 128, 10/2 = 5, 25/2 = 13)
        out = F.adaptive_avg_pool2d(x,1)    # Lấy trung bình trên từng kênh (batch_size, 128, 1, 1)
        out = torch.squeeze(out)            # (batch, n_embed=128)
        out = out.view(x.size(0), -1)       # (n_batch, n_embed=128)
        spk_embedding = self.fc0(out)       
        out = F.relu(self.bn0(spk_embedding)) # (batch, n_embed)
        out = self.last(out)
        return spk_embedding, out