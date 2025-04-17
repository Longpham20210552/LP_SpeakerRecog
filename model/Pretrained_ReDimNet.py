import torch
import torch.nn as nn
import torch.nn.functional as F
from model.redimnet_base import ReDimNetWrap
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.hubconf import ReDimNet 

class classifier_spk(nn.Module):
    def __init__(self, num_classes = 580, feat_dim = 256):
        super(classifier_spk, self).__init__()
        self.num_classes = num_classes

        # Define batch normalization and ReLU activation
        self.bn1 = nn.BatchNorm1d(feat_dim)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(feat_dim, feat_dim, bias=False)

        self.bn1_1 = nn.BatchNorm1d(int(feat_dim*1/2))
        self.relu1_1 = nn.ReLU()
        self.fc1_1 = nn.Linear(feat_dim, int(feat_dim*1/2), bias=False)

        self.bn1_2 = nn.BatchNorm1d(int(feat_dim*1/2))
        self.relu1_2 = nn.ReLU()
        self.fc1_2 = nn.Linear(int(feat_dim*1/2), int(feat_dim*1/2), bias=False)

        self.fc2 = nn.Linear(int(feat_dim*1/2), num_classes, bias=False)

    def forward(self, x, return_feature=False):
        # Flatten the input
        x = torch.flatten(x, 1)
        
        # Apply the first set of layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Apply the second set of layers
        x = self.fc1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)


        # Apply the third set of layers
        x = self.fc1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        # Apply the final fully connected layer to produce logits for classification
        y = self.fc2(x)

        # If return_feature is True, return both the feature and the output
        if return_feature:
            return x, y
        else:
            return y
        
class AAMSoftmax_new(nn.Module):
    def __init__(self, nOut, nClasses, aam_margin=0.2, aam_scale=25, easy_margin=False):
        super(AAMSoftmax_new, self).__init__()
        
        self.m = aam_margin  # AAM margin
        self.s = aam_scale   # AAM scale
        self.in_feats = nOut
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print('Initialised AAM_Softmax without labels')

    def forward(self, x):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # Apply AAM-Softmax margin manipulation
        logits = phi * self.s  # Scale logits
        return logits

    
class AAMSoftmax(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.4, scale =40):
        super(AAMSoftmax, self).__init__()
        self.margin = margin
        self.scale = scale
        self.num_classes = num_classes
        self.fc = nn.Linear(embedding_dim, num_classes, bias=False)

    def forward(self, x, labels=None):
        # Normalize embeddings and weights
        x = F.normalize(x)
        weights = F.normalize(self.fc.weight)

        # Compute logits
        logits = F.linear(x, weights)
        
        # Apply margin if labels are provided
        if labels is not None:
            theta = torch.acos(torch.clamp(logits, -1.0, 1.0))
            target_logits = torch.cos(theta + self.margin)
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
            logits = logits * (1 - one_hot) + target_logits * one_hot

        return logits * self.scale

# Giả sử bạn đã tải mô hình từ link như sau
#model2 = torch.hub.load('IDRnD/ReDimNet', 'b0', pretrained=True, finetuned=False)
model2 = ReDimNet('b0')
#model2 = ReDimNetWrap()

class ReDimNetWithClassifier(nn.Module):
    def __init__(self, model2, num_classes):
        super(ReDimNetWithClassifier, self).__init__()
        self.redimnet2 = model2
        # Giữ lại toàn bộ mô hình ReDimNet
        self.redimnet = model2.backbone 
        self.pool = model2.pool
        self.bn = model2.bn
        self.linear = model2.linear
        # Lấy kích thước đầu ra từ lớp cuối cùng của ReDimNet
        self.pool_out_dim = model2.pool_out_dim
        #self.bn1 = nn.BatchNorm1d(128)
        #self.bn2 = nn.BatchNorm1d(128)
        # Định nghĩa lớp phân loại
        self.classifier = AAMSoftmax_new(192, nClasses = num_classes)
        #self.classifier_SRPL = classifier_spk(num_classes = 580, feat_dim= 512)
        
    def forward(self, x):
        # Truyền đầu vào qua toàn bộ mô hình ReDimNet
        x = self.redimnet(x)
        x = self.bn(self.pool(x))
        embeddings = self.linear(x)
        #x = F.relu(self.bn1(embeddings)) 
        # Áp dụng lớp phân loại cho các embeddings
        logits = self.classifier(embeddings)
        return embeddings, logits

# Khởi tạo mô hình ReDimNetWithClassifier
model_with_classifier = ReDimNetWithClassifier(model2, num_classes=233)
