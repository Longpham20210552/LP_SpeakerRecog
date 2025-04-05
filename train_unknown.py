import torch
from torch.utils.data import Subset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import json
import time
import os
import numpy as np
import configure as c
import pandas as pd
from DB_wav_reader import read_feats_structure_train
from SR_Dataset import read_MFB, TruncatedInputfromMFB, ToTensorInput, ToTensorDevInput, DvectorDataset, collate_fn_feat_padded
#from model.model import background_resnet
import matplotlib.pyplot as plt
from model.Pretrained_ReDimNet import ReDimNetWithClassifier
#from model.Pretrained_ReDimNet import model 
from model.redimnet import ReDimNetWrap
import torch.nn.functional as F 
from SRPL import ARPLoss

import json
import os
import numpy as np
from torch.utils.data import Subset
from torchvision import transforms

import json
import os
import numpy as np
from torch.utils.data import Subset
from torchvision import transforms

#Khởi tạo lại tham số mô hình khi bước vào mỗi foldfold
def reset_model_parameters(model):
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

#Lấy thông tin từ chỉ sốsố
def get_info_from_index(index, train_valid_DB):
    speaker_id = train_valid_DB['speaker_id'][index]
    # Số file của mỗi speaker
    speaker_indices = train_valid_DB[train_valid_DB['speaker_id'] == speaker_id].index.tolist()
    file_number = speaker_indices.index(index) + 1  # Số thứ tự file (bắt đầu từ 1)
    return speaker_id, file_number

#Xử lý dataset theo phương pháp k_fold cross validationvalidation
def load_dataset_kfold_pytorch_new(TRAIN_FEAT_DIR, n_splits=8, fold_info_path="fold_info_unknown.json"):
    train_valid_DB = read_feats_structure_train(TRAIN_FEAT_DIR)
    file_loader = read_MFB
    transform = transforms.Compose([
        TruncatedInputfromMFB(),
        ToTensorInput()
    ])
    transform_T = ToTensorDevInput()

    # Xác định nhãn "unknown"
    unknown_label = "unknown"

    # Loại bỏ nhãn "unknown" khi tạo `spk_to_idx`
    valid_speaker_ids = sorted(set([
        spk for spk, lbl in zip(train_valid_DB['speaker_id'], train_valid_DB['speaker_id'])
        if unknown_label not in lbl.lower()  # Nếu "unknown" có trong nhãn thì loại bỏ
    ]))

    spk_to_idx = {spk: i for i, spk in enumerate(valid_speaker_ids)}  # Ánh xạ các nhãn hợp lệ
    spk_to_idx[unknown_label] = len(spk_to_idx)  # Thêm lớp "unknown" nếu cần thiết

    full_dataset = DvectorDataset(DB=train_valid_DB, loader=file_loader, transform=transform, spk_to_idx=spk_to_idx)

    # Tạo danh sách file của mỗi speaker
    speaker_to_files = {spk: [] for spk in valid_speaker_ids}
    for idx, speaker_id in enumerate(train_valid_DB['speaker_id']):
        if unknown_label not in speaker_id.lower():  # Chỉ thêm file hợp lệ
            speaker_to_files[speaker_id].append(idx)

    # Đảm bảo mỗi speaker có đúng `n_splits` file
    for speaker, files in speaker_to_files.items():
        assert len(files) == n_splits, f"Speaker {speaker} does not have exactly {n_splits} files."

    # Xây dựng folds, chia fold theo các nội dung trêntrên
    fold_info = []
    folds = []

    for fold in range(n_splits):
        train_indices = []
        valid_indices = []
        train_data_pairs = []
        valid_data_pairs = []

        for speaker, files in speaker_to_files.items():
            valid_file_idx = files[fold]
            train_files_idx = [f for i, f in enumerate(files) if i != fold]

            valid_indices.append(valid_file_idx)
            train_indices.extend(train_files_idx)

            valid_data_pairs.append((valid_file_idx, speaker, files.index(valid_file_idx) + 1))
            train_data_pairs.extend([(f_idx, speaker, files.index(f_idx) + 1) for f_idx in train_files_idx])

        train_dataset = Subset(full_dataset, train_indices)
        valid_dataset = Subset(full_dataset, valid_indices)

        fold_info.append({
            'fold': fold + 1,
            'train_data_pairs': train_data_pairs,
            'valid_data_pairs': valid_data_pairs
        })
        folds.append((train_dataset, valid_dataset))

        print(f"\nFold {fold + 1}:")
        print("Training set:")
        for idx, spk, file_no in train_data_pairs:
            print(f"Index: {idx}, Speaker: {spk}, File No: {file_no}")
        print("Validation set:")
        for idx, spk, file_no in valid_data_pairs:
            print(f"Index: {idx}, Speaker: {spk}, File No: {file_no}")

    if fold_info_path:
        with open(fold_info_path, "w") as f:
            json.dump(fold_info, f, indent=4)

    print(len(spk_to_idx))
    return folds, len(spk_to_idx), fold_info


# Xử lý dataset theo phương pháp thông thường
def load_dataset_kfold_pytorch(TRAIN_FEAT_DIR, n_splits=8, fold_info_path="fold_info_final2.json"):
    # Đọc cấu trúc của dữ liệu và khởi tạo các thông số cần thiết
    train_valid_DB = read_feats_structure_train(TRAIN_FEAT_DIR)
    file_loader = read_MFB
    transform = transforms.Compose([
        TruncatedInputfromMFB(),
        ToTensorInput()
    ])
    transform_T = ToTensorDevInput()
    
    speaker_list = sorted(set(train_valid_DB['speaker_id']))
    spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
    full_dataset = DvectorDataset(DB=train_valid_DB, loader=file_loader, transform=transform, spk_to_idx=spk_to_idx)
    
    # Tạo một dictionary để lưu danh sách file của mỗi speaker
    speaker_to_files = {spk: [] for spk in speaker_list}
    for idx, speaker_id in enumerate(train_valid_DB['speaker_id']):
        speaker_to_files[speaker_id].append(idx)

    # Đảm bảo mỗi speaker có đúng 8 file
    for speaker, files in speaker_to_files.items():
        assert len(files) == n_splits, f"Speaker {speaker} does not have exactly {n_splits} files."

    # Xây dựng folds theo yêu cầu
    fold_info = []
    folds = []
    
    for fold in range(n_splits):
        train_indices = []
        valid_indices = []
        train_data_pairs = []
        valid_data_pairs = []

        for speaker, files in speaker_to_files.items():
            # Đảm bảo mỗi fold có một file khác nhau của speaker trong valid
            valid_file_idx = files[fold]
            train_files_idx = [f for i, f in enumerate(files) if i != fold]

            valid_indices.append(valid_file_idx)
            train_indices.extend(train_files_idx)

            # Ghi chú thông tin file và vị trí của nó
            valid_data_pairs.append((valid_file_idx, speaker, files.index(valid_file_idx) + 1))
            train_data_pairs.extend([(f_idx, speaker, files.index(f_idx) + 1) for f_idx in train_files_idx])

        # Tạo Subset cho train và valid
        train_dataset = Subset(full_dataset, train_indices)
        valid_dataset = Subset(full_dataset, valid_indices)

        # Thêm thông tin fold
        fold_info.append({
            'fold': fold + 1,
            'train_data_pairs': train_data_pairs,
            'valid_data_pairs': valid_data_pairs
        })
        folds.append((train_dataset, valid_dataset))
        print(f"\nFold {fold + 1}:")
        print("Training set:")
        for idx, spk, file_no in train_data_pairs:
            print(f"Index: {idx}, Speaker: {spk}, File No: {file_no}")
        print("Validation set:")
        for idx, spk, file_no in valid_data_pairs:
            print(f"Index: {idx}, Speaker: {spk}, File No: {file_no}")

    # Lưu fold_info vào file JSON nếu cần
    if fold_info_path:
        with open(fold_info_path, "w") as f:
            json.dump(fold_info, f, indent=4)

    print (len(spk_to_idx))
    return folds, len(spk_to_idx), fold_info

def load_out_dataset():
    """
    Tải out_dataset từ tập tin đặc trưng.
    Trả về: 
      - out_dataset: Dataset được trích xuất và tiền xử lý.
      - n_classes: Số lượng lớp (số người nói).
      - spk_to_idx: Ánh xạ từ speaker ID sang index.
    """
    # Load toàn bộ database
    out_DB = read_feats_structure_train(c.OUT_FEAT_DIR)  # OUT_FEAT_DIR là thư mục chứa các đặc trưng của out_dataset
    file_loader = read_MFB  # Hàm loader: numpy array (n_frames, n_dims)
    
    # Biến đổi (transform) cho dataset
    transform = transforms.Compose([
        TruncatedInputfromMFB(),  # numpy array: (1, n_frames, n_dims)
        ToTensorInput()  # torch tensor: (1, n_dims, n_frames)
    ])
    
    # Xác định danh sách người nói
    speaker_list = sorted(set(out_DB['speaker_id']))  # Lấy danh sách speaker_id duy nhất
    spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}  # Ánh xạ speaker_id sang index
    
    # Số lượng lớp (n_classes)
    n_classes = len(speaker_list)
    
    # Tạo dataset
    out_dataset = DvectorDataset(DB=out_DB, loader=file_loader, transform=transform, spk_to_idx=spk_to_idx)
    
    return out_dataset, n_classes, spk_to_idx

def load_dataset(val_ratio):
    # Tải dataset, tham số val_ratio là tỉ lệ số đoạn ghi âm được sử dụng để kiểm tra trong quá trình training
    # Chia train và valid từ dataset theo tỉ lệ val_ratio
    train_DB, valid_DB = split_train_dev(c.TRAIN_FEAT_DIR, val_ratio) 
    file_loader = read_MFB # numpy array:(n_frames, n_dims)
    transform = transforms.Compose([
        TruncatedInputfromMFB(), # numpy array:(1, n_frames, n_dims)
        ToTensorInput() # torch tensor:(1, n_dims, n_frames)
    ])
    transform_T = ToTensorDevInput()
    speaker_list = sorted(set(train_DB['speaker_id'])) # len(speaker_list) == n_speakers
    spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
    train_dataset = DvectorDataset(DB=train_DB, loader=file_loader, transform=transform, spk_to_idx=spk_to_idx)
    valid_dataset = DvectorDataset(DB=valid_DB, loader=file_loader, transform=transform_T, spk_to_idx=spk_to_idx)
    n_classes = len(speaker_list) # Số người tham gia training
    return train_dataset, valid_dataset, n_classes

def split_train_dev(train_feat_dir, valid_ratio):
    train_valid_DB = read_feats_structure_train(train_feat_dir)
    total_len = len(train_valid_DB) # 148642
    valid_len = int(total_len * valid_ratio/100.)
    train_len = total_len - valid_len
    shuffled_train_valid_DB = train_valid_DB.sample(frac=1).reset_index(drop=True)
    # Split the DB into train and valid set
    train_DB = shuffled_train_valid_DB.iloc[:train_len]
    valid_DB = shuffled_train_valid_DB.iloc[train_len:]
    # Reset the index
    train_DB = train_DB.reset_index(drop=True)
    valid_DB = valid_DB.reset_index(drop=True)
    print('\nTraining set %d utts (%0.1f%%)' %(train_len, (train_len/total_len)*100))
    print('Validation set %d utts (%0.1f%%)' %(valid_len, (valid_len/total_len)*100))
    print('Total %d utts' %(total_len))
    
    return train_DB, valid_DB


def main():
    torch.cuda.empty_cache()
    # Khởi tạo các hyperparameters
    use_cuda = True
    #torch.backends.cudnn.enabled = False
    val_ratio = 1      # 10% Validation 
    
    embedding_size = 192    # Kích thước của D-vectors
    start = 1               # Vòng lặp bắt đầu 
    n_epochs = 250          # Training kéo dài 30 epoch
    end = start + n_epochs  
    lr = 1e-2            # Learning_rate = 0.12
    wd = 1e-4                # Điều khoản phạt hàm mất mát
    optimizer_type = 'adam'  # ex) sgd, adam, adagrad; chọn SGD làm phương pháp tối ưu
    batch_size = 32         # Kích thước batch training 
    valid_batch_size = 16   # Kích thước batch validation
    use_shuffle = True      # Có xáo trộn dữ liệu không?
    #train_dataset, valid_dataset, n_classes = load_dataset(val_ratio)  # Tải tập dữ liệu
    n_classes = 580
    print('\nNumber of classes (speakers):\n{}\n'.format(n_classes))   # Số lượng người training: 233
    log_dir = 'model_saved_kfold_embeddings_unknown_3' # Lưu checkpoints sau mỗi epoch
    log_dir_2 = 'model_saved_kfold_embeddings_3_128'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Khởi tạo mô hình và tham số ban đầu
    #model = background_resnet(embedding_size=embedding_size, num_classes=n_classes)
    backbone = ReDimNetWrap()
    #backbone = torch.hub.load('IDRnD/ReDimNet', 'b0', pretrained=True, finetuned=False)
    model = ReDimNetWithClassifier(backbone, num_classes = n_classes)
    '''
    # Tải checkpoint
    checkpoint = torch.load(log_dir_2 + '/checkpoint_fold_1_epoch_' + str(175) + '.pth')

    # Nạp tham số vào mô hình
    model.load_state_dict(checkpoint['state_dict'])
    '''
    model.cuda()  # Sử dụng GPU
    
    # Định nghĩa hàm mất mát, hàm tối ưu, và hàm điều chỉnh học trong quá trình training
    criterion = nn.CrossEntropyLoss()  # Hàm mất mát Cross Entropy Loss
    optimizer = create_optimizer(optimizer_type, model, lr, wd) # Sử dụng hàm tối ưu
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 7, factor = 0.5, min_lr=1e-10, verbose=1)  # Giảm lr sau 1 số epohất định nếu không có cải thiện về giá trị hàm mất mát

 
    folds, _, fold_info = load_dataset_kfold_pytorch_new(c.TRAIN_FEAT_DIR, n_splits=8)                  
    
    torch.cuda.empty_cache()

    all_fold_train_losses = []
    all_fold_valid_losses = []
    start_fold = 0
    # Lặp qua từng fold trong k-fold cross-validation
    for fold_idx, (train_dataset, valid_dataset) in enumerate(folds):
        if fold_idx < start_fold:
            continue
        print(f"Starting fold {fold_idx + 1}/{len(folds)}")
        reset_model_parameters(model)
        # Khởi tạo DataLoader cho tập train và valid của fold hiện tại
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = use_shuffle)
        valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = valid_batch_size, shuffle = False,
                                                    collate_fn = collate_fn_feat_padded)
        
        # Mảng lưu giá trị hàm mất mát qua từng epoch cho fold hiện tại
        avg_train_losses = []
        avg_valid_losses = []
        
        # Làm trống cache của CUDA (nếu dùng GPU)
        torch.cuda.empty_cache()
        
        # Vòng lặp huấn luyện qua từng epoch
        for epoch in range(start, end):
            # Huấn luyện một epoch
            train_loss = train(train_loader, model, criterion, optimizer, use_cuda, epoch, n_classes)
            
            # Đánh giá trên tập valid sau khi huấn luyện xong mỗi epoch
            valid_loss = validate(valid_loader, model, criterion, use_cuda, epoch)
            
            # Điều chỉnh learning rate nếu cần
            scheduler.step(valid_loss, epoch)
            
            # Lưu giá trị hàm mất mát vào các mảng
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            # Lưu model sau mỗi epoch
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, f'{log_dir}/checkpoint_fold_{fold_idx+1}_epoch_{epoch}.pth')

        
        # Lưu kết quả hàm mất mát của fold hiện tại vào danh sách chung
        all_fold_train_losses.append(avg_train_losses)
        all_fold_valid_losses.append(avg_valid_losses)
        visualize_the_losses(avg_train_losses, avg_valid_losses, fold_idx+1)
        print(f"Fold {fold_idx + 1} completed. Average Training Loss: {np.mean(avg_train_losses):.4f}, "
              f"Average Validation Loss: {np.mean(avg_valid_losses):.4f}\n")
    

def train(train_loader, model, criterion, optimizer, use_cuda, epoch, n_classes):
    batch_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    
    n_correct, n_total = 0, 0
    log_interval = 84
    # switch to train mode
    model.train()
    
    end = time.time()
    # pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data) in enumerate(train_loader):
        inputs, targets = data  # target size:(batch size,1), input size:(batch size, 1, dim, win)
        targets = targets.view(-1) # target size:(batch size)
        current_sample = inputs.size(0)  # batch size
       
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        embeddings, output = model(inputs) # out size:(batch size = 64 , #classes ), for softmax
        #_, loss = criterion(x=embeddings, y=output, labels=targets)

        # Tính toán độ chính xác training
        n_correct += (torch.max(output, 1)[1].long().view(targets.size()) == targets).sum().item()
        n_total += current_sample
        train_acc_temp = 100. * n_correct / n_total
        train_acc.update(train_acc_temp, inputs.size(0)) # Tính độ chính xác trung bình
         
        loss = criterion(output , targets) # Tính toán hàm mất mát embeddings
        losses.update(loss.item(), inputs.size(0)) # Tính trung bình hàm mất mát
        
        # Tính toán gradient và cập nhật tham số
        optimizer.zero_grad() #Xóa gradient các tham số trước
        loss.backward()       #Backpropagation để tính toán tham số  
        optimizer.step()      #Cập nhật tham số mô hình

        # Thực thi các tác vụ đo thời gian
        batch_time.update(time.time() - end)  
        end = time.time()
    # Khi đã train hết dữ liệu, kết thúc 1 epoch, in ra màn hình
        if batch_idx % log_interval == 0:
            print(
                    'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.avg:.4f}\t'
                    'Acc {train_acc.avg:.4f}'.format(
                     epoch, batch_idx * len(inputs), len(train_loader.dataset),
                     100. * batch_idx / len(train_loader), 
                     batch_time=batch_time, loss=losses, train_acc=train_acc))
    return losses.avg
                     
def validate(val_loader, model, criterion, use_cuda, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    val_acc = AverageMeter()
    
    # Khởi tạo các biến tính độ chính xác
    n_correct, n_total = 0, 0
    
    # Chuyển model sang chế độ đánh giá
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data) in enumerate(val_loader):
            inputs, targets = data
            current_sample = inputs.size(0)  # batch size
            
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            
            # Tính toán đầu ra mô hình
            embeddings, output = model(inputs) # out size:(batch size = 64 , #classes ), for softmax
           # _, loss = criterion(x=embeddings, y=output, labels=targets)
            
            # Tính toán độ chính xác
            n_correct += (torch.max(output, 1)[1].long().view(targets.size()) == targets).sum().item()
            n_total += current_sample
            val_acc_temp = 100. * n_correct / n_total

            # Update độ chính xác mới, tính độ chính xác trung bìnhbình
            val_acc.update(val_acc_temp, inputs.size(0))
            
            # Tính toán giá trị hàm mất mát và update trung bình
            loss = criterion(output, targets)
            losses.update(loss.item(), inputs.size(0))

            # Tính toán thời gian kiểm tra
            batch_time.update(time.time() - end)
            end = time.time()
        
            # In ra các giá trị hàm mất mát trung bình, độ chính xác trung bình
        print('  * Validation: '
                  'Loss {loss.avg:.4f}\t'
                  'Acc {val_acc.avg:.4f}'.format(
                  loss=losses, val_acc=val_acc))
    
    return losses.avg

class AverageMeter(object):
    # Lớp khởi tạo và lưu trữ các giá trị trung bình
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # Lưu trữ giá trị hiện tại, giá trị tổng, số lượng giá trị, giá trị trung bình
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_optimizer(optimizer, model, new_lr, wd):
    # Tối ưu hàm mất mát
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,       # Sử dụng thuật toán Stochastic GD, với momentum = 0.9, độ
                              momentum=0.9, dampening=0,
                              weight_decay=wd)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=wd)
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  weight_decay=wd)
    return optimizer

def visualize_the_losses(train_loss, valid_loss, fold_number):
    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss, label='Validation Loss')
    
    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 3.5) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    #file_name = f'loss_plot_fold_{fold_number}.png'
    file_name = f'loss_plot_embedding_srpl_fold_{fold_number}.png'
    fig.savefig(file_name, bbox_inches='tight')

if __name__ == '__main__':
    main()
  