import torch
import torch.nn as nn
import math
from dataset import DatasetPrepare
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix, roc_auc_score
import optuna
import os
import time
import warnings
import argparse
import torch.nn.functional as F
import torch.optim as optim
import random
import pandas as pd
import pytz
from datetime import datetime
from logger import Logger

warnings.filterwarnings("ignore")



layout = {
    "CAE-Transformer": {
        "losses": ["Multiline", ["loss/train", "loss/test"]],
        "learning rate": ["Multiline", ["learning_rate/lr"]],
    },
}

def draw_confusion(label_y, pre_y, path):
    confusion = confusion_matrix(label_y, pre_y)
    print(confusion)


def write_result(fin, label_y, pre_y):
    accuracy = accuracy_score(label_y, pre_y)
    precision = precision_score(label_y, pre_y)
    recall = recall_score(label_y, pre_y)
    f1 = f1_score(label_y, pre_y)
    print('  -- test result: ')
    print('    -- accuracy: ', accuracy)
    fin.write('    -- accuracy: ' + str(accuracy) + '\n')
    print('    -- recall: ', recall)
    fin.write('    -- recall: ' + str(recall) + '\n')
    print('    -- precision: ', precision)
    fin.write('    -- precision: ' + str(precision) + '\n')
    print('    -- f1 score: ', f1)
    fin.write('    -- f1 score: ' + str(f1) + '\n\n')
    report = classification_report(label_y, pre_y)
    fin.write(report)
    fin.write('\n\n')
    return f1

class Config:
    def __init__(self, args):
        self.model_name = 'ETD'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if getattr(torch, 'has_mps', False) else 'cpu')

        if args.mode == 'cae':
            self.dout_mess = 20 # 4 weeks
            self.d_model = self.dout_mess
            self.nhead = 10  # ori: 5
        if args.mode == 'dnn':
            self.dout_mess = 20 # 4 weeks
            self.d_model = self.dout_mess
            self.nhead = 10  # ori: 5
        else:
            self.dout_mess = 30
            self.d_model = 30
            self.nhead = 10  # ori: 5

        self.pad_size = args.window_size  
        self.window_size = args.window_size  
        self.max_time_position = 10000
        self.num_layers = 6
        self.log_e = 2
        
        self.mode = args.mode

        self.classes_num = 2

        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.lr = args.lr  # 0.0001 learning rate
        self.root_dir = args.indir

        self.model_save_path = './model/' + self.model_name + '/'
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        self.result_file = 'results/' + args.indir.split('/')[1] + '_' + args.mode + '.txt'

        self.isload_model = False
        # self.model_path = 'model/' + self.model_name + '/' + self.model_name + '_model_' + str(self.start_epoch) + '.pth'


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class Autoencoder1D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Autoencoder1D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return F.relu(x)
    
class ConvAutoencoder1D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvAutoencoder1D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            # Original: input_dim -> 32
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            # Original: 32 -> 64
            nn.ReLU()
        )
        self.fc = nn.Linear(128*28, output_dim)
        # Original: 64*28
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class DNN(nn.Module):
    def __init__(self, d_in, d_out):  # config.slsum_count, config.dnn_out_d
        super(DNN, self).__init__()
        self.l1 = nn.Linear(d_in, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, d_out)

    def forward(self, x):
        # print('x: ', x.numpy ()[0])
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        # print('dnn out: ', out.detach().numpy()[0])
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerPredictor(nn.Module):
    def __init__(self, config):
        super(TransformerPredictor, self).__init__()
        self.pad_size = config.pad_size

        self.ae = Autoencoder1D(28, config.dout_mess).to(config.device)
        self.cae = ConvAutoencoder1D(1, config.dout_mess).to(config.device)
        self.dnn = DNN(28, config.dout_mess).to(config.device)
        
        self.dout_mess = config.dout_mess
        self.mode = config.mode
        
        self.position_embedding = PositionalEncoding(config.d_model, dropout=0.0, max_len=config.max_time_position).to(config.device)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.nhead).to(config.device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config.num_layers).to(config.device)
        self.fc = nn.Linear(config.d_model, config.classes_num).to(config.device)

    def forward(self, data, mask):
        # Normal case
        # x = data.permute(2, 0, 1)
        
        # With AE 
        x = data
        # print("X: ", x.shape)
        # X:  torch.Size([128, 37, 28])
        
        if self.mode == 'cae':
            # Conv Autoencoder 1D =================================================   
            cae_out = torch.empty((x.shape[0], self.dout_mess, 0)).to(config.device)
            for i in range(self.pad_size):
                # shape of x[:, i, :] is (batch_size, 28)
                # tmp = self.cae(x[:, i, :]).unsqueeze(2)
                tmp = self.cae(x[:, i:i+1, :]).unsqueeze(2)
                cae_out = torch.concat((cae_out, tmp), dim=2)
                # sharp of cae_out is (batch_size, 20, 36)
                    
            # print("CAE OUT: ", cae_out.shape)
            # CAE OUT:  torch.Size([128, 20, 37]
            
            x = cae_out.permute(2, 0, 1)
            # sharp of x is (36, batch_size, 20)
        if self.mode == 'dnn':
            dnn_out = torch.empty((x.shape[0], self.dout_mess, 0)).to(config.device)
            for i in range(self.pad_size):
                # shape of x[:, i, :] is (batch_size, 28)
                # tmp = self.cae(x[:, i, :]).unsqueeze(2)
                tmp = self.dnn(x[:, i, :]).unsqueeze(2)
                dnn_out = torch.concat((dnn_out, tmp), dim=2)
                
            x = dnn_out.permute(2, 0, 1)
        else:
            x = x.permute(1, 0, 2)
        
        # Autoencoder 1D =======================================================
        # ae_out = torch.empty((x.shape[0], self.dout_mess, 0)).to(config.device)
        # for i in range(self.pad_size):
        #     tmp = self.ae(x[:, i, :]).unsqueeze(2)
        #     ae_out = torch.concat((ae_out, tmp), dim=2)
        # x = ae_out.permute(2, 0, 1)
        
        out = self.position_embedding(x)
        # print("OUT: ", out.shape)
        out2 = self.transformer_encoder(out, src_key_padding_mask=mask)
        out = out2.permute(1, 0, 2)
        out = torch.sum(out, 1)
        out = self.fc(out)
        return out

def prepare_fin(config):
    fin = open(config.result_file, 'a')
    fin.write('-------------------------------------\n')
    fin.write(config.model_name + '\n')
    fin.write('Begin time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())) + '\n')
    fin.write('Data root dir: ' + config.root_dir + '\n')
    fin.write('d_model: ' + str(config.d_model) + '\t pad_size: ' + str(config.pad_size) + '\t nhead: ' + str(config.nhead) + '\t num_layers: ' + str(config.num_layers) + '\n')
    fin.write('batch_size: ' + str(config.batch_size) + '\t learning rate: ' + str(config.lr) + '\t smooth factor: ' + '\n\n')
    fin.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="data/H_dataset_bestcase")
    parser.add_argument('--window_size', type=int, default=37)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--mode', type=str, default='cae')
    args = parser.parse_args()

    config = Config(args)
    prepare_fin(config)
    
    # Set print options
    # torch.set_printoptions(profile="full")
    
    logger = Logger('./logs/transformer_'+ datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y-%m-%d %H:%M:%S"), layout)
    
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_dataset = DatasetPrepare(config.root_dir, config.window_size, config.pad_size, config.d_model, config.max_time_position, config.log_e, is_train=True)
    test_dataset = DatasetPrepare(config.root_dir, config.window_size, config.pad_size, config.d_model, config.max_time_position, config.log_e, is_train=False)


    print("TRAIN SIZE:", len(train_dataset), " TEST SIZE:", len(test_dataset), " SIZE:", len(train_dataset)+len(test_dataset), " TRAIN RATIO:", round(len(train_dataset)/(len(train_dataset)+len(test_dataset))*100), "%")
    print("MODE: " + config.mode, " INDIR: " + args.indir, " WINDOW SIZE: " + str(args.window_size), " EPOCH: " + str(args.epoch), " BATCH SIZE: " + str(args.batch_size), " LR: " + str(args.lr))
    
    # 2 DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size)
    print('finish load data')
    
    if config.isload_model:
        print("Case loaded")
        fin = open(config.result_file, 'a')
        fin.write('load trained model :    model_path: ' + config.model_path)
        model = torch.load(config.model_path)
        start_epoch = config.start_epoch
        fin.close()
    else:
        print("Case trained")
        model = TransformerPredictor(config)
        start_epoch = -1
        
    loss_func = nn.CrossEntropyLoss().to(config.device)
    criterion = nn.CrossEntropyLoss().to(config.device)
    opt = optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = CosineWarmupScheduler(opt, warmup=50, max_iters=config.epoch_num*len(train_loader))
    
    mapk_scores = []
    auc_scores = []
    
    for epoch in range(start_epoch + 1, config.epoch_num):
        fin = open(config.result_file, 'a')
        print('--- epoch ', epoch)
        fin.write('-- epoch ' + str(epoch) + '\n')
        
        epoch_train_loss = 0  # Initialize epoch training loss
        
        for i, sample_batch in enumerate(train_loader):
            batch_data = sample_batch['data'].type(torch.FloatTensor).to(config.device)
            # batch_data = sample_batch['data'].type(torch.IntTensor).to(config.device)
            batch_mask = sample_batch['mask'].to(config.device)
            batch_label = sample_batch['label'].to(config.device)
            out = model(batch_data, batch_mask)
            # print("DATA: ", batch_data)
            # print("LABEL: ",batch_label)
            # print("OUT: ", out)
            loss = loss_func(out, batch_label)
            epoch_train_loss += loss.item() 
            
            # print("LOSS: ", loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()
            
            if i % 20 == 0:
                print('iter {} loss: '.format(i), loss.item())
        
        avg_train_loss = epoch_train_loss / len(train_loader)        
        torch.save(model, (config.model_save_path + config.model_name + '_model_{}.pth').format(epoch))
        
        # test
        label_y = []
        pre_y = []
        total_test_loss = 0  # Initialize total test loss
        with torch.no_grad():
            for j, test_sample_batch in enumerate(test_loader):
                test_data = test_sample_batch['data'].type(torch.FloatTensor).to(config.device)
                test_mask = test_sample_batch['mask'].to(config.device)
                test_label = test_sample_batch['label'].to(config.device)
                
                test_out = model(test_data, test_mask)
                
                loss_t = criterion(test_out, test_label)
                total_test_loss += loss_t.item()  # Accumulate test loss

                pre = torch.max(test_out, 1)[1].cpu().numpy()
                
                pre_y = np.concatenate([pre_y, pre], 0)
                label_y = np.concatenate([label_y, test_label.cpu().numpy()], 0)
            # print('label_y: ', label_y)
            # print('pre_y: ', pre_y)

            avg_test_loss = total_test_loss / len(test_loader)
            
            # Calculate AUC
            auc = roc_auc_score(label_y, pre_y)
            auc_scores.append(auc)
            print(f'Epoch {epoch+1}/{config.epoch_num}, AUC: {auc:.4f}')   
            
            write_result(fin, label_y, pre_y)
            draw_confusion(label_y, pre_y, '')
        fin.close()
    
        # ============ TensorBoard logging ============# 
        info = {
            'loss/train': avg_train_loss,
            'loss/test': avg_test_loss,
            'learning_rate/lr': opt.param_groups[0]['lr']
        }
        
        for tag, value in info.items():
            # Only apply .cpu() to tensors, skip for floats
            if isinstance(value, torch.Tensor):
                value = value.cpu()
            logger.scalar_summary(tag, value, epoch + 1)
        
    auc_scores_df = pd.DataFrame(auc_scores, columns=['AUC'])
    auc_scores_df.to_csv('auc_scores.csv', index=False)