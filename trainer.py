import torch
import torch.nn as nn
import math
from dataset import DatasetPrepare
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix
import os
import time
import warnings
import argparse
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings("ignore")


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


class Config:
    def __init__(self, args):
        self.model_name = 'ETD'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if getattr(torch, 'has_mps', False) else 'cpu')

        self.dout_mess = 28 # 4 weeks
        self.d_model = self.dout_mess
        self.nhead = 7  # ori: 5

        self.pad_size = args.window_size  # 28
        self.window_size = args.window_size  # 28
        self.max_time_position = 10000
        self.num_layers = 6
        self.gran = 1e-7  # ori: 1e-6
        self.log_e = 2

        self.classes_num = 2

        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.lr = args.lr  # 0.0001 learning rate
        self.root_dir = args.indir

        self.model_save_path = './model/' + self.model_name + '/'
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        self.result_file = 'result/' + 'ETD.txt'

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

        self.position_embedding = PositionalEncoding(config.d_model, dropout=0.0, max_len=config.max_time_position).to(config.device)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.nhead).to(config.device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config.num_layers).to(config.device)
        self.fc = nn.Linear(config.d_model, config.classes_num).to(config.device)

    def forward(self, data, mask):
        x = data.permute(1, 0, 2)
        # print(type(x), x.shape)
        
        out = self.position_embedding(x)
        out2 = self.transformer_encoder(out, src_key_padding_mask=mask)
        # if torch.isnan(out2).any(): 
        #     with open("nan.txt", "a") as f:
        #         f.write('\n' + "="*20+'\n')
        #         f.write("data: " + str(data))
        #         f.write("PE: " + str(out))
        #         f.write(str(out2))
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
    fin.write('batch_size: ' + str(config.batch_size) + '\t learning rate: ' + str(config.lr) + '\t smooth factor: ' + str(config.gran) + '\n\n')
    fin.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="data/outliners2")
    parser.add_argument('--window_size', type=int, default=36)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()

    config = Config(args)   
    prepare_fin(config)
    
    # Set print options
    torch.set_printoptions(profile="full")
    
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_dataset = DatasetPrepare(config.root_dir, config.window_size, config.pad_size, config.d_model, config.max_time_position, config.gran, config.log_e, is_train=True)
    test_dataset = DatasetPrepare(config.root_dir, config.window_size, config.pad_size, config.d_model, config.max_time_position, config.gran, config.log_e, is_train=False)


    print("TRAIN SIZE:", len(train_dataset), " TEST SIZE:", len(test_dataset), " SIZE:", len(train_dataset)+len(test_dataset), " TRAIN RATIO:", round(len(train_dataset)/(len(train_dataset)+len(test_dataset))*100), "%")
    
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
    opt = optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = CosineWarmupScheduler(opt, warmup=50, max_iters=config.epoch_num*len(train_loader))
    
    for epoch in range(start_epoch + 1, config.epoch_num):
        fin = open(config.result_file, 'a')
        print('--- epoch ', epoch)
        fin.write('-- epoch ' + str(epoch) + '\n')
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
            # print("LOSS: ", loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()
            
            if i % 20 == 0:
                print('iter {} loss: '.format(i), loss.item())
        torch.save(model, (config.model_save_path + config.model_name + '_model_{}.pth').format(epoch))
        
        # test
        label_y = []
        pre_y = []
        with torch.no_grad():
            for j, test_sample_batch in enumerate(test_loader):
                test_data = test_sample_batch['data'].type(torch.FloatTensor).to(config.device)
                test_mask = test_sample_batch['mask'].to(config.device)
                test_label = test_sample_batch['label'].to(config.device)
                
                test_out = model(test_data, test_mask)

                pre = torch.max(test_out, 1)[1].cpu().numpy()
                
                pre_y = np.concatenate([pre_y, pre], 0)
                label_y = np.concatenate([label_y, test_label.cpu().numpy()], 0)
            write_result(fin, label_y, pre_y)
            draw_confusion(label_y, pre_y, '')
        fin.close()