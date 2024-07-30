import numpy as np
import torch
import os
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import Dataset
import torch.nn.functional as F

class ETDdataset(Dataset):
    def __init__(self, root_dir, window_size, is_train=True, include_data=False, transform=None):
        if is_train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'test')
            
        # self.num_classes = num_classes
        self.is_train = is_train
        self.transform = transform
        self.window_size = window_size
        self.total_size = len(os.listdir(self.root_dir))
    
    def __getitem__(self, idx):
        # filenames = '{}/{}.tfrec'.format(self.root_dir, idx)
        filenames = '{}/{}.npz'.format(self.root_dir, idx)
        
        # index_path = None
        # description = {'id_seq': 'int', 'data_seq': 'int','label': 'int'}
        # dataset = TFRecordDataset(filenames, index_path, description)
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        
        dataloader = np.load(filenames, allow_pickle=True)
        
        # pass header
        # data = next(iter(dataloader))
        
        # id_seq, data_seq, label = data['id_seq'], data['data_seq'], data['label']
        data, label = dataloader['X'], dataloader['y']
        
        # id_seq = id_seq.to(torch.float)
        # data_seq = data_seq.to(torch.float)
        data = torch.tensor(data, dtype=torch.int64)
        label = torch.tensor(label, dtype=torch.long)
        
        ori_seq_len = data.shape[0]
        pad_len = self.window_size - ori_seq_len
        
        data = F.pad(data.T, (0, pad_len)).T.numpy()
        
        sample = {'data': data, 'label': label, 'idx': idx}
        # id_seq[id_seq == 0] = -1
        # id_seq = id_seq.view(-1, self.window_size, ID_LEN)
        # data_seq = data_seq.view(-1, self.window_size, DATA_LEN)
        
        # if self.include_data:
        #     return id_seq, data_seq, label[0][0]
        # else:
        #     return id_seq, label[0][0]
            
        return data, label 
        
    def __len__(self):
        return self.total_size