import numpy as np
import torch
import os
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import Dataset
import torch.nn.functional as F

ID_LEN = 33 #CAN bus 2.0 has 29 bits
DATA_LEN = 8 #Data field in Can message has 8 bytes
HIST_LEN = 256

class ETDDataset(Dataset):
    def __init__(self, root_dir, window_size, is_train=True, include_data=False, transform=None):
        if is_train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'val')
            
        # self.num_classes = num_classes
        self.include_data = include_data
        self.is_train = is_train
        self.transform = transform
        self.window_size = window_size
        self.total_size = len(os.listdir(self.root_dir))
    
    def __getitem__(self, idx):
        # ETD Dataset
        filenames = '{}/{}.npz'.format(self.root_dir, idx)
        
        dataloader = np.load(filenames, allow_pickle=True)
        data, label = dataloader['X'], dataloader['y']
        data = torch.tensor(data, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        
        data = data.view(-1, self.window_size, ID_LEN)
        
        # ori_seq_len = data.shape[0]
        # pad_len = self.window_size - ori_seq_len
        
        # data = F.pad(data.T, (0, pad_len)).T.numpy()
        
        # sample = {'data': data, 'label': label, 'idx': idx}
        
        return data, label 
        
    def __len__(self):
        return self.total_size