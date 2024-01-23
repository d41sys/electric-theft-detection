import numpy as np
import torch
import os
import math
import torch.nn.functional as F
from torch.utils.data import Dataset
from copy import copy, deepcopy

ID_LEN = 4 #CAN bus 2.0 has 29 bits
DATA_LEN = 8 #Data field in Can message has 8 bytes

class DatasetPrepare(Dataset):
    def __init__(self, root_dir, sequence_size, pad_size, embed, max_time_position, gran, log_e, transform=None, is_train=True):
        if is_train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'test')
        # self.root_dir = root_dir
        self.transform = transform
        self.pad_size = pad_size
        self.embed = embed
        self.max_time_position = max_time_position
        self.gran = gran
        self.log_e = log_e
        self.sequence_size = sequence_size
        self.total_size = len(os.listdir(self.root_dir))

        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / self.embed)) for i in range(self.embed)] for pos in range(self.max_time_position)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])  # Use sin for even columns
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])  # Use cos for odd columns
        
    def __len__(self):
        return self.total_size
    
    def get_time(self, time_position):
        # Segment the corresponding position code according to the time position
        pe = torch.index_select(self.pe, 0, time_position)
        return pe
    
    def __getitem__(self, idx):
        filenames = '{}/{}.npz'.format(self.root_dir, idx)
        
        if not os.path.isfile(filenames):
            print(filenames + 'does not exist!')
        
        dataloader = np.load(filenames, allow_pickle=True)
        data, label = dataloader['X'], dataloader['y']
        # print("DATA: ", data, " AND LENGTH: ", len(data))
        # print("DATA: ", type(data), " AND LENGTH: ", len(data))
        # print("LABEL: ", label)
        
        # PREPROCESS
        # data = torch.from_numpy(data)
        # data = torch.tensor(data, dtype=torch.float32)
        data = torch.tensor(data, dtype=torch.int64)
        label = torch.tensor(label, dtype=torch.long)
        # print("DATA: ", data, " AND LENGTH: ", len(data))
        
        ori_seq_len = data.shape[0]
        pad_len = self.sequence_size - ori_seq_len
        # print(pad_len)
        # dasdasf
        ## PAD WITH MAX SIZE = 100
        data = F.pad(data.T, (0, pad_len)).T.numpy()

        if pad_len == 0:
            mask = np.array([False] * ori_seq_len)
        else:
            mask = np.concatenate((np.array([False] * ori_seq_len), np.array([True] * pad_len)))
        
        sample = {'data': data, 'mask': mask, 'label': label, 'idx': idx}
        
        # print("DATA FEATURE: ", data, " AND LENGTH: ", len(data))
        # print("MASK FEATURE: ", mask, " AND LENGTH: ", len(mask))
        # print("LABEL: ", label)
        # print("INDEX: ", idx)
        # print("DONE")
        # print(f"SAMPLE: {sample}")
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample