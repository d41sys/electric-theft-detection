import torch
import torch.nn as nn
import math
from dataset import DatasetPrepare
from torch.utils.data import DataLoader
import numpy as np

def init_seed(seed):
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def data_preparing(config, args):
    init_seed(199)
    train_dataset = DatasetPrepare(config.root_dir, config.window_size, config.pad_size, config.d_model, config.max_time_position, config.log_e, is_train=True)
    test_dataset = DatasetPrepare(config.root_dir, config.window_size, config.pad_size, config.d_model, config.max_time_position, config.log_e, is_train=False)


    print("TRAIN SIZE:", len(train_dataset), " TEST SIZE:", len(test_dataset), " SIZE:", len(train_dataset)+len(test_dataset), " TRAIN RATIO:", round(len(train_dataset)/(len(train_dataset)+len(test_dataset))*100), "%")
    print("MODE: " + config.mode, " INDIR: " + args.indir, " WINDOW SIZE: " + str(args.window_size), " EPOCH: " + str(args.epoch), " BATCH SIZE: " + str(args.batch_size), " LR: " + str(args.lr))

    # 2 DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size)
    print('finish load data')
    
    return train_loader, test_loader