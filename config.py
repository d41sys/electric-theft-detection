import torch
import os
import time
import argparse

def prepare_fin(config):
    fin = open(config.result_file, 'a')
    fin.write('-------------------------------------\n')
    fin.write(config.model_name + '\n')
    fin.write('Begin time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())) + '\n')
    fin.write('Data root dir: ' + config.root_dir + '\n')
    fin.write('d_model: ' + str(config.d_model) + '\t pad_size: ' + str(config.pad_size) + '\t nhead: ' + str(config.nhead) + '\t num_layers: ' + str(config.num_layers) + '\n')
    fin.write('batch_size: ' + str(config.batch_size) + '\t learning rate: ' + str(config.lr) + '\t smooth factor: ' + '\n\n')
    fin.close()
    
def parser_process():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="data/ksm_transformer_best_result")
    parser.add_argument('--model', type=str, default="old")
    parser.add_argument('--log_mode', type=str, default="train")
    parser.add_argument('--window_size', type=int, default=37)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--mode', type=str, default='transformer')
    parser.add_argument('--model_name', type=str, default='KSM_Transformer')
    parser.add_argument('--model_path', type=str, default='model/')
    args,_ = parser.parse_known_args()
    return args

class Config:
    def __init__(self, args):
        self.model_name = args.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if getattr(torch, 'has_mps', False) else 'cpu')

        if args.mode == 'cae':
            self.dout_mess = 20 # 4 weeks
            self.d_model = self.dout_mess
            self.nhead = 20  # ori: 5
            print("The mode using CAE component")
        elif args.mode == 'dnn':
            self.dout_mess = 20 # 4 weeks
            self.d_model = self.dout_mess
            self.nhead = 10  # ori: 5
            print("The mode using DNN component")
        else:
            self.dout_mess = 28
            self.d_model = 28
            self.nhead = 28  # ori: 5
            print("Not using")

        self.model_path = args.model_path
        self.pad_size = args.window_size  
        self.window_size = args.window_size  
        self.max_time_position = 10000
        self.num_layers = 7
        self.log_e = 2
        self.model = args.model
        self.log_mode = args.log_mode
        print("Log mode: ", self.log_mode)
        
        self.mode = args.mode
        self.use_embedding = False
        self.embedding_size = 64
        self.add_norm = False

        self.classes_num = 2

        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.lr = args.lr  # 0.0001 learning rate
        self.root_dir = args.indir

        self.model_save_path = args.model_path + self.model_name + '/'
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        self.result_file = 'results/' + args.indir.split('/')[1] + '_' + args.mode + '.txt'

        self.isload_model = False
        # self.model_path = 'model/' + self.model_name + '/' + self.model_name + '_model_' + str(self.start_epoch) + '.pth'