import pandas as pd
import torch
import torch.nn as nn
# import vaex
import numpy as np
import glob

import argparse 
import json
import math
import csv

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path

np.random.seed(0)

class Writer:
    def __init__(self, outdir, type_name, start_idx=0,):
        self.outdir = Path(outdir)/ type_name
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.idx = start_idx
    def write(self, X, y):
        save_file = self.outdir / f'{self.idx}.npz'
        np.savez_compressed(save_file, X=X, y=y)
        self.idx += 1
    def start(self):
        print('Start writing to: ', self.outdir)

def write_to_file(writer, X, y):
    writer.start()
    try:
        for xi, yi in tqdm(zip(X, y)):
            writer.write(xi, yi)
    except: return False
    return True

def train_test_split(N, test_fraction):
    """
    Input: 
    N: the size of the dataset
    test_fraction: the portion for test set
    Output:
    Return the index for train, val, and test set
    """
    test_size = int(N * test_fraction)
    indices = np.random.permutation(N) 
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return [train_idx, test_idx]

def create_folder_from_npz_file(out_dir, dir_type, fn):
    file_name = Path("./data") / fn
    print('\n==> Loading from: ', file_name)
    df_np = np.load(file_name, allow_pickle=True)
    # print(df_np.files)
    data, label = df_np['data'], df_np['label']
    # print(len(data), len(label))
    np.set_printoptions(threshold=np.inf)
    # print(data[0])
    
    indices_lists = train_test_split(len(data), test_fraction=0.2) 
    prefix = ['train', 'val']
    
    for prefix, indices in zip(prefix, indices_lists):
        if prefix == dir_type:
            print(f'{prefix} size: ', len(indices))
            data_subset = data[indices] 
            label_subset = label[indices]

    out_path = out_dir  
    writer = Writer(outdir=out_path, type_name=f'{dir_type}')
    return write_to_file(writer, data_subset, label_subset)

def undersampling(X, y, ratio=1.0, choice=1):
    # Count of number of samples in each class
    class_counts = Counter(y)

    # Determining the class with fewer samples
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)

    # Creating SMOTE and Undersampler instances with the specified ratio
    undersampler = RandomUnderSampler(sampling_strategy=ratio, random_state=42)
    
    if choice == 0:
        print("=> Using SMOTE")
        smote = SMOTE(sampling_strategy=ratio, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    elif choice == 1:
        print("=> Using ADASYN")
        adasyn = ADASYN(sampling_strategy=ratio, random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
    else: 
        print("=> Using KMeansSMOTE")
        kmean_sm = KMeansSMOTE(kmeans_estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=42)
        X_resampled, y_resampled = kmean_sm.fit_resample(X, y)
        
        
    # Applying undersampling to reduce the number of samples in the majority class
    X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)

    # Counting the number of samples in each class in the final resampled dataset
    final_class_counts = Counter(y_resampled)

    return X_resampled, y_resampled, final_class_counts

def parse_option():
    parser = argparse.ArgumentParser('Arguments for preprocessing')
    parser.add_argument('--data_path', type=str, default='./data/new_clean_data2.csv', help='path to raw data')
    parser.add_argument('--label_path', type=str, default='./data/new_info2.csv', help='path to raw data')
    parser.add_argument('--output_path', type=str, default='./data/ksm_conv33x33', help='path to save processed data')
    parser.add_argument('--sampling', type=int, default=2, help='whether to use variant smote sampling or not')
    parser.add_argument('--window_size', type=int, default=33, help='window size for the dataset')
    parser.add_argument('--padding', type=int, default=1, help='whether to pad the data or not')
    parser.add_argument('--outfile', type=str, default='ksm33x33', help='name of the output file')
    opt = parser.parse_args()
    
    return opt

def load_data(opt):
    data = pd.read_csv(opt.data_path)
    data.drop('Unnamed: 0', axis=1, inplace=True)
    label = pd.read_csv(opt.label_path)
    label.drop('Unnamed: 0', axis=1, inplace=True)
    return pd.concat([label, data], axis=1)

def stride(x, opt):
    as_strided = np.lib.stride_tricks.as_strided
    if opt.padding == 1:
        output_shape = (1089 // opt.window_size, opt.window_size) # (33, 33)
        x = np.pad(pd.Series(x), (0, 1089 - 1036 + 2), 'constant')
    else:
        output_shape = (1036 // opt.window_size, opt.window_size) # (36, 28)
        x = np.pad(pd.Series(x), (0, 2), 'constant')
    return as_strided(x, output_shape, (8*opt.window_size, 8))

def save_data(df_dpr, opt):
    data_dir = './data'
    save_file = f'{data_dir}/{opt.outfile}.npz'
    print('=> Saving to: ', save_file) 
    data = df_dpr['data']
    label = df_dpr['label']
    np.savez_compressed(save_file, data=data, label=label)

def main():
    opt = parse_option()
    sampling_df = load_data(opt)
    print("Window size:", opt.window_size)
    if opt.padding == 1:
        print("=> Padding data")
    else:
        print("=> No padding")
    
    # SEPARATING DEPENDENT AND INDEPENDENT VARIABLES
    x = sampling_df.iloc[:, 2:].values
    y = sampling_df.iloc[:, 0].values
    print("Shape of data:", x.shape)
    print("Shape of label:", y.shape)
    
    if opt.sampling > 0:
        print("\n=> Using sampling method")
        # desired_ratio = 1.0
        x_resampled, y_resampled, final_class_counts = undersampling(x, y, ratio=1.0, choice=opt.sampling)
        print("Final class counts:", final_class_counts, "Total samples:", x_resampled.shape[0])
        resample = pd.DataFrame(x_resampled)
    else:
        print("=> No sampling method specified. Using original data")
        print("Final class counts:", x.shape[0])
        resample = pd.DataFrame(x)
    
    print("\nStriding data...")
    resample['data'] = resample[resample.columns].values.tolist()
    df_dpr = resample['data'].apply(stride, args=(opt,)).to_frame()
    if opt.sampling > 0:
        df_dpr['label'] = y_resampled
    else: 
        df_dpr['label'] = y
    
    save_data(df_dpr, opt)
    
    create_folder_from_npz_file(out_dir=opt.output_path, dir_type='train', fn=f'{opt.outfile}.npz')
    create_folder_from_npz_file(out_dir=opt.output_path, dir_type='val', fn=f'{opt.outfile}.npz')
    return 0

if __name__ == '__main__':
    main()