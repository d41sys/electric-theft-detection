import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import tensorflow as tf
from torchvision import utils
import numpy as np

import torch.nn.functional as F
import pandas as pd
import pytz
from datetime import datetime


class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=(4, 3), stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x3)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x6)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(6, 6), stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x12)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=(5, 4), stride=2, padding=1)
            # output of main module --> Image (Cx37x28)
        )

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

generator = Generator(1)
# Load the pre-trained generator model
generator.load_state_dict(torch.load('models/generator_26000.pkl'))
# Set the generator to evaluation mode

# Move the generator to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)

generator.eval()

# Function to generate synthetic samples
def generate_synthetic_samples(generator, num_samples, latent_dim=100):
    # Generate latent space vectors (noise)
    z = torch.randn(num_samples, latent_dim, 1, 1, device=device)  # Adjust latent_dim if different
    
    with torch.no_grad():  # Disable gradient calculation for faster inference
        synthetic_data = generator(z).cpu().numpy()  # Generate samples
    
    # Reshape the generated samples as needed for your dataset
    return synthetic_data

# Number of synthetic samples to generate
# num_synthetic_samples = 5
num_synthetic_samples = 27840-8091

# Generate synthetic samples
synthetic_samples = generate_synthetic_samples(generator, num_synthetic_samples)

# Now, `synthetic_samples` contains the generated data
# You can save them, integrate them into your dataset, or use them as needed
print(f"Generated {synthetic_samples.shape[0]} synthetic samples.")

synthetic_samples_reshaped = [sample.reshape(37, 28) for sample in synthetic_samples]
synthetic_samples_df = pd.DataFrame({'data': synthetic_samples_reshaped})

synthetic_samples_df['label'] = 1

data_dir = './data'
save_file = f'{data_dir}/GAN_data.npz'
print('Saving to: ', save_file)
data = synthetic_samples_df['data']
label = synthetic_samples_df['label']
np.savez_compressed(save_file, data=data, label=label)