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

SAVE_PER_TIMES = 100

from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

from logger import Logger


layout = {
    "WGAN-GP": {
        "main loss": ["Multiline", ["loss/generator", "loss/discriminator"]],
        "discriminator loss": ["Multiline", ["loss_D/discriminator_real", "loss_D/discriminator_fake"]],
        "wasserstein distance": ["Multiline", ["wasserstein_distance"]],
        "learning rate": ["Multiline", ["learning_rate/lr_G", "learning_rate/lr_D"]],
    },
}

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


class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx37x28)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx37x28)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x18x13)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x6)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(4, 4), stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x3x2)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(3, 2), stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*3*2)
    
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
    
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        """
        Early stopping to stop training when a monitored metric has stopped improving.
        
        Parameters:
        - patience: How long to wait after the last time the monitored metric improved.
        - min_delta: Minimum change in the monitored metric to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, g_loss, d_loss):
        # if self.best_loss is None:
        #     self.best_loss = g_loss
        # elif g_loss < self.best_loss - self.min_delta:
        #     self.best_loss = g_loss
        #     self.counter = 0  # Reset patience counter
        # else:
        #     self.counter += 1
        #     if self.counter >= self.patience:
        #         self.early_stop = True
        if g_loss > -0.1:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
   
class WGAN_GP(object):
    def __init__(self, len_loader):
        print("WGAN_GradientPenalty init model.")
        self.G = Generator(1)
        self.D = Discriminator(1)
        self.C = 1

        self.early_stopping = EarlyStopping(patience=20, min_delta=1e-3)
        # Check if cuda is available
        self.check_cuda(True)

        # WGAN values from paper
        self.learning_rate = 0.0001
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = 64

        # WGAN_gradient penalty uses ADAM
        # self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        # self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.d_optimizer = optim.RMSprop(self.D.parameters(), lr=self.learning_rate)
        self.g_optimizer = optim.RMSprop(self.G.parameters(), lr=self.learning_rate)

        # Set the logger
        self.logger = Logger('./logs/wgangp_'+ datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y-%m-%d %H:%M:%S"), layout)
        # self.logger.writer.flush()
        self.number_of_images = 10

        self.generator_iters = 20000 #args.generator_iters
        self.critic_iter = 5
        self.lambda_term = 10
        
        # self.g_lr_scheduler = CosineWarmupScheduler(self.g_optimizer, warmup=50, max_iters=self.generator_iters*len_loader)
        # self.d_lr_scheduler = CosineWarmupScheduler(self.d_optimizer, warmup=50, max_iters=self.generator_iters*len_loader)

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    def generate_synthetic_samples(self, minority_data_loader):
        self.t_begin = t.time()
        self.file = open("inception_score_graph.txt", "w+")

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(minority_data_loader)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        for g_iter in range(self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            # d_loss_real = 0
            # d_loss_fake = 0
            # Wasserstein_D = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                images = self.data.__next__()
                # Check for batch to have full batch_size
                if (images.size()[0] != self.batch_size):
                    continue

                z = torch.rand((self.batch_size, 100, 1, 1))

                images, z = self.get_torch_variable(images), self.get_torch_variable(z)

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                images = images.unsqueeze(1)

                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))

                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
                gradient_penalty.backward()


                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                # self.d_lr_scheduler.step()
                print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()
            # train generator
            # compute loss with fake images
            z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            # self.g_lr_scheduler.step()
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')
            
            # Saving model and sampling images every 1000th generator iterations
            if (g_iter) % SAVE_PER_TIMES == 0:
                # self.save_model()
                # # Workaround because graphic card memory can't store more than 830 examples in memory for generating image
                # # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
                # # This way Inception score is more correct since there are different generated examples from every class of Inception model
                # sample_list = []
                # for i in range(125):
                #     samples  = self.data.__next__()
                # #     z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
                # #     samples = self.G(z)
                #     sample_list.append(samples.data.cpu().numpy())
                # #
                # # # Flattening list of list into one list
                # new_sample_list = list(chain.from_iterable(sample_list))
                # print("Calculating Inception Score over 8k generated images")
                # # # Feeding list of numpy arrays
                # inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
                #                                       resize=True, splits=10)

                if not os.path.exists('training_result_images/'):
                    os.makedirs('training_result_images/')

                # Denormalize images and save them in grid 8x8
                z = self.get_torch_variable(torch.randn(800, 100, 1, 1))
                samples = self.G(z)
                samples = samples.mul(0.5).add(0.5)
                samples = samples.data.cpu()[:64]
                grid = utils.make_grid(samples)
                utils.save_image(grid, 'training_result_images/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))

                # Testing
                time = t.time() - self.t_begin
                #print("Real Inception score: {}".format(inception_score))
                print("Generator iter: {}".format(g_iter))
                print("Time {}".format(time))

                # Write to file inception_score, gen_iters, time
                #output = str(g_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
                #self.file.write(output)
                
                # Extract the learning rates
                gen_lr = self.g_optimizer.param_groups[0]['lr']
                disc_lr = self.d_optimizer.param_groups[0]['lr']
                
                print("Generator learning rate: {}".format(gen_lr))


                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'wasserstein_distance': Wasserstein_D.data,
                    'loss/generator': g_cost.data,
                    'loss/discriminator': d_loss.data,
                    'loss_D/discriminator_real': d_loss_real.data,
                    'loss_D/discriminator_fake': d_loss_fake.data,
                    'learning_rate/lr_G': gen_lr,
                    'learning_rate/lr_D': disc_lr
                }

                for tag, value in info.items():
                    # Only apply .cpu() to tensors, skip for floats
                    if isinstance(value, torch.Tensor):
                        value = value.cpu()
                    self.logger.scalar_summary(tag, value, g_iter + 1)

                # (3) Log the images
                info = {
                    'real_images': self.real_images(images, self.number_of_images),
                    'generated_images': self.generate_img(z, self.number_of_images)
                }

                for tag, images in info.items():
                    self.logger.image_summary(tag, images, g_iter + 1)
                
                self.save_model(str(g_iter))  
            # Check for early stopping
            # self.early_stopping(g_cost.item())

            # if self.early_stopping.early_stop:
            #     print(f"Early stopping triggered at generator iteration {g_iter}")
            #     break
            

        self.t_end = t.time()
        print('Time of training: {}'.format((self.t_end - self.t_begin)))
        #self.file.close()

        # Save the trained parameters
        self.save_model('last')
        return fake_images.data.cpu().numpy()

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')


    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # flatten the gradients to it calculates norm batchwise
        gradients = gradients.view(gradients.size(0), -1)
        
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 37, 28)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 37, 28)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 37, 28))
            else:
                generated_images.append(sample.reshape(37, 28))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self, iter):
        torch.save(self.G.state_dict(), './model/generator_'+iter+'.pkl')
        torch.save(self.D.state_dict(), './model/discriminator_'+iter+'.pkl')
        if iter == 'last':
            print('Models save to ./model/generator.pkl & ./model/discriminator.pkl ')
        print('Models save to ./model/generator_'+iter+'.pkl & ./model/discriminator_'+iter+'.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(self.C,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")
        
        
class MinorityDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe['data'].values
        self.labels = dataframe['label'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample, label

def stride(x):
    as_strided = np.lib.stride_tricks.as_strided
    # x = pd.Series(x) # no padding
    # print(len(x)) 1034 
    x = np.pad(pd.Series(x), (0, 2), 'constant') # padding 
    return as_strided(x, output_shape, (8*window_size, 8))
        
data = pd.read_csv('data/processed_data.csv')
minority_data = data[data['FLAG'] == 1]
majority_data = data[data['FLAG'] == 0]
minority_data.shape, majority_data.shape

y_minor = minority_data['FLAG']
x_minor = minority_data.drop(columns=['FLAG', 'CONS_NO'])

x_minor['data'] = x_minor[x_minor.columns].values.tolist()
df_apr = x_minor['data']


window_size = 28 # 4weeks
output_shape = (1036 // window_size, window_size)
strided_size = output_shape[0]

df_apr = df_apr.apply(stride)
df_dpr = df_apr.to_frame()
df_dpr['label'] = y_minor

# Assuming df_dpr is your DataFrame
minor_dataset = MinorityDataset(df_dpr)
minority_dataloader = DataLoader(minor_dataset, batch_size=64, shuffle=True)

model = WGAN_GP(len(minority_dataloader))

generated_data = model.generate_synthetic_samples(minority_dataloader)

print(generated_data)