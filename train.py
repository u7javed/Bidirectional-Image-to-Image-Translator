import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
import itertools
import time
import argparse

from models import *
from dataset import Image2ImageDataSet
from scheduler import Buffer, Lambda_LR

class Trainer():

    def __init__(self, train_dir_seg, train_dir_real, valid_dir_seg, valid_dir_real, epochs, lr, b1, b2, lambda_cycle, lambda_identity, 
                 batch_size, width, height, channels, sample_size, device):
        #load dataset
        self.train_dataset = Image2ImageDataSet(train_dir_seg, train_dir_real, width, height, sample_size)
        self.test_dataset = Image2ImageDataSet(valid_dir_seg, valid_dir_real, width, height)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=True)

        self.epochs = epochs
        self.channels = channels
        self.width = width
        self.height = height

        #instantiate models
        self.device = device
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        self.gen_seg_to_real = Generator(self.channels, width, height).to(self.device)
        self.gen_real_to_seg = Generator(self.channels, width, height).to(self.device)

        self.dis_seg = Discriminator(self.channels, width, height).to(self.device)
        self.dis_real = Discriminator(self.channels, width, height).to(self.device)

        self.GAN_loss_func = torch.nn.MSELoss().to(self.device)
        self.cycle_loss_func = torch.nn.L1Loss().to(self.device)
        self.identity_loss_func = torch.nn.L1Loss().to(self.device)

        self.optimizer_g = optim.Adam(itertools.chain(self.gen_seg_to_real.parameters(), self.gen_real_to_seg.parameters()), lr=lr, betas=(b1, b2))
        self.optimizer_d_seg = optim.Adam(self.dis_seg.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_d_real = optim.Adam(self.dis_real.parameters(), lr=lr, betas=(b1, b2))

        self.lr_scheduler_gen = optim.lr_scheduler.LambdaLR(self.optimizer_g, lr_lambda=Lambda_LR(epochs, 0, epochs // 2).step)
        self.lr_scheduler_dis_seg = optim.lr_scheduler.LambdaLR(self.optimizer_d_seg, lr_lambda=Lambda_LR(epochs, 0, epochs // 2).step)
        self.lr_scheduler_dis_real = optim.lr_scheduler.LambdaLR(self.optimizer_d_real, lr_lambda=Lambda_LR(epochs, 0, epochs // 2).step)

        self.fake_seg_buffer = Buffer(self.device)
        self.fake_real_buffer = Buffer(self.device)


        

    def train(self, saved_image_directory, saved_model_directory):
        start_time = time.time()

        for epoch in range(self.epochs):
            cur_time = time.time()

            for i, (seg_image, real_image) in enumerate(self.train_loader):
                b_size = len(seg_image)

                seg_image = seg_image.to(self.device)
                real_image = real_image.to(self.device)

                real = torch.ones(b_size, self.channels, self.width // 16, self.height // 16).to(self.device)
                fake = torch.zeros(b_size, self.channels, self.width // 16, self.height // 16).to(self.device)

                #train Generator
                self.optimizer_g.zero_grad()

                #GAN LOSS

                #segmentation generator loss
                fake_seg = self.gen_real_to_seg(real_image)
                fake_seg_pred = self.dis_seg(fake_seg)
                g_loss_seg = self.GAN_loss_func(fake_seg_pred, real)

                #real image generator loss
                fake_image = self.gen_seg_to_real(seg_image)
                fake_image_pred = self.dis_real(fake_image)
                g_loss_real = self.GAN_loss_func(fake_image_pred, real)

                #mean of GAN losses
                mean_g_loss = (g_loss_real + g_loss_seg)*(1/2)

                #IDENTITY LOSS

                #identity loss on segmentation generator
                fake_seg = self.gen_real_to_seg(seg_image)
                identity_loss_seg = self.identity_loss_func(fake_seg, seg_image)

                #identity loss on real image generator
                fake_image = self.gen_seg_to_real(real_image)
                identity_loss_real = self.identity_loss_func(fake_image, real_image)

                #mean of identiy losses
                mean_identity_loss = (identity_loss_real + identity_loss_seg)*(1/2)

                #CYCLE LOSS

                #cycle loss on segmentation
                fake_seg = self.gen_real_to_seg(real_image)
                seg_cycle_loss = self.cycle_loss_func(fake_seg, seg_image)

                #cycle loss on real images
                fake_image = self.gen_seg_to_real(seg_image)
                real_cycle_loss = self.cycle_loss_func(fake_image, real_image)

                #mean of cycle loss
                mean_cycle_loss = (seg_cycle_loss + real_cycle_loss)*(1/2)

                #total Generator loss
                g_loss = mean_g_loss + (self.lambda_identity * mean_identity_loss) + (self.lambda_cycle * mean_cycle_loss)

                g_loss.backward()
                self.optimizer_g.step()

                #train segmentation Discriminator

                self.optimizer_d_seg.zero_grad()

                #real loss
                real_seg_pred = self.dis_seg(seg_image)
                d_seg_loss_real = self.GAN_loss_func(real_seg_pred, real)

                #fake loss
                fake_seg_ = self.fake_seg_buffer.augment(fake_seg)
                fake_seg_pred = self.dis_seg(fake_seg_)
                d_seg_loss_fake = self.GAN_loss_func(fake_seg_pred, fake)

                #mean of fake and real loss
                d_seg_loss = (d_seg_loss_fake + d_seg_loss_real)*(1/2)

                d_seg_loss.backward()
                self.optimizer_d_seg.step()

                #train real image Discriminator

                self.optimizer_d_real.zero_grad()

                #real loss
                real_image_pred = self.dis_real(real_image)
                d_image_loss_real = self.GAN_loss_func(real_image_pred, real)

                fake_image_ = self.fake_real_buffer.augment(fake_image)
                fake_image_pred = self.dis_real(fake_image_)
                d_image_loss_fake = self.GAN_loss_func(fake_image_pred, fake)

                #mean of fake and real loss
                d_image_loss = (d_image_loss_fake + d_image_loss_real)*(1/2)

                d_image_loss.backward()
                self.optimizer_d_real.step()

                #mean of discriminator losses
                d_loss = (d_seg_loss + d_image_loss)*(1/2)

                if i % 50 == 0:
                    print('[{}/{}][{}/{}],    Dis Loss: {:.4f},   Gen Loss: {:.4f},   GAN Loss: {:.4f},   Identity Loss: {:.4f},  Cycle Loss: {:.4f}\n'.format(
                        epoch, self.epochs, i, len(self.train_loader), d_loss.item(), g_loss.item(), mean_g_loss.item(), mean_identity_loss.item(), mean_cycle_loss.item()
                    ))

            #print process
            cur_time = time.time() - cur_time

            print('Epoch {} Finished!. Saved some samples to '.format(epoch), saved_image_directory)
            print('Time Taken for Epoch: {:.4f} seconds or {:.4f} hours. Estimated {:.4f} hours remaining.\n'.format(cur_time, cur_time/3600, (self.epochs-epoch)*(cur_time/3600)))

            #save models
            torch.save(self.gen_real_to_seg.state_dict(), saved_model_directory + '/real2seg_gen_{}.pt'.format(epoch))
            torch.save(self.gen_seg_to_real.state_dict(), saved_model_directory + '/seg2real_gen_{}.pt'.format(epoch))

            #save samples
            validation_segs, validation_images = next(iter(self.test_loader))

            #make real images from segmentations
            fake_reals = self.gen_seg_to_real(validation_segs.to(self.device))
            seg_to_real_grid = torchvision.utils.make_grid(torch.cat([validation_segs.to(self.device), fake_reals], 0).cpu().detach(), nrow=2, normalize=True)

            #make segmentations from real images
            fake_segs = self.gen_real_to_seg(validation_images.to(self.device))
            real_to_seg_grid = torchvision.utils.make_grid(torch.cat([validation_images.to(self.device), fake_segs], 0).cpu().detach(), nrow=2, normalize=True)

            fgrid = torch.cat([seg_to_real_grid, real_to_seg_grid], 1)

            _, plot = plt.subplots(figsize=(16, 16))
            plt.axis('off')
            plot.imshow(fgrid.permute(1, 2, 0))
            plt.savefig(saved_image_directory + '/epoch_{}_checkpoint.jpg'.format(epoch), bbox_inches='tight')

            self.lr_scheduler_gen.step()
            self.lr_scheduler_dis_seg.step()
            self.lr_scheduler_dis_real.step()

        final_time_taken = time.time() - start_time
        print('Training Finished! Time taken: {:.4f} hours'.format(final_time_taken/3600))
        return int(final_time_taken)

def main():
    parser = argparse.ArgumentParser(description='Hyperparameters for training CycleGAN')
    #hyperparameter loading
    parser.add_argument('--train_dir_seg', type=str, default='data', help='directory to training segmented data')
    parser.add_argument('--train_dir_real', type=str, default='data', help='directory to training real image data')
    parser.add_argument('--valid_dir_seg', type=str, default='data', help='directory to validation segmented data')
    parser.add_argument('--valid_dir_real', type=str, default='data', help='directory to validation real image data')
    parser.add_argument('--width', type=int, default=256, help='width of post-processed images')
    parser.add_argument('--height', type=int, default=256, help='height of post-processed images.')
    parser.add_argument('--channels', type=int, default=3, help='feature dimension. Usually RGB or grayscale')
    parser.add_argument('--sample_size', type=int, default=2000, help='limit the number of samples to use for training from training directories due to limited memory')
    parser.add_argument('--saved_image_directory', type=str, default='data/saved_images', help='directory to where image samples will be saved')
    parser.add_argument('--saved_model_directory', type=str, default='saved_models', help='directory to where model weights will be saved')
    parser.add_argument('--epochs', type=int, default=200, help='number of iterations of dataset through network for training')
    parser.add_argument('--batch_size', type=int, default=2, help='size of batches passed through networks at each step')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or gpu depending on availability and compatability')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate of models')
    parser.add_argument('--b1', type=float, default=0.5, help='initial beta coefficient for computing gradient averages and squares')
    parser.add_argument('--b2', type=float, default=0.999, help='second beta coefficient for computing gradient averages and sqaures')
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='coefficient factor for cycle loss')
    parser.add_argument('--lambda_identity', type=float, default=5.0, help='coefficient factor for identity loss')
    args = parser.parse_args()

    train_dir_seg = args.train_dir_seg
    train_dir_real = args.train_dir_real
    valid_dir_seg = args.valid_dir_seg
    valid_dir_real = args.valid_dir_real
    width = args.width
    height = args.height
    channels = args.channels
    sample_size = args.sample_size
    saved_image_directory = args.saved_image_directory
    saved_model_directory = args.saved_model_directory
    epochs = args.epochs
    batch_size = args.batch_size
    device = args.device
    lr = args.lr
    b1 = args.b1
    b2 = args.b2
    lambda_cycle = args.lambda_cycle
    lambda_identity = args.lambda_identity

    cyclegan = Trainer(train_dir_seg, train_dir_real, valid_dir_seg, valid_dir_real, epochs, lr, b1, b2, 
                        lambda_cycle, lambda_identity, batch_size, width, height, channels, sample_size, device)
    training_time = cyclegan.train(saved_image_directory, saved_model_directory)


if __name__ == "__main__":
    main()
