from __future__ import print_function

import argparse
import os
import random
import numpy as np 
import pickle 
import torch
import torchvision
import torchvision.utils as vutils

import utils 
import data 
import nets
import nets_cond
import train_cond

parser = argparse.ArgumentParser()

###### Data arguments
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')

###### Model arguments
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)

###### Checkpointing and Logging arguments
parser.add_argument('--ckptG', type=str, default='ckpts/mnist_epoch_28.pth', help='a given checkpoint file for generator')
parser.add_argument('--ckptD', type=str, default='', help='a given checkpoint file for discriminator')

args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device 


#### defining generator
NC = 1 # hard-coded
netG = nets_cond.Generator(args.imageSize, args.nz, args.ngf, NC).to(device) # 
print('{} Generator: {}'.format("PresGAN", netG))

#### initialize weights
netG.apply(utils.weights_init)
if args.ckptG != '':
    netG.load_state_dict(torch.load(args.ckptG, map_location='cpu'))


#### generate samples
NUM_CLASS = 10
args.num_gen_images = 100


for i in range(NUM_CLASS):
    fixed_noise = torch.randn(args.num_gen_images, args.nz, 1, 1, device=device)
    class_label = torch.full((args.num_gen_images,), i, dtype=torch.long)
    y_one_hot = torch.eye(NUM_CLASS)[class_label, :]
    fake = netG(fixed_noise, y_one_hot).detach()

    for j in range(fake.size(0)):
        vutils.save_image(fake[j, :, :, :], 'samples/mnist_0/{}_{}.png'.format(i,j)) # normalize=True


# fixed_noise = torch.randn(args.num_gen_images, args.nz, 1, 1, device=device)
# rand_y_one_hot = torch.FloatTensor(args.num_gen_images, NUM_CLASS).zero_()
# rand_y_one_hot.scatter_(1, torch.randint(0, NUM_CLASS, size=(args.num_gen_images,1)), 1) 
# fake = netG(fixed_noise, rand_y_one_hot).detach()

# for i in range(fake.size(0)):
#     vutils.save_image(fake[i, :, :, :], 'samples/mnist_6/{}.png'.format(i)) # normalize=True


