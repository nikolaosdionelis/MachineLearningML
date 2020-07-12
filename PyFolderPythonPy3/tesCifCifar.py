import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

# import os
import time

import math
import os.path

import argparse
import numpy as np
from tqdm import tqdm

# MNIST and CIFAR-10
# Density Estimation Experiments

# MNIST:
# train_img.py --data mnist --imagesize 28 --actnorm True --wd 0 --save experiments/mnist

# CIFAR10:
# train_img.py --data cifar10 --actnorm True --save experiments/cifar10

# train_img.py --data mnist
# --imagesize 28 --actnorm True --wd 0 --save experiments/mnist

import gc
import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as vdsets

from lib.resflow import ACT_FNS, ResidualFlow
import lib.datasets as datasets
import lib.optimizers as optim

import lib.utils as utils
import lib.layers as layers
import lib.layers.base as base_layers
from lib.lr_scheduler import CosineAnnealingWarmRestarts

# Arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    '--data', type=str, default='cifar10', choices=[
        'mnist',
        'cifar10',
        'svhn',
        'celebahq',
        'celeba_5bit',
        'imagenet32',
        'imagenet64',
    ]
)

# train_img.py --data mnist
# --imagesize 28 --actnorm True --wd 0 --save experiments/mnist

parser.add_argument('--dataroot', type=str, default='data')
# parser.add_argument('--imagesize', type=int, default=32)

parser.add_argument('--imagesize', type=int, default=32)
# parser.add_argument('--imagesize', type=int, default=28)

parser.add_argument('--nbits', type=int, default=8)  # Only used for celebahq
parser.add_argument('--block', type=str, choices=['resblock', 'coupling'], default='resblock')

parser.add_argument('--coeff', type=float, default=0.98)
parser.add_argument('--vnorms', type=str, default='2222')

# parser.add_argument('--n-lipschitz-iters', type=int, default=None)
parser.add_argument('--n-lipschitz-iters', type=int, default=None)

parser.add_argument('--sn-tol', type=float, default=1e-3)
parser.add_argument('--learn-p', type=eval, choices=[True, False], default=False)

parser.add_argument('--n-power-series', type=int, default=None)
parser.add_argument('--factor-out', type=eval, choices=[True, False], default=False)

parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='poisson')
parser.add_argument('--n-samples', type=int, default=1)

parser.add_argument('--n-exact-terms', type=int, default=2)
parser.add_argument('--var-reduc-lr', type=float, default=0)

parser.add_argument('--neumann-grad', type=eval, choices=[True, False], default=True)
parser.add_argument('--mem-eff', type=eval, choices=[True, False], default=True)

parser.add_argument('--act', type=str, choices=ACT_FNS.keys(), default='swish')
parser.add_argument('--idim', type=int, default=512)

parser.add_argument('--nblocks', type=str, default='16-16-16')
parser.add_argument('--squeeze-first', type=eval, default=False, choices=[True, False])

parser.add_argument('--actnorm', type=eval, default=True, choices=[True, False])
parser.add_argument('--fc-actnorm', type=eval, default=False, choices=[True, False])

parser.add_argument('--batchnorm', type=eval, default=False, choices=[True, False])
parser.add_argument('--dropout', type=float, default=0.)

parser.add_argument('--fc', type=eval, default=False, choices=[True, False])
parser.add_argument('--kernels', type=str, default='3-1-3')

parser.add_argument('--add-noise', type=eval, choices=[True, False], default=True)
parser.add_argument('--quadratic', type=eval, choices=[True, False], default=False)

parser.add_argument('--fc-end', type=eval, choices=[True, False], default=True)
parser.add_argument('--fc-idim', type=int, default=128)

parser.add_argument('--preact', type=eval, choices=[True, False], default=True)
parser.add_argument('--padding', type=int, default=0)

parser.add_argument('--first-resblock', type=eval, choices=[True, False], default=True)
parser.add_argument('--cdim', type=int, default=256)

parser.add_argument('--optimizer', type=str, choices=['adam', 'adamax', 'rmsprop', 'sgd'], default='adam')
parser.add_argument('--scheduler', type=eval, choices=[True, False], default=False)

# parser.add_argument('--nepochs', help='Number of epochs for training', type=int, default=1000)

# parser.add_argument('--nepochs', help='Number of epochs for training', type=int, default=1000)
parser.add_argument('--nepochs', help='Number of epochs for training', type=int, default=10000)

# parser.add_argument('--nepochs', help='Number of epochs for training', type=int, default=1000)
# parser.add_argument('--nepochs', help='Number of epochs for training', type=int, default=5)

# parser.add_argument('--nepochs', help='Number of epochs for training', type=int, default=1000)
# parser.add_argument('--batchsize', help='Minibatch size', type=int, default=64)

# parser.add_argument('--batchsize', help='Minibatch size', type=int, default=64)
# parser.add_argument('--batchsize', help='Minibatch size', type=int, default=32)

# parser.add_argument('--batchsize', help='Minibatch size', type=int, default=32)

# parser.add_argument('--batchsize', help='Minibatch size', type=int, default=32)
parser.add_argument('--batchsize', help='Minibatch size', type=int, default=32)

# parser.add_argument('--batchsize', help='Minibatch size', type=int, default=64)
# parser.add_argument('--batchsize', help='Minibatch size', type=int, default=64)

parser.add_argument('--batch_sizeM', help='Minibatch size', type=int, default=32)
# parser.add_argument('--batch_sizeM', help='Minibatch size', type=int, default=64)

# parser.add_argument('--batch_sizeM', help='Minibatch size', type=int, default=64)

# parser.add_argument('--batch_sizeM', help='Minibatch size', type=int, default=64)
# parser.add_argument('--batch_sizeM', help='Minibatch size', type=int, default=128)

# parser.add_argument('--batch_sizeM', help='Minibatch size', type=int, default=128)
# parser.add_argument('--batch_sizeM', help='Minibatch size', type=int, default=512)

parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--wd', help='Weight decay', type=float, default=0)

parser.add_argument('--warmup-iters', type=int, default=1000)
parser.add_argument('--annealing-iters', type=int, default=0)

parser.add_argument('--save', help='directory to save results', type=str, default='experiment1')
parser.add_argument('--val-batchsize', help='minibatch size', type=int, default=200)

# parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--seed', type=int, default=None)

parser.add_argument('--ema-val', type=eval, choices=[True, False], default=True)
parser.add_argument('--update-freq', type=int, default=1)

parser.add_argument('--task', type=str, choices=['density', 'classification', 'hybrid'], default='density')
parser.add_argument('--scale-dim', type=eval, choices=[True, False], default=False)

parser.add_argument('--rcrop-pad-mode', type=str, choices=['constant', 'reflect'], default='reflect')
parser.add_argument('--padding-dist', type=str, choices=['uniform', 'gaussian'], default='uniform')

# parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--begin-epoch', type=int, default=0)

# parser.add_argument('--resume', type=str, default=None)

# parser.add_argument('--resume', type=str, default=None)
# parser.add_argument('--resume', type=str, default='./mnist_resflow_16-16-16.pth')

# parser.add_argument('--resume', type=str, default='./mnist_resflow_16-16-16.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/theMostRecent.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/theMostRecent.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheMostRecent.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheMostRecent.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheMostRecent.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheMostRecent.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheMostRecent.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheMostRecent.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheMostRecent.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheMostRecent.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheMostRecent.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheMostRecent.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheTheMostRecent.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheTheMostRecent.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheTheTheMostRecent.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheTheTheMostRecent.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheTheTheMostRecent.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheTheTheMostRecent.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheTheTheTheMostRecent.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheTheTheTheMostRecent.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheTheTheTheTheMostRecent.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheTheTheTheTheMostRecent.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheTheTheTheTheMostRecent.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheTheTheTheTheMostRecent.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheTheTheTheTheMostRecent.pth')

# parser.add_argument('--resume', type=str, default=None)
# parser.add_argument('--resume', type=str, default='./experiment1/models/most_recent.pth')

# parser.add_argument('--resume', type=str, default=None)
# parser.add_argument('--resume', type=str, default=None)

# parser.add_argument('--resume', type=str, default='./experiment1/models/mMMRRRee931113rrreee931113rrreee93113rrree93113rrree93113.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/mMMRRRee9123rrreee9123rrreee9123rrree9123rrree9123.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/mMMRRRee9123rrreee9123rrreee9123rrree9123rrree9123.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/mMMRRRee9123rrreee9123rrreee9123rrree9123rrree9123.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/mMMRRRee9123rrreee9123rrreee9123rrree9123rrree9123.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/mMMRRRee9123rrreee9123rrreee9123rrree9123rrree9123.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/mMMRRRee9123rrreee9123rrreee9123rrree9123rrree9123.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/mMMRRRee9123rrreee9123rrreee9123rrree9123rrree9123.pth')
# parser.add_argument('--resume', type=str, default='./exps/mostRecent.pth')

# parser.add_argument('--resume', type=str, default='./exps/mostRecent.pth')

# parser.add_argument('--resume', type=str, default='./exps/mostRecent.pth')
# parser.add_argument('--resume', type=str, default='./exps/mostRecent.pth')

# parser.add_argument('--resume', type=str, default='./exps/mostRecent.pth')
# parser.add_argument('--resume', type=str, default='./exps2/nikNikMostRecent.pth')

# parser.add_argument('--resume', type=str, default='./exps2/nikNikMostRecent.pth')

# parser.add_argument('--resume', type=str, default='./exps2/nikNikMostRecent.pth')
# parser.add_argument('--resume', type=str, default='./exps2/nikNikMostRecent.pth')

# parser.add_argument('--resume', type=str, default='./exps2/nikNikMostRecent.pth')
# parser.add_argument('--resume', type=str, default='./expsPretrained/mnist_resflow_16-16-16.pth')

# parser.add_argument('--resume', type=str, default='./expsPretrained/mnist_resflow_16-16-16.pth')
# parser.add_argument('--resume', type=str, default='../expsPretrained/mnist_resflow_16-16-16.pth')

# parser.add_argument('--resume', type=str, default='../expsPretrained/mnist_resflow_16-16-16.pth')
# parser.add_argument('--resume', type=str, default='./expsPretrained/mnist_resflow_16-16-16.pth')

# parser.add_argument('--resume', type=str, default='./expsPretrained/mnist_resflow_16-16-16.pth')
# parser.add_argument('--resume', type=str, default='./exps/mostRecent.pth')

# parser.add_argument('--resume', type=str, default=None)

# parser.add_argument('--resume', type=str, default=None)
# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/most_recent.pth')

# parser.add_argument('--resume', type=str, default='./exps/mostRecent.pth')

# parser.add_argument('--resume', type=str, default='./exps/mostRecent.pth')
# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/most_recent.pth')

# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/most_recent.pth')

# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/most_recent.pth')
# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/most_recent.pth')

# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/most_recent.pth')
# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/00for5_for5_55000mostReRecent.pth')

# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/00for5_for5_55000mostReRecent.pth')

# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/00for5_for5_55000mostReRecent.pth')
# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/00for5_for5_55000mostReRecent.pth')

# parser.add_argument('--resume', type=str, default=None)
# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/000for7_for7mostReRecent.pth')

# os.path.join(args.save, 'models', 'forFor2_moMostFor2_recentFor2.pth'))
# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/forFor2_moMostFor2_recentFor2.pth')

# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/forFor2_moMostFor2_recentFor2.pth')
# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/foFoFor2_mosMostFoFor2_recRecentFoFor2.pth')

# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/foFoFor2_mosMostFoFor2_recRecentFoFor2.pth')

# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/foFoFor2_mosMostFoFor2_recRecentFoFor2.pth')
# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/foFoFor2_mosMostFoFor2_recRecentFoFor2.pth')

# use: os.path.join(args.save, 'models', 'for4_for4_MoostReecent.pth'))
# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/for4_for4_MoostReecent.pth')

# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/for4_for4_MoostReecent.pth')
# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/foFor4_foFor4_MoosstReeccent.pth')

# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/foFor4_foFor4_MoosstReeccent.pth')

# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/foFor4_foFor4_MoosstReeccent.pth')
# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/foFor4_foFor4_MoosstReeccent.pth')

# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/foFor4_foFor4_MoosstReeccent.pth')
# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/foFoFoFoForFor4_foForFor4_moMoForMoFor4_mosstReeccent.pth')

# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/foFoFoFoForFor4_foForFor4_moMoForMoFor4_mosstReeccent.pth')

# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/foFoFoFoForFor4_foForFor4_moMoForMoFor4_mosstReeccent.pth')
# parser.add_argument('--resume', type=str, default='./finalExpFor1/foFoFoFoFoForFor1_foFoForFor1_foFor1most_for1RRRecent.pth')

# parser.add_argument('--resume', type=str, default='./finalExpFor1/foFoFoFoFoForFor1_foFoForFor1_foFor1most_for1RRRecent.pth')
# parser.add_argument('--resume', type=str, default='./fromEddie_expFor2/00for5_for5_55000mostReRecent.pth')

# parser.add_argument('--resume', type=str, default='./fromEddie_expFor2/00for5_for5_55000mostReRecent.pth')
# parser.add_argument('--resume', type=str, default='./cifarTwoEpochs/most_recent.pth')

# parser.add_argument('--resume', type=str, default='./cifarTwoEpochs/most_recent.pth')
# parser.add_argument('--resume', type=str, default='./cifarFourEpochs/most_recent.pth')

# parser.add_argument('--resume', type=str, default='./cifarFourEpochs/most_recent.pth')
# parser.add_argument('--resume', type=str, default='./toTestCifar6/checkpt-0002.pth')

# parser.add_argument('--resume', type=str, default='./toTestCifar6/checkpt-0002.pth')

# parser.add_argument('--resume', type=str, default='./toTestCifar6/checkpt-0002.pth')
#parser.add_argument('--resume', type=str, default='./ciCiCiCifar1/checkpt-0035.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/moFor8_ModeModels/moMosFoFor8_MostReceReRecent.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/moFor8For8_ModeModels/moMosFoFor8For8_MostReceReRecent.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/moFor8For8_ModeModels/moMosFoFor8For8_MostReceReRecent.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/moFor8For8_ModeModels/moMosFoFor8For8_MostReceReRecent.pth')
parser.add_argument('--resume', type=str, default='./experiment1/moMoModeFor0Models/mosMoMosFor0MostFor0ReceRecent.pth')

# os.path.join(args.save, 'moMoModeFor0Models', 'mosMoMosFor0MostFor0ReceRecent.pth'))
# use: os.path.join(args.save, 'moMoModeFor0Models', 'mosMoMosFor0MostFor0ReceRecent.pth'))

# os.path.join(args.save, 'moFor8For8_ModeModels', 'moMosFoFor8For8_MostReceReRecent.pth'))
# use: os.path.join(args.save, 'moFor8For8_ModeModels', 'moMosFoFor8For8_MostReceReRecent.pth'))

# toTestCifar6
# checkpt-0002.pth

# use: cifarTwoEpochs
# we use: cifarTwoEpochs

# use: finalExpFor1
# we use: finalExpFor1

# use: os.path.join(args.save, 'models', '000for7_for7mostReRecent.pth'))
# parser.add_argument('--resume', type=str, default='./experiments/mnist/models/000for7_for7mostReRecent.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/mMMRRRee9123rrreee9123rrreee9123rrree9123rrree9123.pth')
# parser.add_argument('--resume', type=str, default='./exps/checkpt-0008.pth')

# parser.add_argument('--resume', type=str, default='./experiment1/models/mMMRRRee9123rrreee9123rrreee9123rrree9123rrree9123.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/mMMRRRee9123rrreee9123rrreee9123rrree9123rrree9123.pth')

'''
parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheTheTheTheMostRecent.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/theTheTheTheTheTheTheTheMostRecent.pth')
'''

# parser.add_argument('--resume', type=str, default=None)
# parser.add_argument('--resume', type=str, default=None)

# torch.save({
#    'state_dict': model.state_dict(),
#    'optimizer_state_dict': optimizer.state_dict(),
#    'args': args,
#    'ema': ema,
#    'test_bpd': test_bpd,
# }, os.path.join(args.save, 'models', 'mostMostMostRecent.pth'))

# parser.add_argument('--resume', type=str, default='./experiment1/models/mostMostMostRecent.pth')
# parser.add_argument('--resume', type=str, default='./experiment1/models/mmoostMostMostRecent.pth')

# parser.add_argument('--nworkers', type=int, default=4)

# parser.add_argument('--nworkers', type=int, default=4)
parser.add_argument('--nworkers', type=int, default=1)

# parser.add_argument('--nworkers', type=int, default=4)
# parser.add_argument('--nworkers', type=int, default=4)

parser.add_argument('--print-freq', help='Print progress every so iterations', type=int, default=20)
parser.add_argument('--vis-freq', help='Visualize progress every so iterations', type=int, default=100)

args = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

"""
nrand = 200
#nrand = 100
gen = DCGANGenerator(nrand)
"""

import torch.nn.init as init


class DCGANGenerator2(nn.Module):
    def __init__(self, nrand):
        super(DCGANGenerator2, self).__init__()

        """
        self.lin1 = nn.Linear(nrand, 4 * 4 * 512)
        init.xavier_uniform_(self.lin1.weight, gain=0.1)
        self.lin1bn = nn.BatchNorm1d(4 * 4 * 512)
        self.dc1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.dc1bn = nn.BatchNorm2d(256)
        self.dc2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dc2bn = nn.BatchNorm2d(128)
        self.dc3a = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dc3abn = nn.BatchNorm2d(64)
        self.dc3b = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        """

        self.lin1 = nn.Linear(nrand, 4 * 4 * 448)
        init.xavier_uniform_(self.lin1.weight, gain=0.1)
        self.lin1bn = nn.BatchNorm1d(4 * 4 * 448)
        self.dc1 = nn.ConvTranspose2d(448, 224, 4, stride=2, padding=1)
        self.dc1bn = nn.BatchNorm2d(224)
        self.dc2 = nn.ConvTranspose2d(224, 112, 4, stride=2, padding=1)
        self.dc2bn = nn.BatchNorm2d(112)
        self.dc3a = nn.ConvTranspose2d(112, 56, 4, stride=2, padding=1)
        self.dc3abn = nn.BatchNorm2d(56)
        self.dc3b = nn.Conv2d(56, 1, 3, stride=1, padding=1)

    def forward(self, z):
        h = F.relu(self.lin1bn(self.lin1(z)))
        # h = torch.reshape(h, (-1, 512, 4, 4))

        # h = torch.reshape(h, (-1, 512, 4, 4))
        h = torch.reshape(h, (-1, 448, 4, 4))

        # deconv stack
        h = F.relu(self.dc1bn(self.dc1(h)))
        h = F.relu(self.dc2bn(self.dc2(h)))
        h = F.relu(self.dc3abn(self.dc3a(h)))
        x = self.dc3b(h)

        return x


class DCGANGenerator(nn.Module):
    def __init__(self, nrand):
        super(DCGANGenerator, self).__init__()

        # self.lin1 = nn.Linear(nrand, 4*4*512)
        # self.lin1 = nn.Linear(nrand, 1024)

        self.lin1 = nn.Linear(nrand, 1024)
        # self.lin1 = nn.Linear(nrand, 896)

        # init.xavier_uniform_(self.lin1.weight, gain=0.1)
        # self.lin1bn = nn.BatchNorm1d(4*4*512)
        # self.lin1bn = nn.BatchNorm1d(1024)

        self.lin1bn = nn.BatchNorm1d(1024)
        # self.lin1bn = nn.BatchNorm1d(896)

        # self.lin2 = nn.Linear(1024, 4*4*512)
        # self.lin2 = nn.Linear(1024, 7*7*128)
        # self.lin2 = nn.Linear(1024, 4 * 4 * 256)

        self.lin2 = nn.Linear(1024, 4 * 4 * 256)
        # self.lin2 = nn.Linear(896, 4 * 4 * 224)

        # init.xavier_uniform_(self.lin2.weight, gain=0.1)
        # self.lin2bn = nn.BatchNorm1d(4*4*512)
        # self.lin2bn = nn.BatchNorm1d(7*7*128)
        # self.lin2bn = nn.BatchNorm1d(4 * 4 * 256)

        self.lin2bn = nn.BatchNorm1d(4 * 4 * 256)
        # self.lin2bn = nn.BatchNorm1d(4 * 4 * 224)

        self.dc1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dc1bn = nn.BatchNorm2d(128)

        self.dc2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dc2bn = nn.BatchNorm2d(64)

        self.dc3 = nn.Conv2d(64, 1, 4, stride=2, padding=1)

        """
        self.dc1 = nn.ConvTranspose2d(224, 112, 4, stride=2, padding=1)
        self.dc1bn = nn.BatchNorm2d(112)

        self.dc2 = nn.ConvTranspose2d(112, 56, 4, stride=2, padding=1)
        self.dc2bn = nn.BatchNorm2d(56)

        self.dc3 = nn.Conv2d(56, 1, 4, stride=2, padding=1)
        """

    def forward(self, z):
        h = F.relu(self.lin1bn(self.lin1(z)))
        h = F.relu(self.lin2bn(self.lin2(h)))

        # h = torch.reshape(h, (-1, 512, 4, 4))
        # h = torch.reshape(h, (-1, 256, 4, 4))

        h = torch.reshape(h, (-1, 256, 4, 4))
        # h = torch.reshape(h, (-1, 224, 4, 4))

        h = F.relu(self.dc1bn(self.dc1(h)))
        h = F.relu(self.dc2bn(self.dc2(h)))

        # x = self.dc3(h)
        # x = F.tanh(self.dc3(h))

        # x = F.tanh(self.dc3(h))
        x = torch.tanh(self.dc3(h))

        return x


class DCGANGenerator3(nn.Module):
    def __init__(self, nrand):
        super(DCGANGenerator3, self).__init__()

        """
        self.lin1 = nn.Linear(nrand, 1024)
        self.lin1bn = nn.BatchNorm1d(1024)

        self.lin2 = nn.Linear(1024, 4 * 4 * 256)
        self.lin2bn = nn.BatchNorm1d(4 * 4 * 256)

        self.dc1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dc1bn = nn.BatchNorm2d(128)

        self.dc2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dc2bn = nn.BatchNorm2d(64)

        self.dc3 = nn.Conv2d(64, 1, 4, stride=2, padding=1)
        """

        self.lin1 = nn.Linear(nrand, 1024)
        self.lin1bn = nn.BatchNorm1d(1024)

        self.lin2 = nn.Linear(1024, 14 * 14 * 256)
        self.lin2bn = nn.BatchNorm1d(14 * 14 * 256)

        self.dc1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dc1bn = nn.BatchNorm2d(128)

        self.dc2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dc2bn = nn.BatchNorm2d(64)

        self.dc3 = nn.Conv2d(64, 1, 4, stride=2, padding=1)

    def forward(self, z):
        h = F.relu(self.lin1bn(self.lin1(z)))
        h = F.relu(self.lin2bn(self.lin2(h)))

        # h = torch.reshape(h, (-1, 256, 4, 4))
        h = torch.reshape(h, (-1, 256, 14, 14))

        h = F.relu(self.dc1bn(self.dc1(h)))
        h = F.relu(self.dc2bn(self.dc2(h)))

        # x = self.dc3(h)
        x = torch.tanh(self.dc3(h))

        return x


# G(z)
class DCGANgeneratorDCGAN(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(DCGANgeneratorDCGAN, self).__init__()

        """
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        """

        self.deconv1 = nn.ConvTranspose2d(100, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = F.tanh(self.deconv5(x))
        x = torch.tanh(self.deconv5(x))

        return x


# G(z)
class GANgeneratorGAN(nn.Module):
    # initializers

    # initializers
    def __init__(self, input_size=32, n_class=10):
        super(GANgeneratorGAN, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)

        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)

        # x = F.tanh(self.fc4(x))
        x = torch.tanh(self.fc4(x))

        return x


class GAN1generatorGAN1(nn.Module):
    # initializers

    # initializers
    def __init__(self, input_size=32, n_class=10):
        super(GAN1generatorGAN1, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        # x += input

        # x = F.leaky_relu(self.fc2(x), 0.2)
        # x = F.leaky_relu(self.fc3(x), 0.2)

        x2 = F.leaky_relu(self.fc2(x), 0.2)
        # x2 += x

        x3 = F.leaky_relu(self.fc3(x2), 0.2)
        # x3 += x2

        # x = torch.tanh(self.fc4(x))
        # x4 = torch.tanh(self.fc4(x3))

        x4 = torch.tanh(self.fc4(x3))
        x4 = x4 + input

        # return x
        return x4


class GAN2generatorGAN2(nn.Module):
    # initializers
    def __init__(self, input_size=32, n_class=10):
        super(GAN2generatorGAN2, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc1bn = nn.BatchNorm1d(256, momentum=0.8)

        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc2bn = nn.BatchNorm1d(512, momentum=0.8)

        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc3bn = nn.BatchNorm1d(1024, momentum=0.8)

        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    # forward method
    def forward(self, input):
        # x = F.leaky_relu(self.fc1(input), 0.2)
        x = self.fc1bn(F.leaky_relu(self.fc1(input), 0.2))

        # x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc2bn(F.leaky_relu(self.fc2(x), 0.2))

        # x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.fc3bn(F.leaky_relu(self.fc3(x), 0.2))

        # x = F.tanh(self.fc4(x))
        x = torch.tanh(self.fc4(x))

        return x


"""
# torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# use: torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

# torch.nn.BatchNorm1d(num_features, momentum=0.8)
# use: torch.nn.BatchNorm1d(num_features, momentum=0.8)

def build_generator(self):
    noise_shape = (100,)

    model = Sequential()

    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(self.img_shape), activation='tanh'))
    model.add(Reshape(self.img_shape))

    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)

    return Model(noise, img)
"""


# # FenceGAN Model
# def get_generative():
#    G_in = Input(shape=(2,))
#
#    x = Dense(10, activation='relu')(G_in)
#    x = Dense(10, activation='relu')(x)
#
#    #G_out = Dense(2)(x)
#
#    x = Dense(2)(x)
#    G_out = Add()([G_in,x])
#    # Res, Residual connection
#
#    G = Model(G_in, G_out)
#    return G

def loss_fn2(genFGen2, args, model):
    """
    first_term_loss = compute_loss2(genFGen2, args, model)

    #first_term_loss2 = compute_loss2(genFGen2, args, model)
    #first_term_loss = torch.log(first_term_loss2)

    #print('')
    #print(first_term_loss)

    #mu = torch.from_numpy(np.array([2.805741, -0.00889241], dtype="float32")).to(device)
    #S = torch.from_numpy(np.array([[pow(0.3442525,2), 0.0], [0.0, pow(0.35358343,2)]], dtype="float32")).to(device)

    #storeAll = torch.from_numpy(np.array(0.0, dtype="float32")).to(device)
    #toUse_storeAll = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=S)
    #for loopIndex_i in range(genFGen2.size()[0]):
    #    storeAll += torch.exp(toUse_storeAll.log_prob(genFGen2[loopIndex_i:1 + loopIndex_i, :].squeeze(0)))
    #storeAll /= genFGen2.size()[0]

    #print(storeAll)
    #print('')

    #print('')
    #print(compute_loss2(mu.unsqueeze(0), args, model))

    #print(torch.exp(toUse_storeAll.log_prob(mu)))
    #print('')

    #first_term_loss = storeAll

    xData = toy_data.inf_train_gen(args.data, batch_size=args.batch_size)
    xData = torch.from_numpy(xData).type(torch.float32).to(device)

    #var2 = []
    #for i in genFGen2:
    #    var1 = []
    #    for j in xData:
    #        new_stuff = torch.dist(i, j, 2)  # this is a tensor
    #        var1.append(new_stuff.unsqueeze(0))
    #    var1_tensor = torch.cat(var1)
    #    second_term_loss2 = torch.min(var1_tensor) / args.batch_size
    #    var2.append(second_term_loss2.unsqueeze(0))
    #var2_tensor = torch.cat(var2)
    #second_term_loss = torch.mean(var2_tensor) / args.batch_size
    #second_term_loss *= 100.0

    #print('')
    #print(second_term_loss)

    # If you know in advance the size of the final tensor, you can allocate
    # an empty tensor beforehand and fill it in the for loop.

    #x = torch.empty(size=(len(items), 768))
    #for i in range(len(items)):
    #    x[i] = calc_result

    #print(len(genFGen2))
    #print(genFGen2.shape[0])
    # len(.) and not .shape[0]

    #print(len(xData))
    #print(xData.shape[0])
    # Use len(.) and not .shape[0]

    #second_term_loss = torch.empty(size=(len(genFGen2), len(xData))).to(device)
    #second_term_loss = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=True)
    #second_term_loss3 = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=True)
    second_term_loss3 = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=False)
    for i in range(len(genFGen2)):
        for j in range(len(xData)):
            #second_term_loss[i, j] = torch.dist(genFGen2[i,:], xData[j,:], 2)
            #second_term_loss[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.tensor(0.1, requires_grad=True)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1).requires_grad_()

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1).requires_grad_()
            second_term_loss3[i, j] = (torch.dist(genFGen2[i, :], xData[j, :], 2)**2).requires_grad_()

            #second_term_loss[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 2)**2
    #second_term_loss2, _ = torch.min(second_term_loss, 1)
    second_term_loss2, _ = torch.min(second_term_loss3, 1)
    second_term_loss = 500000.0 * torch.mean(second_term_loss2) / (args.batch_size**2)
    #second_term_loss = torch.atan(torch.mean(second_term_loss2) / (args.batch_size ** 2)) / (0.5 * math.pi)

    #print(second_term_loss)
    #print('')

    print('')
    print(first_term_loss)
    print(second_term_loss)

    #third_term_loss = torch.from_numpy(np.array(0.0, dtype='float32')).to(device)
    #for i in range(args.batch_size):
    #    for j in range(args.batch_size):
    #        if i != j:
    #            # third_term_loss += ((np.linalg.norm(genFGen3[i,:].cpu().detach().numpy()-genFGen3[j,:].cpu().detach().numpy())) / (np.linalg.norm(genFGen2[i,:].cpu().detach().numpy()-genFGen2[j,:].cpu().detach().numpy())))
    #
    #            # third_term_loss += ((torch.norm(genFGen3[i,:]-genFGen3[j,:], 2)) / (torch.norm(genFGen2[i,:]-genFGen2[j,:], 2)))
    #            # third_term_loss += ((torch.norm(genFGen3[i,:]-genFGen3[j,:])) / (torch.norm(genFGen2[i,:]-genFGen2[j,:])))
    #
    #            # third_term_loss += ((torch.norm(genFGen3[i,:] - genFGen3[j,:])) / (torch.norm(genFGen2[i,:] - genFGen2[j,:])))
    #            third_term_loss += ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2)) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2)))
    #    third_term_loss /= (args.batch_size - 1)
    #third_term_loss /= args.batch_size
    ##third_term_loss *= 1000.0

    genFGen3 = torch.randn([args.batch_size, 2], device=device, requires_grad=True)
    #third_term_loss = torch.from_numpy(np.array(0.0, dtype='float32')).to(device)
    third_term_loss3 = torch.empty(size=(args.batch_size, args.batch_size), device=device, requires_grad=False)
    for i in range(args.batch_size):
        for j in range(args.batch_size):
            if i != j:
                # third_term_loss += ((np.linalg.norm(genFGen3[i,:].cpu().detach().numpy()-genFGen3[j,:].cpu().detach().numpy())) / (np.linalg.norm(genFGen2[i,:].cpu().detach().numpy()-genFGen2[j,:].cpu().detach().numpy())))

                # third_term_loss += ((torch.norm(genFGen3[i,:]-genFGen3[j,:], 2)) / (torch.norm(genFGen2[i,:]-genFGen2[j,:], 2)))
                # third_term_loss += ((torch.norm(genFGen3[i,:]-genFGen3[j,:])) / (torch.norm(genFGen2[i,:]-genFGen2[j,:])))

                # third_term_loss += ((torch.norm(genFGen3[i,:] - genFGen3[j,:])) / (torch.norm(genFGen2[i,:] - genFGen2[j,:])))
                #third_term_loss += ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2)) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2)))

                #third_term_loss += ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2)) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2)))
                #third_term_loss3[i][j] = ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2).requires_grad_()) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2).requires_grad_()))

                third_term_loss3[i][j] = ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2).requires_grad_()) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2).requires_grad_()))
    #third_term_loss /= (args.batch_size - 1)
    #third_term_loss2 = third_term_loss3 / (args.batch_size - 1)
    third_term_loss2 = torch.mean(third_term_loss3, 1)
    #third_term_loss /= args.batch_size
    #third_term_loss = third_term_loss2 / args.batch_size
    third_term_loss = torch.mean(third_term_loss2)
    #third_term_loss *= 1000.0

    print(third_term_loss)
    print('')

    #return first_term_loss + second_term_loss + third_term_loss
    #return first_term_loss + second_term_loss

    #return second_term_loss
    #return first_term_loss + second_term_loss
    return first_term_loss + second_term_loss + third_term_loss
    """

    first_term_loss = compute_loss2(genFGen2, args, model)
    # first_term_loss2 = compute_loss2(genFGen2, args, model)
    # first_term_loss = torch.log(first_term_loss2 / (1.0 - first_term_loss2))

    # print('')
    # print(first_term_loss)

    # mu = torch.from_numpy(np.array([2.805741, -0.00889241], dtype="float32")).to(device)
    # S = torch.from_numpy(np.array([[pow(0.3442525,2), 0.0], [0.0, pow(0.35358343,2)]], dtype="float32")).to(device)

    # mu = torch.from_numpy(np.array([0.0, 0.0], dtype="float32")).to(device)
    # S = torch.from_numpy(np.array([[pow(1.0,2), 0.0], [0.0, pow(1.0,2)]], dtype="float32")).to(device)

    # storeAll = torch.from_numpy(np.array(0.0, dtype="float32")).to(device)
    # toUse_storeAll = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=S)
    # for loopIndex_i in range(genFGen2.size()[0]):
    #    storeAll += torch.exp(toUse_storeAll.log_prob(genFGen2[loopIndex_i:1 + loopIndex_i, :].squeeze(0)))
    # storeAll /= genFGen2.size()[0]

    # print(storeAll)
    # print('')

    # print('')
    # print(compute_loss2(mu.unsqueeze(0), args, model))

    # print(torch.exp(toUse_storeAll.log_prob(mu)))
    # print('')

    # first_term_loss = storeAll

    xData = toy_data.inf_train_gen(args.data, batch_size=args.batch_size)
    xData = torch.from_numpy(xData).type(torch.float32).to(device)

    # var2 = []
    # for i in genFGen2:
    #    var1 = []
    #    for j in xData:
    #        new_stuff = torch.dist(i, j, 2)  # this is a tensor
    #        var1.append(new_stuff.unsqueeze(0))
    #    var1_tensor = torch.cat(var1)
    #    second_term_loss2 = torch.min(var1_tensor) / args.batch_size
    #    var2.append(second_term_loss2.unsqueeze(0))
    # var2_tensor = torch.cat(var2)
    # second_term_loss = torch.mean(var2_tensor) / args.batch_size
    # second_term_loss *= 100.0

    # print('')
    # print(second_term_loss)

    # If you know in advance the size of the final tensor, you can allocate
    # an empty tensor beforehand and fill it in the for loop.

    # x = torch.empty(size=(len(items), 768))
    # for i in range(len(items)):
    #    x[i] = calc_result

    # print(len(genFGen2))
    # print(genFGen2.shape[0])
    # len(.) and not .shape[0]

    # print(len(xData))
    # print(xData.shape[0])
    # Use len(.) and not .shape[0]

    """
    #second_term_loss = torch.empty(size=(len(genFGen2), len(xData))).to(device)
    #second_term_loss = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=True)
    #second_term_loss3 = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=True)
    #second_term_loss3 = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=False)
    second_term_loss3 = torch.empty(size=(args.batch_size, args.batch_size), device=device, requires_grad=False)
    #for i in range(len(genFGen2)):
    for i in range(args.batch_size):
        #for j in range(len(xData)):
        for j in range(args.batch_size):
            #second_term_loss[i, j] = torch.dist(genFGen2[i,:], xData[j,:], 2)
            #second_term_loss[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.tensor(0.1, requires_grad=True)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1).requires_grad_()

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1).requires_grad_()
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 2).requires_grad_()**2

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 2).requires_grad_()**2
            second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 2).requires_grad_()

            #second_term_loss[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 2)**2
    #second_term_loss2, _ = torch.min(second_term_loss, 1)
    second_term_loss2, _ = torch.min(second_term_loss3, 1)
    #second_term_loss = 5000.0 * torch.mean(second_term_loss2) / (args.batch_size**2)
    #second_term_loss = lambda1 * torch.mean(second_term_loss2) / (args.batch_size ** 2)
    #second_term_loss = lambda1 * torch.mean(second_term_loss2)
    second_term_loss = torch.mean(second_term_loss2)

    #print(second_term_loss)
    #print('')

    print('')
    print(first_term_loss)
    print(second_term_loss)

    print('')
    """

    second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    for i in range(args.batch_size):
        # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p='fro', dim=1).requires_grad_()
        # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()
        second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_() ** 2
        # print(second_term_loss22.shape)
        second_term_loss32[i] = torch.min(second_term_loss22)
    # print(second_term_loss32)
    # print(second_term_loss32.shape)
    # print(torch.norm(genFGen2 - xData, p=None, dim=0).shape)
    # second_term_loss22 = torch.min(second_term_loss32)
    # print(second_term_loss22)
    # print(second_term_loss22.shape)
    second_term_loss2 = torch.mean(second_term_loss32)
    # second_term_loss2 = 7.62939453125 * torch.mean(second_term_loss32)
    # print(second_term_loss2)
    # print(second_term_loss2.shape)

    print('')
    print(first_term_loss)
    print(second_term_loss2)

    """
    #second_term_loss = torch.empty(size=(len(genFGen2), len(xData))).to(device)
    #second_term_loss = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=True)
    #second_term_loss3 = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=True)
    second_term_loss3 = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=False)
    for i in range(len(genFGen2)):
        for j in range(len(xData)):
            #second_term_loss[i, j] = torch.dist(genFGen2[i,:], xData[j,:], 2)
            #second_term_loss[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.tensor(0.1, requires_grad=True)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1).requires_grad_()

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1).requires_grad_()
            second_term_loss3[i, j] = (torch.dist(genFGen2[i, :], xData[j, :], 2)**2).requires_grad_()

            #second_term_loss[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 2)**2
    #second_term_loss2, _ = torch.min(second_term_loss, 1)
    second_term_loss2, _ = torch.min(second_term_loss3, 1)
    second_term_loss = 500000.0 * torch.mean(second_term_loss2) / (args.batch_size**2)
    #second_term_loss = torch.atan(torch.mean(second_term_loss2) / (args.batch_size ** 2)) / (0.5 * math.pi)
    """

    # print(second_term_loss)
    # print('')

    # print('')
    # print(second_term_loss)
    # print(second_term_loss2)

    # print('')
    # print(first_term_loss)
    # print(second_term_loss2)

    # third_term_loss = torch.from_numpy(np.array(0.0, dtype='float32')).to(device)
    # for i in range(args.batch_size):
    #    for j in range(args.batch_size):
    #        if i != j:
    #            # third_term_loss += ((np.linalg.norm(genFGen3[i,:].cpu().detach().numpy()-genFGen3[j,:].cpu().detach().numpy())) / (np.linalg.norm(genFGen2[i,:].cpu().detach().numpy()-genFGen2[j,:].cpu().detach().numpy())))
    #
    #            # third_term_loss += ((torch.norm(genFGen3[i,:]-genFGen3[j,:], 2)) / (torch.norm(genFGen2[i,:]-genFGen2[j,:], 2)))
    #            # third_term_loss += ((torch.norm(genFGen3[i,:]-genFGen3[j,:])) / (torch.norm(genFGen2[i,:]-genFGen2[j,:])))
    #
    #            # third_term_loss += ((torch.norm(genFGen3[i,:] - genFGen3[j,:])) / (torch.norm(genFGen2[i,:] - genFGen2[j,:])))
    #            third_term_loss += ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2)) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2)))
    #    third_term_loss /= (args.batch_size - 1)
    # third_term_loss /= args.batch_size
    ##third_term_loss *= 1000.0

    """
    genFGen3 = torch.randn([args.batch_size, 2], device=device, requires_grad=True)
    #third_term_loss = torch.from_numpy(np.array(0.0, dtype='float32')).to(device)
    third_term_loss3 = torch.empty(size=(args.batch_size, args.batch_size), device=device, requires_grad=False)
    for i in range(args.batch_size):
        for j in range(args.batch_size):
            if i != j:
                # third_term_loss += ((np.linalg.norm(genFGen3[i,:].cpu().detach().numpy()-genFGen3[j,:].cpu().detach().numpy())) / (np.linalg.norm(genFGen2[i,:].cpu().detach().numpy()-genFGen2[j,:].cpu().detach().numpy())))

                # third_term_loss += ((torch.norm(genFGen3[i,:]-genFGen3[j,:], 2)) / (torch.norm(genFGen2[i,:]-genFGen2[j,:], 2)))
                # third_term_loss += ((torch.norm(genFGen3[i,:]-genFGen3[j,:])) / (torch.norm(genFGen2[i,:]-genFGen2[j,:])))

                # third_term_loss += ((torch.norm(genFGen3[i,:] - genFGen3[j,:])) / (torch.norm(genFGen2[i,:] - genFGen2[j,:])))
                #third_term_loss += ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2)) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2)))

                #third_term_loss += ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2)) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2)))
                #third_term_loss3[i][j] = ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2).requires_grad_()) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2).requires_grad_()))

                third_term_loss3[i][j] = ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2).requires_grad_()) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2).requires_grad_()))
    #third_term_loss /= (args.batch_size - 1)
    #third_term_loss2 = third_term_loss3 / (args.batch_size - 1)
    third_term_loss2 = torch.mean(third_term_loss3, 1)
    #third_term_loss /= args.batch_size
    #third_term_loss = third_term_loss2 / args.batch_size
    #third_term_loss = torch.mean(third_term_loss2)
    #third_term_loss = 0.01 * torch.mean(third_term_loss2)
    #third_term_loss = lambda2 * torch.mean(third_term_loss2)
    third_term_loss = torch.mean(third_term_loss2)
    #third_term_loss *= 1000.0

    print(third_term_loss)
    print('')
    """

    genFGen3 = torch.randn([args.batch_size, 2], device=device, requires_grad=True)
    third_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    for i in range(args.batch_size):
        # third_term_loss22 = (torch.norm(genFGen3[i, :] - genFGen3, p='fro', dim=1).requires_grad_()) / (1.0e-32+torch.norm(genFGen2[i, :] - genFGen2, p='fro', dim=1).requires_grad_())
        # third_term_loss22 = (torch.norm(genFGen3[i, :] - genFGen3, p=None, dim=1).requires_grad_()) / (1.0e-32+torch.norm(genFGen2[i, :] - genFGen2, p=None, dim=1).requires_grad_())
        third_term_loss22 = (torch.norm(genFGen3[i, :] - genFGen3, p=None, dim=1).requires_grad_()) / (
                1.0e-17 + torch.norm(genFGen2[i, :] - genFGen2, p=None, dim=1).requires_grad_())
        # print(third_term_loss22.shape)
        third_term_loss32[i] = torch.mean(third_term_loss22)
    # print(third_term_loss32)
    # print(third_term_loss32.shape)
    # print(third_term_loss22)
    # print(third_term_loss22.shape)
    third_term_loss12 = torch.mean(third_term_loss32)
    # print(third_term_loss2)
    # print(third_term_loss12.shape)

    """
    #genFGen3 = torch.randn([args.batch_size, 2], device=device, requires_grad=True)
    #third_term_loss = torch.from_numpy(np.array(0.0, dtype='float32')).to(device)
    third_term_loss3 = torch.empty(size=(args.batch_size, args.batch_size), device=device, requires_grad=False)
    for i in range(args.batch_size):
        for j in range(args.batch_size):
            if i != j:
                # third_term_loss += ((np.linalg.norm(genFGen3[i,:].cpu().detach().numpy()-genFGen3[j,:].cpu().detach().numpy())) / (np.linalg.norm(genFGen2[i,:].cpu().detach().numpy()-genFGen2[j,:].cpu().detach().numpy())))

                # third_term_loss += ((torch.norm(genFGen3[i,:]-genFGen3[j,:], 2)) / (torch.norm(genFGen2[i,:]-genFGen2[j,:], 2)))
                # third_term_loss += ((torch.norm(genFGen3[i,:]-genFGen3[j,:])) / (torch.norm(genFGen2[i,:]-genFGen2[j,:])))

                # third_term_loss += ((torch.norm(genFGen3[i,:] - genFGen3[j,:])) / (torch.norm(genFGen2[i,:] - genFGen2[j,:])))
                #third_term_loss += ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2)) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2)))

                #third_term_loss += ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2)) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2)))
                #third_term_loss3[i][j] = ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2).requires_grad_()) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2).requires_grad_()))

                third_term_loss3[i][j] = ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2).requires_grad_()) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2).requires_grad_()))
    #third_term_loss /= (args.batch_size - 1)
    #third_term_loss2 = third_term_loss3 / (args.batch_size - 1)
    third_term_loss2 = torch.mean(third_term_loss3, 1)
    #third_term_loss /= args.batch_size
    #third_term_loss = third_term_loss2 / args.batch_size
    third_term_loss = torch.mean(third_term_loss2)
    #third_term_loss *= 1000.0
    """

    # print(third_term_loss)
    # print(third_term_loss12)

    print(third_term_loss12)
    print('')

    # print(third_term_loss12)
    # print(third_term_loss)
    # print('')

    # return first_term_loss + second_term_loss + third_term_loss
    # return first_term_loss + second_term_loss

    # return second_term_loss
    # return first_term_loss + second_term_loss
    # return first_term_loss + second_term_loss + third_term_loss

    # return first_term_loss + second_term_loss + third_term_loss
    # return first_term_loss + second_term_loss2 + third_term_loss
    # return first_term_loss + second_term_loss2 + third_term_loss12

    # return first_term_loss + second_term_loss2
    # return first_term_loss + second_term_loss2 + third_term_loss12

    # return first_term_loss + second_term_loss2 + third_term_loss12
    return first_term_loss + second_term_loss2 + third_term_loss12, xData


# def use_loss_fn2(genFGen2, args, model, genFGen3, toUse_storeAll, toUse_storeAll2):
# def use_loss_fn2(genFGen2, args, model, genFGen3):

# def use_loss_fn2(genFGen2, args, model, genFGen3):
def use_loss_fn2(genFGen2, args, model, genFGen3, xData):
    """
    first_term_loss = compute_loss2(genFGen2, args, model)

    #first_term_loss2 = compute_loss2(genFGen2, args, model)
    #first_term_loss = torch.log(first_term_loss2/(1.0-first_term_loss2))
    #first_term_loss = torch.log(first_term_loss2)

    #print('')
    #print(first_term_loss)

    #mu = torch.from_numpy(np.array([2.805741, -0.00889241], dtype="float32")).to(device)
    #S = torch.from_numpy(np.array([[pow(0.3442525,2), 0.0], [0.0, pow(0.35358343,2)]], dtype="float32")).to(device)

    #storeAll = torch.from_numpy(np.array(0.0, dtype="float32")).to(device)
    #toUse_storeAll = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=S)
    #for loopIndex_i in range(genFGen2.size()[0]):
    #    storeAll += torch.exp(toUse_storeAll.log_prob(genFGen2[loopIndex_i:1 + loopIndex_i, :].squeeze(0)))
    #storeAll /= genFGen2.size()[0]

    #print(storeAll)
    #print('')

    #print('')
    #print(compute_loss2(mu.unsqueeze(0), args, model))

    #print(torch.exp(toUse_storeAll.log_prob(mu)))
    #print('')

    #first_term_loss = storeAll

    xData = toy_data.inf_train_gen(args.data, batch_size=args.batch_size)
    xData = torch.from_numpy(xData).type(torch.float32).to(device)

    #var2 = []
    #for i in genFGen2:
    #    var1 = []
    #    for j in xData:
    #        new_stuff = torch.dist(i, j, 2)  # this is a tensor
    #        var1.append(new_stuff.unsqueeze(0))
    #    var1_tensor = torch.cat(var1)
    #    second_term_loss2 = torch.min(var1_tensor) / args.batch_size
    #    var2.append(second_term_loss2.unsqueeze(0))
    #var2_tensor = torch.cat(var2)
    #second_term_loss = torch.mean(var2_tensor) / args.batch_size
    #second_term_loss *= 100.0

    #print('')
    #print(second_term_loss)

    # If you know in advance the size of the final tensor, you can allocate
    # an empty tensor beforehand and fill it in the for loop.

    #x = torch.empty(size=(len(items), 768))
    #for i in range(len(items)):
    #    x[i] = calc_result

    #print(len(genFGen2))
    #print(genFGen2.shape[0])
    # len(.) and not .shape[0]

    #print(len(xData))
    #print(xData.shape[0])
    # Use len(.) and not .shape[0]

    #second_term_loss = torch.empty(size=(len(genFGen2), len(xData))).to(device)
    #second_term_loss = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=True)
    #second_term_loss3 = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=True)
    second_term_loss3 = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=False)
    for i in range(len(genFGen2)):
        for j in range(len(xData)):
            #second_term_loss[i, j] = torch.dist(genFGen2[i,:], xData[j,:], 2)
            #second_term_loss[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.tensor(0.1, requires_grad=True)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1).requires_grad_()

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1).requires_grad_()
            second_term_loss3[i, j] = (torch.dist(genFGen2[i, :], xData[j, :], 2)**2).requires_grad_()

            #second_term_loss[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 2)**2
    #second_term_loss2, _ = torch.min(second_term_loss, 1)
    second_term_loss2, _ = torch.min(second_term_loss3, 1)
    second_term_loss = 500000.0 * torch.mean(second_term_loss2) / (args.batch_size**2)
    #second_term_loss = torch.atan(torch.mean(second_term_loss2) / (args.batch_size ** 2)) / (0.5 * math.pi)

    #print(second_term_loss)
    #print('')

    print('')
    print(first_term_loss)
    print(second_term_loss)

    #third_term_loss = torch.from_numpy(np.array(0.0, dtype='float32')).to(device)
    #for i in range(args.batch_size):
    #    for j in range(args.batch_size):
    #        if i != j:
    #            # third_term_loss += ((np.linalg.norm(genFGen3[i,:].cpu().detach().numpy()-genFGen3[j,:].cpu().detach().numpy())) / (np.linalg.norm(genFGen2[i,:].cpu().detach().numpy()-genFGen2[j,:].cpu().detach().numpy())))
    #
    #            # third_term_loss += ((torch.norm(genFGen3[i,:]-genFGen3[j,:], 2)) / (torch.norm(genFGen2[i,:]-genFGen2[j,:], 2)))
    #            # third_term_loss += ((torch.norm(genFGen3[i,:]-genFGen3[j,:])) / (torch.norm(genFGen2[i,:]-genFGen2[j,:])))
    #
    #            # third_term_loss += ((torch.norm(genFGen3[i,:] - genFGen3[j,:])) / (torch.norm(genFGen2[i,:] - genFGen2[j,:])))
    #            third_term_loss += ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2)) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2)))
    #    third_term_loss /= (args.batch_size - 1)
    #third_term_loss /= args.batch_size
    ##third_term_loss *= 1000.0

    genFGen3 = torch.randn([args.batch_size, 2], device=device, requires_grad=True)
    #third_term_loss = torch.from_numpy(np.array(0.0, dtype='float32')).to(device)
    third_term_loss3 = torch.empty(size=(args.batch_size, args.batch_size), device=device, requires_grad=False)
    for i in range(args.batch_size):
        for j in range(args.batch_size):
            if i != j:
                # third_term_loss += ((np.linalg.norm(genFGen3[i,:].cpu().detach().numpy()-genFGen3[j,:].cpu().detach().numpy())) / (np.linalg.norm(genFGen2[i,:].cpu().detach().numpy()-genFGen2[j,:].cpu().detach().numpy())))

                # third_term_loss += ((torch.norm(genFGen3[i,:]-genFGen3[j,:], 2)) / (torch.norm(genFGen2[i,:]-genFGen2[j,:], 2)))
                # third_term_loss += ((torch.norm(genFGen3[i,:]-genFGen3[j,:])) / (torch.norm(genFGen2[i,:]-genFGen2[j,:])))

                # third_term_loss += ((torch.norm(genFGen3[i,:] - genFGen3[j,:])) / (torch.norm(genFGen2[i,:] - genFGen2[j,:])))
                #third_term_loss += ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2)) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2)))

                #third_term_loss += ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2)) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2)))
                #third_term_loss3[i][j] = ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2).requires_grad_()) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2).requires_grad_()))

                third_term_loss3[i][j] = ((torch.dist(genFGen3[i, :], genFGen3[j, :], 2).requires_grad_()) / (torch.dist(genFGen2[i, :], genFGen2[j, :], 2).requires_grad_()))
    #third_term_loss /= (args.batch_size - 1)
    #third_term_loss2 = third_term_loss3 / (args.batch_size - 1)
    third_term_loss2 = torch.mean(third_term_loss3, 1)
    #third_term_loss /= args.batch_size
    #third_term_loss = third_term_loss2 / args.batch_size
    third_term_loss = torch.mean(third_term_loss2)
    #third_term_loss *= 1000.0

    print(third_term_loss)
    print('')

    #return first_term_loss + second_term_loss + third_term_loss
    #return first_term_loss + second_term_loss

    #return second_term_loss
    #return first_term_loss + second_term_loss
    return first_term_loss + second_term_loss + third_term_loss
    """

    # first_term_loss = compute_loss2(genFGen2, args, model)
    # first_term_loss2 = compute_loss2(genFGen2, args, model)
    # first_term_loss = torch.log(first_term_loss2 / (1.0 - first_term_loss2))

    # first_term_loss = compute_loss2(genFGen2, args, model)

    # first_term_loss = compute_loss2(genFGen2, args, model)
    # first_term_loss = compute_loss2(genFGen2, args, model)

    # print('')
    # print(first_term_loss)

    # mu = torch.from_numpy(np.array([2.805741, -0.00889241], dtype="float32")).to(device)
    # S = torch.from_numpy(np.array([[pow(0.3442525,2), 0.0], [0.0, pow(0.35358343,2)]], dtype="float32")).to(device)

    # mu = torch.from_numpy(np.array([2.8093171, 1.2994107e-03], dtype="float32")).to(device)
    # S = torch.from_numpy(np.array([[pow(0.35840544, 2), 0.0], [0.0, pow(0.34766033, 2)]], dtype="float32")).to(device)

    # mu = torch.from_numpy(np.array([0.0, 0.0], dtype="float32")).to(device)
    # S = torch.from_numpy(np.array([[pow(1.0,2), 0.0], [0.0, pow(1.0,2)]], dtype="float32")).to(device)

    """
    #storeAll = torch.from_numpy(np.array(0.0, dtype="float32")).to(device)
    storeAll = torch.empty(args.batch_size, device=device, requires_grad=False)
    #toUse_storeAll = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=S)
    #for loopIndex_i in range(genFGen2.size()[0]):
    for loopIndex_i in range(args.batch_size):
        #storeAll += torch.exp(toUse_storeAll.log_prob(genFGen2[loopIndex_i:1 + loopIndex_i, :].squeeze(0)))
        #storeAll[loopIndex_i] = torch.exp(toUse_storeAll.log_prob(genFGen2[loopIndex_i:1 + loopIndex_i, :].squeeze(0)).requires_grad_())

        #storeAll[loopIndex_i] = torch.exp(
        #    toUse_storeAll.log_prob(genFGen2[loopIndex_i:1 + loopIndex_i, :].squeeze(0)).requires_grad_())

        storeAll[loopIndex_i] = 0.5 * torch.exp(toUse_storeAll.log_prob(genFGen2[loopIndex_i:1 + loopIndex_i, :].squeeze(0)).requires_grad_())\
                                + 0.5 * torch.exp(toUse_storeAll2.log_prob(genFGen2[loopIndex_i:1 + loopIndex_i, :].squeeze(0)).requires_grad_())
    #storeAll /= genFGen2.size()[0]
    first_term_loss = torch.mean(storeAll)
    """

    # print(first_term_loss)
    # first_term_loss = compute_loss2(genFGen2, args, model)

    # print(genFGen2)
    # dasfasdfs

    # first_term_loss = compute_loss2(genFGen2, args, model)
    # first_term_loss = compute_loss2(genFGen2, model)

    # print(xData.shape)
    # print(genFGen2.shape)

    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    imageStore = xData[0,:,:,:].squeeze().cpu().numpy()
    #imageStore = genFGen2[0, :, :, :].squeeze().cpu().detach().numpy()

    plt.imshow(imageStore)
    plt.show()
    """

    # pilTrans = transforms.ToTensor()
    # plt.imshow(xData[1, :])

    # first_term_loss = compute_loss2(genFGen2, model)
    # first_term_loss = compute_loss2(xData, model)

    # first_term_loss = compute_loss2(genFGen2, model)

    # first_term_loss = compute_loss2(genFGen2, model)
    # first_term_loss = compute_loss2(genFGen2, model)

    # first_term_loss = compute_loss2(genFGen2, model)

    # first_term_loss = compute_loss2(genFGen2, model)
    # first_term_loss = compute_loss2(xData, model)

    # print(xData)
    # print(genFGen2)

    # print(genFGen2.shape)
    # print(xData.shape)

    # print(compute_loss2(genFGen2, model))
    # print(compute_loss2(xData, model))

    # print(compute_loss(xData, model))
    # print(compute_loss(xData, model).item())

    # (tensor(0.9740, device='cuda:0', grad_fn=<DivBackward0>), tensor([0.], device='cuda:0'),
    # tensor(-1139.7253, device='cuda:0'), tensor(4957.8486, device='cuda:0'))

    # print(computeLoss(genFGen2, model))
    # print(computeLoss(xData, model))

    # first_term_loss = compute_loss2(genFGen2, model)
    # first_term_loss = compute_loss2(genFGen2, model)

    # first_term_loss = compute_loss2(genFGen2, model)

    # first_term_loss = compute_loss2(genFGen2, model)
    first_term_loss = computeLoss(genFGen2, model)

    # print(genFGen2.shape)
    # print(first_term_loss)

    # first_term_loss.retain_grad()

    # first_term_loss.retain_grad()
    # first_term_loss.retain_grad()

    # (?)
    # first_term_loss.retain_grad()
    # (?)

    # print(first_term_loss)
    # print('')

    """
    second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    for i in range(args.batch_size):
        second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_() ** 2
        second_term_loss32[i] = torch.min(second_term_loss22)
    second_term_loss2 = torch.mean(second_term_loss32)
    """

    # print(first_term_loss)
    # print('')

    # print('')
    # print(compute_loss2(mu.unsqueeze(0), args, model))

    # print(torch.exp(toUse_storeAll.log_prob(mu)))
    # print('')

    # first_term_loss = storeAll

    # xData = toy_data.inf_train_gen(args.data, batch_size=args.batch_size)
    # xData = torch.from_numpy(xData).type(torch.float32).to(device)

    # print(xData.shape)
    # print(torch.mean(xData))
    # print(torch.std(xData))

    # xData = torch.empty((args.batch_size, 2), device=device)
    # xData[:args.batch_size//2, :] = toUse_storeAll.sample((args.batch_size//2,)) # .sample_n(args.batch_size // 2)
    # xData[args.batch_size//2:, :] = toUse_storeAll2.sample((args.batch_size//2,)) # .sample_n(args.batch_size//2)

    """
    xData = torch.empty((args.batch_sizeM, 2), device=device)
    xData[:args.batch_sizeM // 2, :] = toUse_storeAll.sample((args.batch_sizeM // 2,))  # .sample_n(args.batch_size // 2)
    xData[args.batch_sizeM // 2:, :] = toUse_storeAll2.sample((args.batch_sizeM // 2,))  # .sample_n(args.batch_size//2)
    """

    # xData = torch.empty((args.batch_size, 2)).normal_(mean=[2.82507515, 1.92882611e-04 + 0.8], std=0.5)
    # xData[args.batch_size//2:,:] = torch.empty((args.batch_size, 2)).normal_(mean=4, std=0.5)

    # mu = torch.from_numpy(np.array([2.82507515, 1.92882611e-04 + 0.8], dtype="float32")).to(device)
    # S = torch.from_numpy(np.array([[pow(0.07166782, 2), 0.0], [0.0, pow(0.06917527, 2)]], dtype="float32")).to(device)
    # mu2 = torch.from_numpy(np.array([2.82507515, 1.92882611e-04 - 0.8], dtype="float32")).to(device)
    # toUse_storeAll = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=S)
    # toUse_storeAll2 = torch.distributions.MultivariateNormal(loc=mu2, covariance_matrix=S)

    # print(xData.shape)
    # print(torch.mean(xData))
    # print(torch.std(xData))

    # var2 = []
    # for i in genFGen2:
    #    var1 = []
    #    for j in xData:
    #        new_stuff = torch.dist(i, j, 2)  # this is a tensor
    #        var1.append(new_stuff.unsqueeze(0))
    #    var1_tensor = torch.cat(var1)
    #    second_term_loss2 = torch.min(var1_tensor) / args.batch_size
    #    var2.append(second_term_loss2.unsqueeze(0))
    # var2_tensor = torch.cat(var2)
    # second_term_loss = torch.mean(var2_tensor) / args.batch_size
    # second_term_loss *= 100.0

    # print('')
    # print(second_term_loss)

    # If you know in advance the size of the final tensor, you can allocate
    # an empty tensor beforehand and fill it in the for loop.

    # x = torch.empty(size=(len(items), 768))
    # for i in range(len(items)):
    #    x[i] = calc_result

    # print(len(genFGen2))
    # print(genFGen2.shape[0])
    # len(.) and not .shape[0]

    # print(len(xData))
    # print(xData.shape[0])
    # Use len(.) and not .shape[0]

    """
    #second_term_loss = torch.empty(size=(len(genFGen2), len(xData))).to(device)
    #second_term_loss = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=True)
    #second_term_loss3 = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=True)
    #second_term_loss3 = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=False)
    second_term_loss3 = torch.empty(size=(args.batch_size, args.batch_size), device=device, requires_grad=False)
    #for i in range(len(genFGen2)):
    for i in range(args.batch_size):
        #for j in range(len(xData)):
        for j in range(args.batch_size):
            #second_term_loss[i, j] = torch.dist(genFGen2[i,:], xData[j,:], 2)
            #second_term_loss[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.tensor(0.1, requires_grad=True)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1).requires_grad_()

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1).requires_grad_()
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 2).requires_grad_()**2

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 2).requires_grad_()**2
            second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 2).requires_grad_()

            #second_term_loss[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 2)**2
    #second_term_loss2, _ = torch.min(second_term_loss, 1)
    second_term_loss2, _ = torch.min(second_term_loss3, 1)
    #second_term_loss = 5000.0 * torch.mean(second_term_loss2) / (args.batch_size**2)
    #second_term_loss = lambda1 * torch.mean(second_term_loss2) / (args.batch_size ** 2)
    #second_term_loss = lambda1 * torch.mean(second_term_loss2)
    second_term_loss = torch.mean(second_term_loss2)

    #print(second_term_loss)
    #print('')

    print('')
    print(first_term_loss)
    print(second_term_loss)

    print('')
    """

    # args.batch_size = 2
    # genFGen2 = torch.from_numpy(np.array([[3, 0], [2, 0]], dtype="float32")).to(device)
    # xData = torch.from_numpy(np.array([[1, 0], [0, 1]], dtype="float32")).to(device)

    # import timeit
    # start = timeit.default_timer()
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)

    """
    second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    for i in range(args.batch_size):
        #second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p='fro', dim=1).requires_grad_()
        #second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()
        second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()**2
        #second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()
        #print(second_term_loss22.shape)
        second_term_loss32[i] = torch.min(second_term_loss22)
    #print(second_term_loss32)
    #print(second_term_loss32.shape)
    #print(torch.norm(genFGen2 - xData, p=None, dim=0).shape)
    #second_term_loss22 = torch.min(second_term_loss32)
    #print(second_term_loss22)
    #print(second_term_loss22.shape)
    second_term_loss2 = torch.mean(second_term_loss32)
    #second_term_loss2 = 7.62939453125 * torch.mean(second_term_loss32)
    #print(second_term_loss2)
    #print(second_term_loss2.shape)
    """

    # import timeit
    # start = timeit.default_timer()
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)

    # print('')
    # print(second_term_loss2)

    # distances = torch.norm(vertices - point_locs, p=2, dim=1)
    # distances = torch.sqrt((vertices - point_locs).pow(2).sum(1))

    # import timeit
    # start = timeit.default_timer()
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)

    xData = xData.view(-1, 28 * 28)
    genFGen2 = genFGen2.view(-1, 28 * 28)

    # genFGen2 = genFGen2.view(-1, 28*28)
    genFGen3 = genFGen3.view(-1, 28 * 28)

    # xData = torch.transpose(xData, 0, 1)
    # genFGen2 = torch.transpose(genFGen2, 0, 1)

    # genFGen2 = torch.transpose(genFGen2, 0, 1)
    # genFGen3 = torch.transpose(genFGen3, 0, 1)

    # print(genFGen2.shape)
    # print(xData.shape)
    # print(genFGen3.shape)

    # second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    second_term_loss32 = torch.empty(args.batchsize, device=device, requires_grad=False)
    # for i in range(args.batch_size):
    for i in range(args.batchsize):
        """
        print(torch.mean(torch.sqrt((genFGen2[i, :] - xData).view(args.batchsize, -1).pow(2).sum(1))))
        print(torch.mean(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchsize, -1).pow(2).sum(1))))
        print(torch.mean(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1))))
        print('')

        print(torch.mean(torch.norm((genFGen2[i, :] - xData).view(args.batchsize, -1), p=None, dim=1)))
        print(torch.mean(torch.norm((genFGen2[i, :] - genFGen2).view(args.batchsize, -1), p=None, dim=1)))
        print(torch.mean(torch.norm((genFGen3[i, :] - genFGen3), p=None, dim=1)))
        print('')
        """

        # print(torch.mean(torch.sqrt((genFGen2[i, :] - xData).view(args.batchsize, -1).pow(2).sum(1))))
        # print(torch.mean(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchsize, -1).pow(2).sum(1))))
        # print(torch.mean(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1))))
        # print('')

        # print(torch.sqrt((genFGen2[i, :] - xData).view(args.batchsize, -1).pow(2).sum(1)))
        # print(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchsize, -1).pow(2).sum(1)))
        # print(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1)))
        # print('')

        # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p='fro', dim=1).requires_grad_()
        # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()
        # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()**2
        # second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1))**2
        # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_()**2

        # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2

        # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2
        # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).view(args.batchsize, -1).pow(2).sum(1)).requires_grad_() ** 2

        # second_term_loss22 = torch.sqrt(
        #    1e-17 + (genFGen2[i, :] - xData).view(args.batchsize, -1).pow(2).sum(1)).requires_grad_() ** 2

        # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()**2
        # second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2

        # tempVarVar21 = genFGen2[i, :] - xData
        # print(tempVarVar21.shape)

        # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2
        second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2

        # 61562.1641
        # 4.7732

        # print(genFGen2[i, :].shape)
        # print(xData.shape)

        # tempVarVar21 = genFGen2[i, :] - xData
        # print(tempVarVar21.shape)

        # print(second_term_loss22.shape)
        # adsfasfs

        # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()
        # print(second_term_loss22.shape)
        second_term_loss32[i] = torch.min(second_term_loss22)
    # print(second_term_loss32)
    # print(second_term_loss32.shape)
    # print(torch.norm(genFGen2 - xData, p=None, dim=0).shape)
    # second_term_loss22 = torch.min(second_term_loss32)
    # print(second_term_loss22)
    # print(second_term_loss22.shape)
    # second_term_loss2 = torch.mean(second_term_loss32)
    # second_term_loss2 = 0.3 * torch.mean(second_term_loss32)
    # second_term_loss2 = 3.0 * torch.mean(second_term_loss32)
    # second_term_loss2 = 7.62939453125 * torch.mean(second_term_loss32)
    # print(second_term_loss2)
    # print(second_term_loss2.shape)

    # second_term_loss2 = 0.3 * torch.mean(second_term_loss32)

    # second_term_loss2 = 0.3 * torch.mean(second_term_loss32)
    # second_term_loss2 = 0.001 * torch.mean(second_term_loss32)

    # second_term_loss2 = 0.001 * torch.mean(second_term_loss32)

    # second_term_loss2 = 0.001 * torch.mean(second_term_loss32)
    second_term_loss2 = torch.mean(second_term_loss32)

    # print(second_term_loss2)
    # asdfasfd

    # second_term_loss2.retain_grad()

    # second_term_loss2.retain_grad()
    # second_term_loss2.retain_grad()

    # (?)
    # second_term_loss2.retain_grad()
    # (?)

    # import timeit
    # start = timeit.default_timer()

    # stop = timeit.default_timer()
    # print('Time: ', stop - start)

    # print(second_term_loss2)
    # print('')

    # print('')
    # print(first_term_loss)

    # print(second_term_loss2)
    # print('')

    # print(first_term_loss)
    # print(second_term_loss2)

    # second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    # for i in range(args.batch_size):
    #    second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_()**2
    #    second_term_loss32[i] = torch.min(second_term_loss22)
    # second_term_loss2 = torch.mean(second_term_loss32)

    # print(genFGen2.shape)
    # print(genFGen3.shape)

    # print(xData.shape)
    # print('')

    # third_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    third_term_loss32 = torch.empty(args.batchsize, device=device, requires_grad=False)
    # for i in range(args.batch_size):
    for i in range(args.batchsize):
        # print(xData.shape)
        # print(genFGen2.shape)

        # print(genFGen3.shape)
        # print('')

        # print(xData.squeeze().shape)
        # print(genFGen2.squeeze().shape)
        # print('')

        # print((genFGen2[i, :] - xData).pow(2).sum(1).shape)
        # print((genFGen2[i, :] - genFGen2).pow(2).sum(1).shape)

        # print((genFGen2[i, :].squeeze() - xData.squeeze()).pow(2).sum(1).shape)
        # print((genFGen2[i, :].squeeze() - genFGen2.squeeze()).pow(2).sum(1).shape)

        # print((genFGen3[i, :] - genFGen3).pow(2).sum(1).shape)
        # print('')

        # print(torch.norm(genFGen2[i, :] - xData, p=None, dim=2).shape)
        # print(torch.norm(genFGen2[i, :] - genFGen2, p=None, dim=2).shape)

        # print(torch.norm(genFGen2[i, :].squeeze() - xData.squeeze(), p=None, dim=2).shape)
        # print(torch.norm(genFGen2[i, :].squeeze() - genFGen2.squeeze(), p=None, dim=2).shape)

        # print(torch.norm(genFGen3[i, :] - genFGen3, p=None, dim=1).shape)
        # print('')

        # a = torch.randn(64, 3, 32, 32)
        # a = a.view(64, -1)
        # b = torch.norm(a, p=2, dim=1)

        # a = torch.randn(64, 1, 28, 28)
        # a = a.view(64, -1)
        # b = torch.norm(a, p=2, dim=1)

        # print((genFGen2[i, :] - xData).view(args.batchsize,-1).pow(2).sum(1).shape)
        # print((genFGen2[i, :] - genFGen2).view(args.batchsize,-1).pow(2).sum(1).shape)
        # print('')

        # print(torch.norm((genFGen2[i, :] - xData).view(args.batchsize,-1), p=None, dim=1).shape)
        # print(torch.norm((genFGen2[i, :] - genFGen2).view(args.batchsize,-1), p=None, dim=1).shape)
        # print('')

        """
        print(torch.mean(torch.sqrt((genFGen2[i, :] - xData).view(args.batchsize, -1).pow(2).sum(1))))
        print(torch.mean(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchsize, -1).pow(2).sum(1))))
        print(torch.mean(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1))))
        print('')

        print(torch.mean(torch.norm((genFGen2[i, :] - xData).view(args.batchsize, -1), p=None, dim=1)))
        print(torch.mean(torch.norm((genFGen2[i, :] - genFGen2).view(args.batchsize, -1), p=None, dim=1)))
        print(torch.mean(torch.norm((genFGen3[i, :] - genFGen3), p=None, dim=1)))
        print('')
        """

        # print(torch.mean(torch.sqrt((genFGen2[i, :] - xData).view(args.batchsize, -1).pow(2).sum(1))))
        # print(torch.mean(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchsize, -1).pow(2).sum(1))))
        # print(torch.mean(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1))))
        # print('')

        # print(torch.sqrt((genFGen2[i, :] - xData).view(args.batchsize, -1).pow(2).sum(1)))
        # print(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchsize, -1).pow(2).sum(1)))
        # print(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1)))
        # print('')

        # third_term_loss22 = (torch.norm(genFGen3[i, :] - genFGen3, p=None, dim=1).requires_grad_()) / (
        #            1.0e-17 + torch.norm(genFGen2[i, :] - genFGen2, p=None, dim=1).requires_grad_())
        # third_term_loss22 = (torch.sqrt(1e-17 + (genFGen3[i, :] - genFGen3).pow(2).sum(1)).requires_grad_()) / (
        #        1e-17 + torch.sqrt(1e-17 + (genFGen2[i, :] - genFGen2).pow(2).sum(1)).requires_grad_())

        # third_term_loss22 = (torch.sqrt(1e-17 + (genFGen3[i, :] - genFGen3).pow(2).sum(1)).requires_grad_()) / (
        #        1e-17 + torch.sqrt(1e-17 + (genFGen2[i, :] - genFGen2).pow(2).sum(1)).requires_grad_())

        # hbdafj = genFGen3[i, :] - genFGen3
        # print(hbdafj.shape)

        # adfa = genFGen2[i, :] - xData
        # print(adfa.shape)

        # third_term_loss22 = (torch.sqrt(1e-17 + (genFGen3[i, :] - genFGen3).pow(2).sum(1)).requires_grad_()) / (
        #        1e-17 + torch.sqrt(1e-17 + (genFGen2[i, :] - genFGen2).view(args.batchsize, -1).pow(2).sum(1)).requires_grad_())

        # third_term_loss22 = (torch.sqrt(1e-17 + (genFGen3[i, :] - genFGen3).pow(2).sum(1)).requires_grad_()) / (
        #       1e-17 + torch.sqrt(1e-17 + (genFGen2[i, :] - genFGen2).view(args.batchsize, -1).pow(2).sum(1)).requires_grad_())

        # third_term_loss22 = (torch.sqrt(1e-17 + (genFGen3[i, :] - genFGen3).pow(2).sum(1)).requires_grad_()) / (
        #            1e-17 + torch.sqrt(1e-17 + (genFGen2[i, :] - genFGen2).pow(2).sum(1)).requires_grad_())

        # third_term_loss22 = (torch.norm(genFGen3[i, :] - genFGen3, p=None, dim=1).requires_grad_()) / (
        #            1.0e-17 + torch.norm(genFGen2[i, :] - genFGen2, p=None, dim=1).requires_grad_())

        third_term_loss22 = (torch.sqrt(1e-17 + (genFGen3[i, :] - genFGen3).pow(2).sum(1)).requires_grad_()) / (
                1e-17 + torch.sqrt(1e-17 + (genFGen2[i, :] - genFGen2).pow(2).sum(1)).requires_grad_())

        # print(third_term_loss22.shape)

        third_term_loss32[i] = torch.mean(third_term_loss22)
    # third_term_loss12 = torch.mean(third_term_loss32)
    # third_term_loss12 = 0.01 * torch.mean(third_term_loss32)
    # third_term_loss12 = 0.025 * torch.mean(third_term_loss32)
    # third_term_loss12 = 0.25 * torch.mean(third_term_loss32)
    # third_term_loss12 = 0.1 * torch.mean(third_term_loss32)

    # third_term_loss12 = 0.25 * torch.mean(third_term_loss32)

    # third_term_loss12 = 0.25 * torch.mean(third_term_loss32)
    # third_term_loss12 = 0.1 * torch.mean(third_term_loss32)

    # third_term_loss12 = 0.1 * torch.mean(third_term_loss32)

    # third_term_loss12 = 0.1 * torch.mean(third_term_loss32)
    third_term_loss12 = torch.mean(third_term_loss32)

    # print(third_term_loss12)
    # adfdfasc

    # third_term_loss12.retain_grad()

    # third_term_loss12.retain_grad()
    # third_term_loss12.retain_grad()

    # (?)
    # third_term_loss12.retain_grad()
    # (?)

    # print(third_term_loss12)
    # print('')

    # return first_term_loss + second_term_loss2
    # return first_term_loss + second_term_loss2, xData
    # return first_term_loss + second_term_loss2 + third_term_loss12, xData

    # return first_term_loss + second_term_loss2 + third_term_loss12, xData
    # return first_term_loss + second_term_loss2 + third_term_loss12

    # print(first_term_loss)
    # print(second_term_loss2)

    # print(third_term_loss12)
    # print('')

    # torch.set_printoptions(sci_mode=False)

    # print(first_term_loss)
    # print('')

    """
    #print(torch.isnan(first_term_loss))
    if torch.isnan(first_term_loss):
        first_term_loss = 0.0
    """

    # print(first_term_loss)
    # print('')

    # return first_term_loss + second_term_loss2 + third_term_loss12
    # return first_term_loss + second_term_loss2 + third_term_loss12

    # print(second_term_loss2)
    # print(third_term_loss12)

    # if torch.isnan(first_term_loss):
    #    return second_term_loss2 + third_term_loss12
    # else:
    #    return first_term_loss + second_term_loss2 + third_term_loss12

    # return first_term_loss + second_term_loss2 + third_term_loss12

    # return first_term_loss + second_term_loss2 + third_term_loss12
    # return first_term_loss + second_term_loss2 + third_term_loss12

    # return first_term_loss + second_term_loss2 + third_term_loss12
    # return first_term_loss + second_term_loss2 + third_term_loss12, first_term_loss, second_term_loss2

    print('')
    print(first_term_loss.item())

    print(second_term_loss2.item())
    print(third_term_loss12.item())

    # print('')
    # print(first_term_loss.grad)

    # print(second_term_loss2.grad)
    # print(third_term_loss12.grad)

    print('')

    # total_totTotalLoss = first_term_loss * second_term_loss2 * third_term_loss12
    # total_totTotalLoss = first_term_loss + second_term_loss2 + third_term_loss12

    # total_totTotalLoss = first_term_loss + second_term_loss2 + third_term_loss12

    # total_totTotalLoss = first_term_loss + second_term_loss2 + third_term_loss12
    # total_totTotalLoss = first_term_loss + 0.001 * second_term_loss2 + 0.1 * third_term_loss12

    # total_totTotalLoss = first_term_loss + 0.001 * second_term_loss2 + 0.1 * third_term_loss12

    # total_totTotalLoss = first_term_loss + 0.001 * second_term_loss2 + 0.1 * third_term_loss12
    # total_totTotalLoss = first_term_loss + 0.001 * second_term_loss2 + 10.0 * third_term_loss12

    # total_totTotalLoss = first_term_loss + 0.001 * second_term_loss2 + 0.1 * third_term_loss12
    # total_totTotalLoss = first_term_loss + 10.0 * second_term_loss2 + 0.1 * third_term_loss12

    # total_totTotalLoss = first_term_loss + 0.001 * second_term_loss2 + 0.1 * third_term_loss12

    # total_totTotalLoss = first_term_loss + 0.001 * second_term_loss2 + 0.1 * third_term_loss12
    # total_totTotalLoss = first_term_loss + 1.0 * second_term_loss2 + 0.1 * third_term_loss12

    # total_totTotalLoss = first_term_loss + 1.0 * second_term_loss2 + 0.1 * third_term_loss12

    # total_totTotalLoss = first_term_loss + 1.0 * second_term_loss2 + 0.1 * third_term_loss12
    total_totTotalLoss = first_term_loss + 0.3 * second_term_loss2 + 0.025 * third_term_loss12

    # total_totTotalLoss.retain_grad()

    # total_totTotalLoss.retain_grad()
    total_totTotalLoss.retain_grad()

    # return first_term_loss + second_term_loss2 + third_term_loss12, first_term_loss, second_term_loss2
    # return first_term_loss + second_term_loss2 + third_term_loss12, first_term_loss, second_term_loss2, third_term_loss12

    # return first_term_loss + second_term_loss2 + third_term_loss12, first_term_loss, second_term_loss2, third_term_loss12
    return total_totTotalLoss, first_term_loss, second_term_loss2, third_term_loss12


def use_loss_fn3(genFGen2, args, model, genFGen3, toUse_storeAll, toUse_storeAll2, xData):
    # first_term_loss = compute_loss2(genFGen2, args, model)
    # first_term_loss2 = compute_loss2(genFGen2, args, model)
    # first_term_loss = torch.log(first_term_loss2 / (1.0 - first_term_loss2))

    # first_term_loss = compute_loss2(genFGen2, args, model)

    # first_term_loss = compute_loss2(genFGen2, args, model)
    # first_term_loss = compute_loss2(genFGen2, args, model)

    # print('')
    # print(first_term_loss)

    # mu = torch.from_numpy(np.array([2.805741, -0.00889241], dtype="float32")).to(device)
    # S = torch.from_numpy(np.array([[pow(0.3442525,2), 0.0], [0.0, pow(0.35358343,2)]], dtype="float32")).to(device)

    # mu = torch.from_numpy(np.array([2.8093171, 1.2994107e-03], dtype="float32")).to(device)
    # S = torch.from_numpy(np.array([[pow(0.35840544, 2), 0.0], [0.0, pow(0.34766033, 2)]], dtype="float32")).to(device)

    # mu = torch.from_numpy(np.array([0.0, 0.0], dtype="float32")).to(device)
    # S = torch.from_numpy(np.array([[pow(1.0,2), 0.0], [0.0, pow(1.0,2)]], dtype="float32")).to(device)

    """
    #storeAll = torch.from_numpy(np.array(0.0, dtype="float32")).to(device)
    storeAll = torch.empty(args.batch_size, device=device, requires_grad=False)
    #toUse_storeAll = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=S)
    #for loopIndex_i in range(genFGen2.size()[0]):
    for loopIndex_i in range(args.batch_size):
        #storeAll += torch.exp(toUse_storeAll.log_prob(genFGen2[loopIndex_i:1 + loopIndex_i, :].squeeze(0)))
        #storeAll[loopIndex_i] = torch.exp(toUse_storeAll.log_prob(genFGen2[loopIndex_i:1 + loopIndex_i, :].squeeze(0)).requires_grad_())

        #storeAll[loopIndex_i] = torch.exp(
        #    toUse_storeAll.log_prob(genFGen2[loopIndex_i:1 + loopIndex_i, :].squeeze(0)).requires_grad_())

        storeAll[loopIndex_i] = 0.5 * torch.exp(toUse_storeAll.log_prob(genFGen2[loopIndex_i:1 + loopIndex_i, :].squeeze(0)).requires_grad_())\
                                + 0.5 * torch.exp(toUse_storeAll2.log_prob(genFGen2[loopIndex_i:1 + loopIndex_i, :].squeeze(0)).requires_grad_())
    #storeAll /= genFGen2.size()[0]
    first_term_loss = torch.mean(storeAll)
    """

    # print(first_term_loss)
    first_term_loss = compute_loss2(genFGen2, args, model)

    # print(first_term_loss)

    """
    second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    for i in range(args.batch_size):
        second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_() ** 2
        second_term_loss32[i] = torch.min(second_term_loss22)
    second_term_loss2 = torch.mean(second_term_loss32)
    """

    # print(first_term_loss)
    # print('')

    # print('')
    # print(compute_loss2(mu.unsqueeze(0), args, model))

    # print(torch.exp(toUse_storeAll.log_prob(mu)))
    # print('')

    # first_term_loss = storeAll

    # xData = toy_data.inf_train_gen(args.data, batch_size=args.batch_size)
    # xData = torch.from_numpy(xData).type(torch.float32).to(device)

    # print(xData.shape)
    # print(torch.mean(xData))
    # print(torch.std(xData))

    # xData = torch.empty((args.batch_size, 2), device=device)
    # xData[:args.batch_size//2, :] = toUse_storeAll.sample((args.batch_size//2,)) # .sample_n(args.batch_size // 2)
    # xData[args.batch_size//2:, :] = toUse_storeAll2.sample((args.batch_size//2,)) # .sample_n(args.batch_size//2)

    """
    xData = torch.empty((args.batch_sizeM, 2), device=device)
    xData[:args.batch_sizeM // 2, :] = toUse_storeAll.sample((args.batch_sizeM // 2,))  # .sample_n(args.batch_size // 2)
    xData[args.batch_sizeM // 2:, :] = toUse_storeAll2.sample((args.batch_sizeM // 2,))  # .sample_n(args.batch_size//2)
    """

    # xData = torch.empty((args.batch_size, 2)).normal_(mean=[2.82507515, 1.92882611e-04 + 0.8], std=0.5)
    # xData[args.batch_size//2:,:] = torch.empty((args.batch_size, 2)).normal_(mean=4, std=0.5)

    # mu = torch.from_numpy(np.array([2.82507515, 1.92882611e-04 + 0.8], dtype="float32")).to(device)
    # S = torch.from_numpy(np.array([[pow(0.07166782, 2), 0.0], [0.0, pow(0.06917527, 2)]], dtype="float32")).to(device)
    # mu2 = torch.from_numpy(np.array([2.82507515, 1.92882611e-04 - 0.8], dtype="float32")).to(device)
    # toUse_storeAll = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=S)
    # toUse_storeAll2 = torch.distributions.MultivariateNormal(loc=mu2, covariance_matrix=S)

    # print(xData.shape)
    # print(torch.mean(xData))
    # print(torch.std(xData))

    # var2 = []
    # for i in genFGen2:
    #    var1 = []
    #    for j in xData:
    #        new_stuff = torch.dist(i, j, 2)  # this is a tensor
    #        var1.append(new_stuff.unsqueeze(0))
    #    var1_tensor = torch.cat(var1)
    #    second_term_loss2 = torch.min(var1_tensor) / args.batch_size
    #    var2.append(second_term_loss2.unsqueeze(0))
    # var2_tensor = torch.cat(var2)
    # second_term_loss = torch.mean(var2_tensor) / args.batch_size
    # second_term_loss *= 100.0

    # print('')
    # print(second_term_loss)

    # If you know in advance the size of the final tensor, you can allocate
    # an empty tensor beforehand and fill it in the for loop.

    # x = torch.empty(size=(len(items), 768))
    # for i in range(len(items)):
    #    x[i] = calc_result

    # print(len(genFGen2))
    # print(genFGen2.shape[0])
    # len(.) and not .shape[0]

    # print(len(xData))
    # print(xData.shape[0])
    # Use len(.) and not .shape[0]

    """
    #second_term_loss = torch.empty(size=(len(genFGen2), len(xData))).to(device)
    #second_term_loss = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=True)
    #second_term_loss3 = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=True)
    #second_term_loss3 = torch.empty(size=(len(genFGen2), len(xData)), device=device, requires_grad=False)
    second_term_loss3 = torch.empty(size=(args.batch_size, args.batch_size), device=device, requires_grad=False)
    #for i in range(len(genFGen2)):
    for i in range(args.batch_size):
        #for j in range(len(xData)):
        for j in range(args.batch_size):
            #second_term_loss[i, j] = torch.dist(genFGen2[i,:], xData[j,:], 2)
            #second_term_loss[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.tensor(0.1, requires_grad=True)

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1)
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1).requires_grad_()

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 1).requires_grad_()
            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 2).requires_grad_()**2

            #second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 2).requires_grad_()**2
            second_term_loss3[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 2).requires_grad_()

            #second_term_loss[i, j] = torch.dist(genFGen2[i, :], xData[j, :], 2)**2
    #second_term_loss2, _ = torch.min(second_term_loss, 1)
    second_term_loss2, _ = torch.min(second_term_loss3, 1)
    #second_term_loss = 5000.0 * torch.mean(second_term_loss2) / (args.batch_size**2)
    #second_term_loss = lambda1 * torch.mean(second_term_loss2) / (args.batch_size ** 2)
    #second_term_loss = lambda1 * torch.mean(second_term_loss2)
    second_term_loss = torch.mean(second_term_loss2)

    #print(second_term_loss)
    #print('')

    print('')
    print(first_term_loss)
    print(second_term_loss)

    print('')
    """

    # args.batch_size = 2
    # genFGen2 = torch.from_numpy(np.array([[3, 0], [2, 0]], dtype="float32")).to(device)
    # xData = torch.from_numpy(np.array([[1, 0], [0, 1]], dtype="float32")).to(device)

    # import timeit
    # start = timeit.default_timer()
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)

    """
    second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    for i in range(args.batch_size):
        #second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p='fro', dim=1).requires_grad_()
        #second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()
        second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()**2
        #second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()
        #print(second_term_loss22.shape)
        second_term_loss32[i] = torch.min(second_term_loss22)
    #print(second_term_loss32)
    #print(second_term_loss32.shape)
    #print(torch.norm(genFGen2 - xData, p=None, dim=0).shape)
    #second_term_loss22 = torch.min(second_term_loss32)
    #print(second_term_loss22)
    #print(second_term_loss22.shape)
    second_term_loss2 = torch.mean(second_term_loss32)
    #second_term_loss2 = 7.62939453125 * torch.mean(second_term_loss32)
    #print(second_term_loss2)
    #print(second_term_loss2.shape)
    """

    # import timeit
    # start = timeit.default_timer()
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)

    # print('')
    # print(second_term_loss2)

    # distances = torch.norm(vertices - point_locs, p=2, dim=1)
    # distances = torch.sqrt((vertices - point_locs).pow(2).sum(1))

    # import timeit
    # start = timeit.default_timer()
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)

    second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    for i in range(args.batch_size):
        # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p='fro', dim=1).requires_grad_()
        # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()
        # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()**2
        # second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1))**2
        second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2
        # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()
        # print(second_term_loss22.shape)
        second_term_loss32[i] = torch.min(second_term_loss22)
    # print(second_term_loss32)
    # print(second_term_loss32.shape)
    # print(torch.norm(genFGen2 - xData, p=None, dim=0).shape)
    # second_term_loss22 = torch.min(second_term_loss32)
    # print(second_term_loss22)
    # print(second_term_loss22.shape)
    # second_term_loss2 = torch.mean(second_term_loss32)
    second_term_loss2 = 0.3 * torch.mean(second_term_loss32)
    # second_term_loss2 = 3.0 * torch.mean(second_term_loss32)
    # second_term_loss2 = 7.62939453125 * torch.mean(second_term_loss32)
    # print(second_term_loss2)
    # print(second_term_loss2.shape)

    # import timeit
    # start = timeit.default_timer()
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)

    # print(second_term_loss2)
    # print('')

    # print('')
    # print(first_term_loss)

    # print(second_term_loss2)
    # print('')

    # second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    # for i in range(args.batch_size):
    #    second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_()**2
    #    second_term_loss32[i] = torch.min(second_term_loss22)
    # second_term_loss2 = torch.mean(second_term_loss32)

    third_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    for i in range(args.batch_size):
        # third_term_loss22 = (torch.norm(genFGen3[i, :] - genFGen3, p=None, dim=1).requires_grad_()) / (
        #            1.0e-17 + torch.norm(genFGen2[i, :] - genFGen2, p=None, dim=1).requires_grad_())
        third_term_loss22 = (torch.sqrt(1e-17 + (genFGen3[i, :] - genFGen3).pow(2).sum(1)).requires_grad_()) / (
                1e-17 + torch.sqrt(1e-17 + (genFGen2[i, :] - genFGen2).pow(2).sum(1)).requires_grad_())
        third_term_loss32[i] = torch.mean(third_term_loss22)
    # third_term_loss12 = torch.mean(third_term_loss32)
    # third_term_loss12 = 0.01 * torch.mean(third_term_loss32)
    third_term_loss12 = 0.025 * torch.mean(third_term_loss32)
    # third_term_loss12 = 0.1 * torch.mean(third_term_loss32)

    # print(third_term_loss12)
    # print('')

    # return first_term_loss + second_term_loss2
    # return first_term_loss + second_term_loss2, xData
    return first_term_loss + second_term_loss2 + third_term_loss12


# nrand = 200
# nrand = 100

# nrand = 200
nrand = 28 * 28

# gen = DCGANGenerator(nrand)

# gen = DCGANGenerator(nrand)
# genGen = DCGANGenerator(nrand)

# genGen = DCGANGenerator(nrand)
# genGen = DCGANGenerator2(nrand)

# genGen = DCGANGenerator(nrand)
# genGen = DCGANGenerator3(nrand)

# genGen = DCGANGenerator3(nrand)
# genGen = DCGANgeneratorDCGAN()

# genGen = DCGANgeneratorDCGAN()
# genGen = GANgeneratorGAN(input_size=100, n_class=28*28)

# genGen = GANgeneratorGAN(input_size=100, n_class=28*28)
# genGen = DCGANGenerator3(nrand)

# genGen = DCGANGenerator3(nrand)

# genGen = DCGANGenerator3(nrand)
# genGen = DCGANGenerator3(nrand)

# genGen = DCGANGenerator3(nrand)

# genGen = DCGANGenerator3(nrand)
# genGen = GAN2generatorGAN2(input_size=nrand, n_class=28*28)

# genGen = GAN2generatorGAN2(input_size=nrand, n_class=28*28)
# genGen = GANgeneratorGAN(input_size=nrand, n_class=28*28)

# genGen = GANgeneratorGAN(input_size=nrand, n_class=28*28)

# genGen = GANgeneratorGAN(input_size=nrand, n_class=28*28)
# genGen = GAN1generatorGAN1(input_size=28*28, n_class=28*28)

# genGen = GAN1generatorGAN1(input_size=28*28, n_class=28*28)
genGen = DCGANGenerator3(nrand)

# optimizerGen = torch.optim.Adam(genGen.parameters(), lr=args.lr)

# optimizerGen = torch.optim.Adam(genGen.parameters(), lr=args.lr)
# optimizerGen = torch.optim.Adam(genGen.parameters(), lr=args.lr)

# optimizerGen = torch.optim.Adam(genGen.parameters(), lr=args.lr)

# optimizerGen = torch.optim.Adam(genGen.parameters(), lr=args.lr)
# optimizerGen = torch.optim.Adam(genGen.parameters(), lr=args.lr*0.1)

# optimizerGen = torch.optim.Adam(genGen.parameters(), lr=args.lr*0.1)
# optimizerGen = torch.optim.Adam(genGen.parameters(), lr=args.lr*0.01)

# optimizerGen = torch.optim.Adam(genGen.parameters(), lr=args.lr*0.01)
# optimizerGen = torch.optim.Adam(genGen.parameters(), lr=args.lr*0.0001)

# optimizerGen = torch.optim.Adam(genGen.parameters(), lr=args.lr*0.0001)
# optimizerGen = torch.optim.Adam(genGen.parameters(), lr=args.lr*0.001)

# print(args.lr*0.001)

# optimizerGen = torch.optim.Adam(genGen.parameters(), lr=args.lr)
# optimizerGen = torch.optim.Adam(genGen.parameters(), lr=args.lr*0.01)

optimizerGen = torch.optim.Adam(genGen.parameters(), lr=args.lr * 0.01)
# optimizerGen = torch.optim.Adam(genGen.parameters(), lr=args.lr*0.1)

# Random seed
if args.seed is None:
    args.seed = np.random.randint(100000)

utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

# logger
logger.info(args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# print(device)
# asdfacsfs

if device.type == 'cuda':
    logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024 ** 3)))
else:
    logger.info('WARNING: Using device {}'.format(device))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)


def geometric_logprob(ns, p):
    return torch.log(1 - p + 1e-10) * (ns - 1) + torch.log(p + 1e-10)


def standard_normal_sample(size):
    return torch.randn(size)


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reduce_bits(x):
    if args.nbits < 8:
        x = x * 255
        x = torch.floor(x / 2 ** (8 - args.nbits))
        x = x / 2 ** args.nbits
    return x


def add_noise(x, nvals=256):
    """
    [0, 1] -> [0, nvals] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * (nvals - 1) + noise
        x = x / nvals
    return x


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def add_padding(x, nvals=256):
    # Theoretically, padding should've been added before the add_noise preprocessing.
    # nvals takes into account the preprocessing before padding is added.
    if args.padding > 0:
        if args.padding_dist == 'uniform':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).uniform_()
            logpu = torch.zeros_like(u).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        elif args.padding_dist == 'gaussian':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).normal_(nvals / 2, nvals / 8)
            logpu = normal_logprob(u, nvals / 2, math.log(nvals / 8)).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        else:
            raise ValueError()
    else:
        return x, torch.zeros(x.shape[0], 1).to(x)


def remove_padding(x):
    if args.padding > 0:
        return x[:, :im_dim, :, :]
    else:
        return x


logger.info('Loading dataset {}'.format(args.data))

# Dataset and hyperparameters

# Dataset and hyperparameters
if args.data == 'cifar10':
    im_dim = 3
    n_classes = 10

    if args.task in ['classification', 'hybrid']:

        # Classification-specific pre-processing
        transform_train = transforms.Compose([
            transforms.Resize(args.imagesize),
            transforms.RandomCrop(32, padding=4, padding_mode=args.rcrop_pad_mode),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            add_noise,
        ])

        transform_test = transforms.Compose([
            transforms.Resize(args.imagesize),
            transforms.ToTensor(),
            add_noise,
        ])

        # Remove the logit transform.
        init_layer = layers.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        transform_train = transforms.Compose([
            transforms.Resize(args.imagesize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            add_noise,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(args.imagesize),
            transforms.ToTensor(),
            add_noise,
        ])
        init_layer = layers.LogitTransform(0.05)

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.dataroot, train=True, transform=transform_train),
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.nworkers,
    )

    # test_loader = torch.utils.data.DataLoader(
    #    datasets.CIFAR10(args.dataroot, train=False, transform=transform_test),
    #    batch_size=args.val_batchsize,
    #    shuffle=False,
    #    num_workers=args.nworkers,
    # )

    teTest_loader = datasets.CIFAR10(args.dataroot, train=False, transform=transform_test)

    args.val_batchsize = 1

    """
    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)
    """

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    # print(len(teTest_loader))
    # asdfdasf

    # print(len(teTest_loader))
    # asdfsfs

    # print(len(teTest_loader))
    # asdfsfs

    # print(len(teTest_loader))
    # asdfasfd

    test_loader = torch.utils.data.DataLoader(
        teTest_loader,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.nworkers,
    )

elif args.data == 'mnist':
    im_dim = 1

    init_layer = layers.LogitTransform(1e-6)
    n_classes = 10

    # train_loader = torch.utils.data.DataLoader(
    #    datasets.MNIST(
    #        args.dataroot, train=True, transform=transforms.Compose([
    #            transforms.Resize(args.imagesize),
    #            transforms.ToTensor(),
    #            add_noise,
    #        ])
    #    ),
    #    batch_size=args.batchsize,
    #    shuffle=True,
    #    num_workers=args.nworkers,
    # )

    """
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.dataroot, train=True, transform=transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                add_noise,
            ])
        ),
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.nworkers,
    )
    """

    '''
    train_loader2 = datasets.MNIST(
            args.dataroot, train=True, transform=transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                add_noise,
            ]))

    print(datasets.MNIST(
            args.dataroot, train=True, transform=transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                add_noise,
            ])))
    '''

    '''
    dataset = datasets.MNIST('./data')
    idx = dataset.mnist.train_labels == 1
    idx += dataset.mnist.train_labels == 2
    dataset.train_labels = dataset.mnist.train_labels[idx]
    dataset.train_data = dataset.mnist.train_data[idx]

    print(dataset.train_data)
    print(dataset.train_data.shape)
    '''

    '''
    dataset = datasets.MNIST(
            args.dataroot, train=True, transform=transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                add_noise,
            ]))
    idx = dataset.mnist.train_labels == 1
    idx += dataset.mnist.train_labels == 2
    dataset.train_labels = dataset.mnist.train_labels[idx]
    dataset.train_data = dataset.mnist.train_data[idx]

    print(dataset.train_data)
    print(dataset.train_data.shape)
    '''

    '''
    #trainset = datasets.MNIST('./data', train=True, transform=None)
    trainset = datasets.MNIST(
            args.dataroot, train=True, transform=transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                add_noise,
            ]))

    evens = list(range(0, len(trainset), 2))
    odds = list(range(1, len(trainset), 2))

    trainset_1 = torch.utils.data.Subset(trainset, evens)
    trainset_2 = torch.utils.data.Subset(trainset, odds)

    #trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=4, shuffle=True, num_workers=2)
    #trainloader_2 = torch.utils.data.DataLoader(trainset_2, batch_size=4, shuffle=True, num_workers=2)

    print(len(trainset_1))
    '''

    """
    args.batch_sizeM = len(datasets.MNIST(
            args.dataroot, train=True, transform=transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                add_noise,
            ])))

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.dataroot, train=True, transform=transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                add_noise,
            ])
        ),
        batch_size=args.batch_sizeM,
        shuffle=True,
        num_workers=args.nworkers,
    )
    """

    """
    trainset = datasets.MNIST(
        args.dataroot, train=True, transform=transforms.Compose([
            transforms.Resize(args.imagesize),
            transforms.ToTensor(),
            add_noise,
        ]))

    evens = list(range(0, len(trainset), 2))
    trainset = torch.utils.data.Subset(trainset, evens)
    """

    trainset = datasets.MNIST(
        args.dataroot, train=True, transform=transforms.Compose([
            transforms.Resize(args.imagesize),
            transforms.ToTensor(),
            add_noise,
        ]))

    evens = list(range(0, len(trainset), 2))
    trainset = torch.utils.data.Subset(trainset, evens)

    evens = list(range(0, len(trainset), 2))
    trainset = torch.utils.data.Subset(trainset, evens)

    args.batch_sizeM = len(trainset)

    # print(args.batch_sizeM)
    # adfasfas

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_sizeM,
        shuffle=True,
        num_workers=args.nworkers,
    )

    """
    from options import Options
    opt = Options().parse()

    opt.isize = 32
    opt.dataset = 'mnist'

    opt.nc = 1
    opt.niter = 15

    opt.abnormal_class = 0
    from data import load_data

    # LOAD DATA
    dataloader = load_data(opt)
    """

    args.val_batchsize = 1

    teTest_loader = datasets.MNIST(
        args.dataroot, train=False, transform=transforms.Compose([
            transforms.Resize(args.imagesize),
            transforms.ToTensor(),
            add_noise,
        ])
    )

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    # print(len(teTest_loader))
    # asdfasdf

    # print(len(teTest_loader))
    # asdfsbkf

    '''
    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)
    '''

    # print(len(teTest_loader))
    # asdfasdf

    '''
    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)
    '''

    # print(len(teTest_loader))
    # asdfasdf

    '''
    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)
    '''

    # print(len(teTest_loader))

    # print(len(teTest_loader))
    # print(len(teTest_loader))

    # print(len(teTest_loader))
    # dasfasffa

    """
    teTest_loLoader = datasets.FashionMNIST(
        args.dataroot, train=False, transform=transforms.Compose([
            transforms.Resize(args.imagesize),
            transforms.ToTensor(),
            add_noise,
        ])
    )

    evens = list(range(0, len(teTest_loLoader), 2))
    teTest_loLoader = torch.utils.data.Subset(teTest_loLoader, evens)

    evens = list(range(0, len(teTest_loLoader), 2))
    teTest_loLoader = torch.utils.data.Subset(teTest_loLoader, evens)

    evens = list(range(0, len(teTest_loLoader), 2))
    teTest_loLoader = torch.utils.data.Subset(teTest_loLoader, evens)

    #print(len(teTest_loLoader))
    #dasfasdf

    evens = list(range(0, len(teTest_loLoader), 2))
    teTest_loLoader = torch.utils.data.Subset(teTest_loLoader, evens)

    evens = list(range(0, len(teTest_loader), 2))
    teTest_loader = torch.utils.data.Subset(teTest_loader, evens)

    # use: torch.utils.data.ConcatDataset(datasets)
    teTest_loader = torch.utils.data.ConcatDataset((teTest_loader, teTest_loLoader))
    """

    # print(len(teTest_loader))
    # dasfasdf

    # print(len(teTest_loader))
    # adsfasdf

    # print(len(teTest_loader))
    # asdfsfs

    test_loader = torch.utils.data.DataLoader(
        teTest_loader,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.nworkers,
    )

    """
    teTest_loLoader = datasets.FashionMNIST(
        args.dataroot, train=False, transform=transforms.Compose([
            transforms.Resize(args.imagesize),
            transforms.ToTensor(),
            add_noise,
        ])
    )

    evens = list(range(0, len(teTest_loLoader), 2))
    teTest_loLoader = torch.utils.data.Subset(teTest_loLoader, evens)

    evens = list(range(0, len(teTest_loLoader), 2))
    teTest_loLoader = torch.utils.data.Subset(teTest_loLoader, evens)

    evens = list(range(0, len(teTest_loLoader), 2))
    teTest_loLoader = torch.utils.data.Subset(teTest_loLoader, evens)

    evens = list(range(0, len(teTest_loLoader), 2))
    teTest_loLoader = torch.utils.data.Subset(teTest_loLoader, evens)

    evens = list(range(0, len(teTest_loLoader), 2))
    teTest_loLoader = torch.utils.data.Subset(teTest_loLoader, evens)

    evens = list(range(0, len(teTest_loLoader), 2))
    teTest_loLoader = torch.utils.data.Subset(teTest_loLoader, evens)

    # print(len(teTest_loLoader))
    # dsfasfs

    #print(len(teTest_loLoader))
    #asdfdsfs

    test_loLoader = torch.utils.data.DataLoader(
        teTest_loLoader,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.nworkers,
    )
    """

    # use: test_loLoader
    # now use test_loLoader

    '''
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.dataroot, train=False, transform=transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                add_noise,
            ])
        ),
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.nworkers,
    )
    '''

    '''
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.dataroot, train=False, transform=transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                add_noise,
            ])
        ),
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.nworkers,
    )
    '''

    # test_loader = torch.utils.data.DataLoader(
    #    datasets.MNIST(
    #        args.dataroot, train=False, transform=transforms.Compose([
    #            transforms.Resize(args.imagesize),
    #            transforms.ToTensor(),
    #            add_noise,
    #        ])
    #    ),
    #    batch_size=args.val_batchsize,
    #    shuffle=False,
    #    num_workers=args.nworkers,
    # )

elif args.data == 'svhn':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    n_classes = 10
    train_loader = torch.utils.data.DataLoader(
        vdsets.SVHN(
            args.dataroot, split='train', download=True, transform=transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.RandomCrop(32, padding=4, padding_mode=args.rcrop_pad_mode),
                transforms.ToTensor(),
                add_noise,
            ])
        ),
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.nworkers,
    )
    test_loader = torch.utils.data.DataLoader(
        vdsets.SVHN(
            args.dataroot, split='test', download=True, transform=transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                add_noise,
            ])
        ),
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.nworkers,
    )
elif args.data == 'celebahq':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 256:
        logger.info('Changing image size to 256.')
        args.imagesize = 256
    train_loader = torch.utils.data.DataLoader(
        datasets.CelebAHQ(
            train=True, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                reduce_bits,
                lambda x: add_noise(x, nvals=2 ** args.nbits),
            ])
        ), batch_size=args.batchsize, shuffle=True, num_workers=args.nworkers
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CelebAHQ(
            train=False, transform=transforms.Compose([
                reduce_bits,
                lambda x: add_noise(x, nvals=2 ** args.nbits),
            ])
        ), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
    )
elif args.data == 'celeba_5bit':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 64:
        logger.info('Changing image size to 64.')
        args.imagesize = 64
    train_loader = torch.utils.data.DataLoader(
        datasets.CelebA5bit(
            train=True, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                lambda x: add_noise(x, nvals=32),
            ])
        ), batch_size=args.batchsize, shuffle=True, num_workers=args.nworkers
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CelebA5bit(train=False, transform=transforms.Compose([
            lambda x: add_noise(x, nvals=32),
        ])), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
    )
elif args.data == 'imagenet32':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 32:
        logger.info('Changing image size to 32.')
        args.imagesize = 32
    train_loader = torch.utils.data.DataLoader(
        datasets.Imagenet32(train=True, transform=transforms.Compose([
            add_noise,
        ])), batch_size=args.batchsize, shuffle=True, num_workers=args.nworkers
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.Imagenet32(train=False, transform=transforms.Compose([
            add_noise,
        ])), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
    )
elif args.data == 'imagenet64':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 64:
        logger.info('Changing image size to 64.')
        args.imagesize = 64
    train_loader = torch.utils.data.DataLoader(
        datasets.Imagenet64(train=True, transform=transforms.Compose([
            add_noise,
        ])), batch_size=args.batchsize, shuffle=True, num_workers=args.nworkers
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.Imagenet64(train=False, transform=transforms.Compose([
            add_noise,
        ])), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
    )

if args.task in ['classification', 'hybrid']:
    try:
        n_classes
    except NameError:
        raise ValueError('Cannot perform classification with {}'.format(args.data))
else:
    n_classes = 1

logger.info('Dataset loaded.')
logger.info('Creating model.')

input_size = (args.batchsize, im_dim + args.padding, args.imagesize, args.imagesize)
dataset_size = len(train_loader.dataset)

# print(dataset_size)
# print(len(test_loader.dataset))

# print(len(train_loader.dataset.mnist.data))
# print(len(train_loader.dataset.mnist.train_data))
# print(len(train_loader.dataset.mnist.train_labels))

# print(len(train_loader.dataset.mnist.test_data))
# print(len(train_loader.dataset.mnist.test_labels))

# print(len(test_loader.dataset.mnist.test_data))
# print(len(test_loader.dataset.mnist.test_labels))

if args.squeeze_first:
    input_size = (input_size[0], input_size[1] * 4, input_size[2] // 2, input_size[3] // 2)
    squeeze_layer = layers.SqueezeLayer(2)

# Model

# Model
model = ResidualFlow(
    input_size,
    n_blocks=list(map(int, args.nblocks.split('-'))),
    intermediate_dim=args.idim,
    factor_out=args.factor_out,
    quadratic=args.quadratic,
    init_layer=init_layer,
    actnorm=args.actnorm,
    fc_actnorm=args.fc_actnorm,
    batchnorm=args.batchnorm,
    dropout=args.dropout,
    fc=args.fc,
    coeff=args.coeff,
    vnorms=args.vnorms,
    n_lipschitz_iters=args.n_lipschitz_iters,
    sn_atol=args.sn_tol,
    sn_rtol=args.sn_tol,
    n_power_series=args.n_power_series,
    n_dist=args.n_dist,
    n_samples=args.n_samples,
    kernels=args.kernels,
    activation_fn=args.act,
    fc_end=args.fc_end,
    fc_idim=args.fc_idim,
    n_exact_terms=args.n_exact_terms,
    preact=args.preact,
    neumann_grad=args.neumann_grad,
    grad_in_forward=args.mem_eff,
    first_resblock=args.first_resblock,
    learn_p=args.learn_p,
    classification=args.task in ['classification', 'hybrid'],
    classification_hdim=args.cdim,
    n_classes=n_classes,
    block_type=args.block,
)

model.to(device)
ema = utils.ExponentialMovingAverage(model)


def parallelize(model):
    return torch.nn.DataParallel(model)


# logger.info(model)
logger.info('EMA: {}'.format(ema))


# Optimization
def tensor_in(t, a):
    for a_ in a:
        if t is a_:
            return True
    return False


scheduler = None

if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
    if args.scheduler: scheduler = CosineAnnealingWarmRestarts(optimizer, 20, T_mult=2, last_epoch=args.begin_epoch - 1)
elif args.optimizer == 'adamax':
    optimizer = optim.Adamax(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[60, 120, 160], gamma=0.2, last_epoch=args.begin_epoch - 1
        )
else:
    raise ValueError('Unknown optimizer {}'.format(args.optimizer))

best_test_bpd = math.inf
if (args.resume is not None):
    logger.info('Resuming model from {}'.format(args.resume))
    with torch.no_grad():
        x = torch.rand(1, *input_size[1:]).to(device)
        model(x)
    checkpt = torch.load(args.resume)

    # args = checkpt['args']
    # logger.info(args)

    # torch.save({
    #    'state_dict': model.state_dict(),
    #    'optimizer_state_dict': optimizer.state_dict(),
    #    'args': args,
    #    'ema': ema,
    #    'test_bpd': test_bpd,
    # }, os.path.join(args.save, 'models', 'mostMostMostRecent.pth'))

    test_bpd = checkpt['test_bpd']

    sd = {k: v for k, v in checkpt['state_dict'].items() if 'last_n_samples' not in k}
    state = model.state_dict()
    state.update(sd)
    model.load_state_dict(state, strict=True)
    ema.set(checkpt['ema'])
    if 'optimizer_state_dict' in checkpt:
        optimizer.load_state_dict(checkpt['optimizer_state_dict'])

        # Manually move optimizer state to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    del checkpt
    del state

logger.info(optimizer)

fixed_z = standard_normal_sample([min(32, args.batchsize),
                                  (im_dim + args.padding) * args.imagesize * args.imagesize]).to(device)

criterion = torch.nn.CrossEntropyLoss()


def compute_loss(x, model, beta=1.0):
    bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(n_classes).to(x)
    logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)

    if args.data == 'celeba_5bit':
        nvals = 32
    elif args.data == 'celebahq':
        nvals = 2 ** args.nbits
    else:
        nvals = 256

    x, logpu = add_padding(x, nvals)

    if args.squeeze_first:
        x = squeeze_layer(x)

    if args.task == 'hybrid':
        z_logp, logits_tensor = model(x.view(-1, *input_size[1:]), 0, classify=True)
        z, delta_logp = z_logp
    elif args.task == 'density':
        z, delta_logp = model(x.view(-1, *input_size[1:]), 0)
    elif args.task == 'classification':
        z, logits_tensor = model(x.view(-1, *input_size[1:]), classify=True)

    if args.task in ['density', 'hybrid']:
        # log p(z)
        logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)

        # log p(x)
        logpx = logpz - beta * delta_logp - np.log(nvals) * (
                args.imagesize * args.imagesize * (im_dim + args.padding)
        ) - logpu
        bits_per_dim = -torch.mean(logpx) / (args.imagesize * args.imagesize * im_dim) / np.log(2)

        logpz = torch.mean(logpz).detach()
        delta_logp = torch.mean(-delta_logp).detach()

    return bits_per_dim, logits_tensor, logpz, delta_logp


def computeLoss(x, model, beta=1.0):
    bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(n_classes).to(x)
    logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)

    if args.data == 'celeba_5bit':
        nvals = 32
    elif args.data == 'celebahq':
        nvals = 2 ** args.nbits
    else:
        nvals = 256

    x, logpu = add_padding(x, nvals)
    # _, logpu = add_padding(x, nvals)

    if args.squeeze_first:
        x = squeeze_layer(x)

    if args.task == 'hybrid':
        z_logp, logits_tensor = model(x.view(-1, *input_size[1:]), 0, classify=True)
        z, delta_logp = z_logp
    elif args.task == 'density':
        z, delta_logp = model(x.view(-1, *input_size[1:]), 0)
    elif args.task == 'classification':
        z, logits_tensor = model(x.view(-1, *input_size[1:]), classify=True)

    if args.task in ['density', 'hybrid']:
        # log p(z)
        logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)

        # log p(x)
        logpx = logpz - beta * delta_logp - np.log(nvals) * (
                args.imagesize * args.imagesize * (im_dim + args.padding)
        ) - logpu

        # bits_per_dim = -torch.mean(logpx) / (args.imagesize * args.imagesize * im_dim) / np.log(2)

        # bits_per_dim = -torch.mean(logpx) / (args.imagesize * args.imagesize * im_dim) / np.log(2)
        # bits_per_dim = -torch.mean(torch.exp(logpx)) / (args.imagesize * args.imagesize * im_dim) / np.log(2)

        # bits_per_dim = -torch.mean(torch.exp(logpx)) / (args.imagesize * args.imagesize * im_dim) / np.log(2)
        # bits_per_dim = -torch.mean(logpx) / (args.imagesize * args.imagesize * im_dim) / np.log(2)

        # bits_per_dim = torch.exp(torch.mean(logpx))
        bits_per_dim = torch.exp(torch.mean(logpx) / (args.imagesize * args.imagesize * im_dim) / np.log(2))

        logpz = torch.mean(logpz).detach()
        delta_logp = torch.mean(-delta_logp).detach()

    # return bits_per_dim, logits_tensor, logpz, delta_logp
    return bits_per_dim


"""
def computeLoss(x, model, beta=1.0):
    bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(n_classes).to(x)
    logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)

    if args.data == 'celeba_5bit':
        nvals = 32
    elif args.data == 'celebahq':
        nvals = 2**args.nbits
    else:
        nvals = 256

    #x, logpu = add_padding(x, nvals)
    _, logpu = add_padding(x, nvals)

    if args.squeeze_first:
        x = squeeze_layer(x)

    if args.task == 'hybrid':
        z_logp, logits_tensor = model(x.view(-1, *input_size[1:]), 0, classify=True)
        z, delta_logp = z_logp
    elif args.task == 'density':
        z, delta_logp = model(x.view(-1, *input_size[1:]), 0)
    elif args.task == 'classification':
        z, logits_tensor = model(x.view(-1, *input_size[1:]), classify=True)

    if args.task in ['density', 'hybrid']:
        # log p(z)
        logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)

        # log p(x)
        logpx = logpz - beta * delta_logp - np.log(nvals) * (
            args.imagesize * args.imagesize * (im_dim + args.padding)
        ) - logpu
        #bits_per_dim = -torch.mean(logpx) / (args.imagesize * args.imagesize * im_dim) / np.log(2)
        #bits_per_dim = -torch.mean(torch.exp(logpx)) / (args.imagesize * args.imagesize * im_dim) / np.log(2)
        bits_per_dim = torch.exp(torch.mean(logpx) / (args.imagesize * args.imagesize * im_dim) / np.log(2))

        logpz = torch.mean(logpz).detach()
        delta_logp = torch.mean(-delta_logp).detach()

    #return bits_per_dim, logits_tensor, logpz, delta_logp
    return bits_per_dim
"""


def compute_loss2(x, model, beta=1.0):
    # bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(n_classes).to(x)

    bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(n_classes).to(x)
    logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)

    '''
    if args.data == 'celeba_5bit':
        nvals = 32
    elif args.data == 'celebahq':
        nvals = 2**args.nbits
    else:
        nvals = 256
    '''

    nvals = 256

    # x, logpu = add_padding(x, nvals)

    # x, logpu = add_padding(x, nvals)
    # x, logpu = add_padding(x, nvals)

    # print(x.shape)
    # sadfasdfas

    # x, logpu = add_padding(x, nvals)
    # _, logpu = add_padding(x, nvals)

    x, logpu = add_padding(x, nvals)

    # x, logpu = add_padding(x, nvals)
    # _, logpu = add_padding(x, nvals)

    # print(args.squeeze_first)
    # dasfasdf

    if args.squeeze_first:
        x = squeeze_layer(x)

    # print(args.task)

    # print(x.shape)
    # print((x.view(-1, *input_size[1:])).shape)

    if args.task == 'hybrid':
        z_logp, logits_tensor = model(x.view(-1, *input_size[1:]), 0, classify=True)
        z, delta_logp = z_logp

    elif args.task == 'density':
        # z, delta_logp = model(x.view(-1, *input_size[1:]), 0)

        # z, delta_logp = model(x.view(-1, *input_size[1:]), 0)
        # z, delta_logp = model(x.view(-1, *input_size[1:]), 0)

        # z, delta_logp = model(x.view(-1, *input_size[1:]), 0)

        # z, delta_logp = model(x.view(-1, *input_size[1:]), 0)
        # z, delta_logp = model(x.view(-1, *input_size[1:]), 0.0)

        # prev_delta_logp = delta_logp

        # prev_delta_logp = delta_logp
        # z, delta_logp = model(x.view(-1, *input_size[1:]), 0.0)

        # z, delta_logp = model(x.view(-1, *input_size[1:]), 0.0)
        # z, delta_logp = model(x.view(-1, *input_size[1:]), -1e-7)

        # z, delta_logp = model(x.view(-1, *input_size[1:]), 0.0)
        z, delta_logp = model(x.view(-1, *input_size[1:]), 0.0)

        # if torch.isnan(torch.mean(z)):
        #   z = x.view(-1, x.squeeze().shape[1]**2)

        """
        if torch.isnan(torch.mean(delta_logp)):
            delta_logp = prev_delta_logp

        if torch.isnan(torch.mean(z)):
          z = x.view(-1, x.squeeze().shape[1]**2)
        """

        # asfdgsd

        # print(x.shape)
        # print(z.shape)
        # print(delta_logp.shape)

        # print('')
        # print(x)
        # print(z)
        # print(delta_logp)

        # z[torch.isnan(z)] = 0.0

        # print(x.shape)
        # print(x)

        """
        print(x.shape)
        print(torch.mean(x))

        #print(z.shape)
        #print(z)

        print(z.shape)
        print(torch.mean(z))
        """

        # t = torch.ones((2, 3, 4))
        # t.size()
        # torch.Size([2, 3, 4])
        # t.view(-1, 12).size()
        # torch.Size([2, 12])

        # print(z.shape)
        # print(z)

        # if torch.isnan(torch.mean(z)):
        #    z = x.view(-1, x.squeeze().shape[1]**2)

        # print(z.shape)
        # print(z)

        # print(z)
        # print(delta_logp)

        # asdfkdfs

        # print(z)
        # asdfa

        # print('')
        # print(z)

        # z[torch.isnan(z)] = 0.0
        # delta_logp[torch.isnan(delta_logp)] = 0.0

        # z[z != z] = 0.0
        # delta_logp[delta_logp != delta_logp] = 0.0

    elif args.task == 'classification':
        z, logits_tensor = model(x.view(-1, *input_size[1:]), classify=True)

    if args.task in ['density', 'hybrid']:
        # log p(z)

        # log p(z)
        logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)

        # log p(x)
        logpx = logpz - beta * delta_logp - np.log(nvals) * (
                args.imagesize * args.imagesize * (im_dim + args.padding)) - logpu

        # bits_per_dim = -torch.mean(logpx) / (args.imagesize * args.imagesize * im_dim) / np.log(2)

        # bits_per_dim = -torch.mean(logpx) / (args.imagesize * args.imagesize * im_dim) / np.log(2)
        # bits_per_dim = torch.mean(torch.exp(logpx)) / (args.imagesize * args.imagesize * im_dim) / np.log(2)

        # bits_per_dim = torch.mean(torch.exp(logpx)) / (args.imagesize * args.imagesize * im_dim) / np.log(2)

        # print(torch.mean(logpx))
        # print(torch.mean(torch.exp(logpx)))

        # bits_per_dim = torch.mean(torch.exp(logpx)) / (args.imagesize * args.imagesize * im_dim) / np.log(2)
        bits_per_dim = torch.mean(torch.exp(logpx)) / (args.imagesize * args.imagesize * im_dim) / np.log(2)

        # bits_per_dim = torch.mean(torch.exp(logpx)) / (args.imagesize * args.imagesize * im_dim) / np.log(2)
        # bits_per_dim = torch.mean(torch.exp(logpx))

        # logpz = torch.mean(logpz).detach()
        # logpz = torch.mean(torch.exp(logpx)).detach()

        # logpz = torch.mean(torch.exp(logpx)).detach()

        # logpz = torch.mean(logpz).detach()
        # delta_logp = torch.mean(-delta_logp).detach()

        # delta_logp = torch.mean(-delta_logp).detach()

    # return bits_per_dim, logits_tensor, logpz, delta_logp
    # return logpz

    # return logpz
    return bits_per_dim


"""
# x is a Tensor => batch_size x 2
def compute_loss2(x, args, model, batch_size=None, beta=1.):
   if batch_size is None:
       #batch_size = args.batch_size
       batch_size = args.batchsize

   zero = torch.zeros(x.shape[0], 1).to(x)

   # transform to z
   #z, delta_logp = model(x, zero)

   #z, delta_logp = model(x, zero)

   #z, delta_logp = model(x, zero)
   #z, delta_logp = x, zero

   #z, delta_logp = model(x, zero)
   z, delta_logp = model(x, zero)

   # x is a Tensor => batch_size x 2
   # x is the same as args.data => batch_size x 2

   # compute log p(z)
   logpz = standard_normal_logprob(z).sum(1, keepdim=True)

   # compute log p(x)
   logpx = logpz - beta * delta_logp

   #return torch.mean(logpx)
   return torch.mean(torch.exp(logpx))
   #return torch.mean(torch.log(torch.exp(logpx) / (1.0-torch.exp(logpx))))
"""


def estimator_moments(model, baseline=0):
    avg_first_moment = 0.
    avg_second_moment = 0.
    for m in model.modules():
        if isinstance(m, layers.iResBlock):
            avg_first_moment += m.last_firmom.item()
            avg_second_moment += m.last_secmom.item()
    return avg_first_moment, avg_second_moment


def compute_p_grads(model):
    scales = 0.
    nlayers = 0
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            scales = scales + m.compute_one_iter()
            nlayers += 1
    scales.mul(1 / nlayers).backward()
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            if m.domain.grad is not None and torch.isnan(m.domain.grad):
                m.domain.grad = None


batch_time = utils.RunningAverageMeter(0.97)
bpd_meter = utils.RunningAverageMeter(0.97)
logpz_meter = utils.RunningAverageMeter(0.97)
deltalogp_meter = utils.RunningAverageMeter(0.97)
firmom_meter = utils.RunningAverageMeter(0.97)
secmom_meter = utils.RunningAverageMeter(0.97)
gnorm_meter = utils.RunningAverageMeter(0.97)
ce_meter = utils.RunningAverageMeter(0.97)


def train(epoch, model):
    model = parallelize(model)
    model.train()

    total = 0
    correct = 0

    end = time.time()

    for i, (x, y) in enumerate(train_loader):

        # print(x.shape)
        # print(y.shape)

        # print(y)

        for i21 in range(len(y)):
            if y[i21] == 0 and i21 == 0:
                y[i21] = y[i21 + 1]
                x[i21, :, :, :] = x[i21 + 1, :, :, :]
            elif y[i21] == 0:
                y[i21] = y[i21 - 1]
                x[i21, :, :, :] = x[i21 - 1, :, :, :]

        # print(y)
        # asdfsf

        global_itr = epoch * len(train_loader) + i
        update_lr(optimizer, global_itr)

        # Training procedure:
        # for each sample x:
        #   compute z = f(x)
        #   maximize log p(x) = log p(z) - log |det df/dx|

        x = x.to(device)

        beta = beta = min(1, global_itr / args.annealing_iters) if args.annealing_iters > 0 else 1.
        bpd, logits, logpz, neg_delta_logp = compute_loss(x, model, beta=beta)

        if args.task in ['density', 'hybrid']:
            firmom, secmom = estimator_moments(model)

            bpd_meter.update(bpd.item())
            logpz_meter.update(logpz.item())
            deltalogp_meter.update(neg_delta_logp.item())
            firmom_meter.update(firmom)
            secmom_meter.update(secmom)

        if args.task in ['classification', 'hybrid']:
            y = y.to(device)
            crossent = criterion(logits, y)
            ce_meter.update(crossent.item())

            # Compute accuracy.
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        # compute gradient and do SGD step
        if args.task == 'density':
            loss = bpd
        elif args.task == 'classification':
            loss = crossent
        else:
            if not args.scale_dim: bpd = bpd * (args.imagesize * args.imagesize * im_dim)
            loss = bpd + crossent / np.log(2)  # Change cross entropy from nats to bits.
        loss.backward()

        if global_itr % args.update_freq == args.update_freq - 1:

            if args.update_freq > 1:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad /= args.update_freq

            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.)
            if args.learn_p: compute_p_grads(model)

            optimizer.step()
            optimizer.zero_grad()
            update_lipschitz(model)
            ema.apply()

            gnorm_meter.update(grad_norm)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            s = (
                'Epoch: [{0}][{1}/{2}] | Time {batch_time.val:.3f} | '
                'GradNorm {gnorm_meter.avg:.2f}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, gnorm_meter=gnorm_meter
                )
            )

            if args.task in ['density', 'hybrid']:
                s += (
                    ' | Bits/dim {bpd_meter.val:.4f}({bpd_meter.avg:.4f}) | '
                    'Logpz {logpz_meter.avg:.0f} | '
                    '-DeltaLogp {deltalogp_meter.avg:.0f} | '
                    'EstMoment ({firmom_meter.avg:.0f},{secmom_meter.avg:.0f})'.format(
                        bpd_meter=bpd_meter, logpz_meter=logpz_meter, deltalogp_meter=deltalogp_meter,
                        firmom_meter=firmom_meter, secmom_meter=secmom_meter
                    )
                )

            if args.task in ['classification', 'hybrid']:
                s += ' | CE {ce_meter.avg:.4f} | Acc {0:.4f}'.format(100 * correct / total, ce_meter=ce_meter)

            logger.info(s)
        if i % args.vis_freq == 0:
            visualize(epoch, model, i, x)

            # torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            # }, os.path.join(args.save, 'models', 'myMostRecent.pth'))

            # torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            # }, os.path.join(args.save, 'models', 'myMyMostRecent.pth'))

            # torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            # }, os.path.join(args.save, 'models', 'myMyMyMostRecent.pth'))

            # torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            # }, os.path.join(args.save, 'models', 'myMyMyMyMostRecent.pth'))

            # torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            # }, os.path.join(args.save, 'models', 'myMyMyMyMyMostRecent.pth'))

            # torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            # }, os.path.join(args.save, 'models', 'myMyMyMyMyMyMostRecent.pth'))

            # torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            # }, os.path.join(args.save, 'models', 'myMyMyMyMyMyMyMostRecent.pth'))

        del x
        torch.cuda.empty_cache()
        gc.collect()


def validate(epoch, model, ema=None):
    """
    Evaluates the cross entropy between p_data and p_model.
    """
    bpd_meter = utils.AverageMeter()
    ce_meter = utils.AverageMeter()

    if ema is not None:
        ema.swap()

    update_lipschitz(model)

    model = parallelize(model)
    model.eval()

    correct = 0
    total = 0

    start = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_loader)):
            x = x.to(device)
            bpd, logits, _, _ = compute_loss(x, model)
            bpd_meter.update(bpd.item(), x.size(0))

            if args.task in ['classification', 'hybrid']:
                y = y.to(device)
                loss = criterion(logits, y)
                ce_meter.update(loss.item(), x.size(0))
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
    val_time = time.time() - start

    if ema is not None:
        ema.swap()
    s = 'Epoch: [{0}]\tTime {1:.2f} | Test bits/dim {bpd_meter.avg:.4f}'.format(epoch, val_time, bpd_meter=bpd_meter)
    if args.task in ['classification', 'hybrid']:
        s += ' | CE {:.4f} | Acc {:.2f}'.format(ce_meter.avg, 100 * correct / total)
    logger.info(s)
    return bpd_meter.avg


def visualize(epoch, model, itr, real_imgs):
    model.eval()
    utils.makedirs(os.path.join(args.save, 'imgs'))
    real_imgs = real_imgs[:32]
    _real_imgs = real_imgs

    if args.data == 'celeba_5bit':
        nvals = 32
    elif args.data == 'celebahq':
        nvals = 2 ** args.nbits
    else:
        nvals = 256

    with torch.no_grad():
        # reconstructed real images
        real_imgs, _ = add_padding(real_imgs, nvals)
        if args.squeeze_first: real_imgs = squeeze_layer(real_imgs)
        recon_imgs = model(model(real_imgs.view(-1, *input_size[1:])), inverse=True).view(-1, *input_size[1:])
        if args.squeeze_first: recon_imgs = squeeze_layer.inverse(recon_imgs)
        recon_imgs = remove_padding(recon_imgs)

        # random samples
        fake_imgs = model(fixed_z, inverse=True).view(-1, *input_size[1:])
        if args.squeeze_first: fake_imgs = squeeze_layer.inverse(fake_imgs)
        fake_imgs = remove_padding(fake_imgs)

        fake_imgs = fake_imgs.view(-1, im_dim, args.imagesize, args.imagesize)
        recon_imgs = recon_imgs.view(-1, im_dim, args.imagesize, args.imagesize)
        imgs = torch.cat([_real_imgs, fake_imgs, recon_imgs], 0)

        # filename = os.path.join(args.save, 'imgs', 'e{:03d}_i{:06d}.png'.format(epoch, itr))

        # filename = os.path.join(args.save, 'imgs', 'e{:03d}_i{:06d}.png'.format(epoch, itr))
        # filename = os.path.join(args.save, 'imgs', 'e{:03d}_i{:06d}.png'.format(epoch, itr))

        # filename = os.path.join(args.save, 'imgs', 'e{:03d}_i{:06d}.png'.format(epoch, itr))
        filename = os.path.join(args.save, 'imgs', 'ee{:03d}i{:06d}.png'.format(epoch, itr))

        save_image(imgs.cpu().float(), filename, nrow=16, padding=2)

        print(filename)
        print('')

    model.train()


def get_lipschitz_constants(model):
    lipschitz_constants = []
    for m in model.modules():
        if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
            lipschitz_constants.append(m.scale)
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            lipschitz_constants.append(m.scale)
        if isinstance(m, base_layers.LopConv2d) or isinstance(m, base_layers.LopLinear):
            lipschitz_constants.append(m.scale)
    return lipschitz_constants


def update_lipschitz(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
                m.compute_weight(update=True)
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                m.compute_weight(update=True)


def get_ords(model):
    ords = []
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            domain, codomain = m.compute_domain_codomain()
            if torch.is_tensor(domain):
                domain = domain.item()
            if torch.is_tensor(codomain):
                codomain = codomain.item()
            ords.append(domain)
            ords.append(codomain)
    return ords


def pretty_repr(a):
    return '[[' + ','.join(list(map(lambda i: '{}'.format(i), a))) + ']]'


def main():
    global best_test_bpd

    last_checkpoints = []
    lipschitz_constants = []
    ords = []

    # if args.resume:
    #     validate(args.begin_epoch - 1, model, ema)
    # for epoch in range(args.begin_epoch, args.nepochs):

    # for epoch in range(args.begin_epoch, args.nepochs):
    for epoch in range(0, 0):

        logger.info('Current LR {}'.format(optimizer.param_groups[0]['lr']))

        train(epoch, model)
        lipschitz_constants.append(get_lipschitz_constants(model))

        ords.append(get_ords(model))
        logger.info('Lipsh: {}'.format(pretty_repr(lipschitz_constants[-1])))
        logger.info('Order: {}'.format(pretty_repr(ords[-1])))

        if args.ema_val:
            test_bpd = validate(epoch, model, ema)
        else:
            test_bpd = validate(epoch, model)

        if args.scheduler and scheduler is not None:
            scheduler.step()

        if test_bpd < best_test_bpd:
            best_test_bpd = test_bpd
            utils.save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'ema': ema,
                'test_bpd': test_bpd,
            }, os.path.join(args.save, 'models'), epoch, last_checkpoints, num_checkpoints=5)

        # torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        # }, os.path.join(args.save, 'models', 'theMostRecent.pth'))

        # torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        # }, os.path.join(args.save, 'models', 'theTheMostRecent.pth'))

        # torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        # }, os.path.join(args.save, 'models', 'theTheTheMostRecent.pth'))

        # torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        # }, os.path.join(args.save, 'models', 'theTheTheTheMostRecent.pth'))

        # torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        # }, os.path.join(args.save, 'models', 'theTheTheTheTheMostRecent.pth'))

        # torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        # }, os.path.join(args.save, 'models', 'theTheTheTheTheTheMostRecent.pth'))

        # torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        # }, os.path.join(args.save, 'models', 'theTheTheTheTheTheTheMostRecent.pth'))

        # torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        # }, os.path.join(args.save, 'models', 'theTheTheTheTheTheTheTheMostRecent.pth'))

        # torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        # }, os.path.join(args.save, 'models', 'theTheTheTheTheTheTheTheTheMostRecent.pth'))

    """
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    genGen.to(device)
    genGen.train()

    #for itr in range(1, args.niters2 + 1):
    #for itr in range(1, args.nepochs + 1):

    time2_meter = utils.RunningAverageMeter(0.93)
    loss2_meter = utils.RunningAverageMeter(0.93)

    end = time.time()
    best_loss = float('inf')
    """

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    """
    #genGen = genGen.to(device)

    checkpoint2 = torch.load(os.path.join(os.path.join(args.save, '0000000000IIIIIIRRNbs64model771'), 'checkpt-%04d.pth' % args.niters2))
    genGen.load_state_dict(checkpoint2['state_dict'])

    genGen = genGen.to(device)
    optimizerGen.load_state_dict(checkpoint2['optimizer_state_dict'])
    """

    # utils.save_checkpoint(
    #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
    #     'args': args},
    #    os.path.join(args.save, 'mainMainMainMyMyBest'), args.nepochs)

    # checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'mainMainMainMyMyBest'), 'checkpt-%04d.pth' % args.nepochs))

    # checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'mainMainMainMyMyBest'), 'checkpt-%04d.pth' % args.nepochs))
    # checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'mainMainMainMyMyBest'), 'checkpt-%04d.pth' % args.nepochs))

    # checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'mainMainMainMyMyBest'), 'checkpt-%04d.pth' % args.nepochs))
    # checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maainMaainMaainMyMyMyBest'), 'checkpt-%04d.pth' % args.nepochs))

    '''
    checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maainMaainMaainMyMyMyBest'), 'checkpt-%04d.pth' % args.nepochs))
    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiinMaaiinMaaiinMyMyMyMyBest'), 'checkpt-%04d.pth' % args.nepochs))

    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiinMaaiinMaaiinMyMyMyMyBest'), 'checkpt-%04d.pth' % args.nepochs))

    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiinMaaiinMaaiinMyMyMyMyBest'), 'checkpt-%04d.pth' % args.nepochs))
    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiinMaaiinMaaiinMyMyMyMyMyBeBest'), 'checkpt-%04d.pth' % args.nepochs))

    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiinMaaiinMaaiinMyMyMyMyMyBeBest'), 'checkpt-%04d.pth' % args.nepochs))
    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiiinMaaiiinMyMyMyMyMyBeBest'), 'checkpt-%04d.pth' % args.nepochs))

    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiiinMaaiiinMyMyMyMyMyBeBest'), 'checkpt-%04d.pth' % args.nepochs))
    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiiinMaaiiinMyMyMyMyMyBeBest'), 'checkpt-%04d.pth' % args.nepochs))

    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiiinMaaiiinMyMyMyMyMyBeBest'), 'checkpt-%04d.pth' % args.nepochs))
    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiiinMaaiiinMyMyMyBest9'), 'checkpt-%04d.pth' % args.nepochs))

    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiiinMaaiiinMyMyMyBest9'), 'checkpt-%04d.pth' % args.nepochs))
    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiiinMaaiiinMaaiiinMyBest99'), 'checkpt-%04d.pth' % args.nepochs))

    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiiinMaaiiinMaaiiinMyBest99'), 'checkpt-%04d.pth' % args.nepochs))
    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiiinMaaiiinMaaiiinMyBest99'), 'checkpt-%04d.pth' % args.nepochs))

    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiiinMaaiiinMaaiiinMyBest99'), 'checkpt-%04d.pth' % args.nepochs))
    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiiinNikNikMaaiiinMaaiiinBest'), 'checkpt-%04d.pth' % args.nepochs))

    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiiinNikNikMaaiiinMaaiiinBest'), 'checkpt-%04d.pth' % args.nepochs))
    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'maaiiinNikNik99MaaiiinMaaiiinBest'), 'checkpt-%04d.pth' % args.nepochs))

    #checkpoint2 = torch.load(os.path.join(os.path.join(args.save, 'mainMainMainMyMyBest'), 'checkpt-%04d.pth' % args.nepochs))
    genGen.load_state_dict(checkpoint2['state_dict'])

    #genGen = genGen.to(device)
    genGen.to(device)

    #genGen = genGen.to(device)
    optimizerGen.load_state_dict(checkpoint2['optimizer_state_dict'])
    '''

    # genGen.to(device)

    # genGen.to(device)
    # genGen.to(device)

    # utils.save_checkpoint(
    #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
    #     'args': args},
    #    os.path.join(args.save, 'maaiiinNikNik99MaaiiinMaaiiinBeBe'), args.nepochs)

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, 'maaiiinNikNik99MaaiiinMaaiiinBeBe'), 'checkpt-%04d.pth' % args.nepochs))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, 'maaiiinNikNik99MaaiiinMaaiiinBeBe'), 'checkpt-%04d.pth' % args.nepochs))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '0mainNikNik99MaaiiinMaaiiinBeBe'), 'checkpt-%04d.pth' % args.nepochs))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '0mainNikNik99MaaiiinMaaiiinBeBe'), 'checkpt-%04d.pth' % 1000))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '00mainNikNik99MaaiiinMaaiiinBeBe'), 'checkpt-%04d.pth' % 5))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '000mmainNikNik99MaaiiinMaaiiinBeBe'), 'checkpt-%04d.pth' % 5))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '0000mmainNikNik99MaaiiinMaaiiinBeBe'), 'checkpt-%04d.pth' % 1000))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '00000mmmainNikNik99MaaiiinMaaiiinBeBe'), 'checkpt-%04d.pth' % 500))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '000000mmmmainNikNik99MaaiiinMaaiiinBeBe'), 'checkpt-%04d.pth' % 10000))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '0000000mmmmmainNikNik99MaaiiinMaaiiinBeBe'), 'checkpt-%04d.pth' % 10000))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '00000000mmmmmmaainNikNik99MaaiiinMaaiiinBeBe'), 'checkpt-%04d.pth' % 10000))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '000000000mmmmmmmaaainNikNik99MaaiiinMaaiiinBeBe'),
    #                 'checkpt-%04d.pth' % 10000))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '0000000000mmmmmmmmaaainNikNik99MaaiiinMaaiiinBeBe'),
    #                 'checkpt-%04d.pth' % 10000))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '00000000000mmmmmmmmmaaainNikNik99MaaiiinMaaiiinBeBe'),
    #                 'checkpt-%04d.pth' % 10000))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '000000000000mmmmmmmmmmaaaaiinNikNik99MaaiiinMaaiiinBeBe'),
    #                 'checkpt-%04d.pth' % 10000))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '0000000000000mmmmmmmmmmmaaaaiinNikNik99MaaiiinMaaiiinBeBe'),
    #                 'checkpt-%04d.pth' % 10000))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '00000000000000mmmmmmmmmmmmaaaaiinNikNik99MaaiiinMaaiiinBeBe'),
    #                 'checkpt-%04d.pth' % 10000))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '000000000000000mmmmmmmmmmmmmaaaaiinNikNik99MaaiiinMaaiiinBeBe'),
    #                 'checkpt-%04d.pth' % 10000))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '000000000000000mmmmmmmmmmmmmmaaaaiinNikNik99MaaiiinMaaiiinBe2Be2'),
    #                 'checkpt-%04d.pth' % 10000))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '0000000000000000mmmmmmmmmmmmmmmaaaaiinNikNik99MaaiiinMaaiiinBe2Be2'),
    #                 'checkpt-%04d.pth' % 10000))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '00000000000000000mmmmmmmmmmmmmmmmaaaaaiinNikNik99MaaiiinMaaiiinBe2Be2'),
    #                 'checkpt-%04d.pth' % 10000))

    # checkpoint2 = torch.load(
    #    os.path.join(os.path.join(args.save, '000000000000000000mmmmmmmmmmmmmmmmmaaaaaiinNikNik99MaaiiinMaaiiinBe2Be2'),
    #                 'checkpt-%04d.pth' % 10000))

    # checkpoint2 = torch.load(
    #    os.path.join(
    #        os.path.join(args.save, '0000000000000000000mmmmmmmmmmmmmmmmmmaaaaaiinNikNik99MaaiiinMaaiiinBe2Be2'),
    #        'checkpt-%04d.pth' % 10000))

    # utils.save_checkpoint(
    #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
    #     'args': args}, os.path.join(args.save, '000mmainNikNik99MaaiiinMaaiiinBeBe'), args.nepochs)

    ## Store Best
    # utils.save_checkpoint(
    #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
    #     'args': args},
    #    os.path.join(args.save, '0mainNikNik99MaaiiinMaaiiinBeBe'), args.nepochs)

    genGen.to(device)
    genGen.train()

    # genGen.load_state_dict(checkpoint2['state_dict'])
    # genGen.to(device)

    # optimizerGen.load_state_dict(checkpoint2['optimizer_state_dict'])
    # genGen.train()

    # genGen.to(device)
    # genGen.train()

    # for itr in range(1, args.niters2 + 1):
    # for itr in range(1, args.nepochs + 1):

    time2_meter = utils.RunningAverageMeter(0.93)
    loss2_meter = utils.RunningAverageMeter(0.93)

    filoss2_meter = utils.RunningAverageMeter(0.93)
    seloss2_meter = utils.RunningAverageMeter(0.93)
    thloss2_meter = utils.RunningAverageMeter(0.93)

    gr1loss2_meter = utils.RunningAverageMeter(0.93)
    gr2loss2_meter = utils.RunningAverageMeter(0.93)
    gr3loss2_meter = utils.RunningAverageMeter(0.93)

    arloss2_meter = []
    arfiloss2_meter = []

    arseloss2_meter = []
    arthloss2_meter = []

    argr1loss2_meter = []
    argr2loss2_meter = []
    argr3loss2_meter = []

    end = time.time()
    best_loss = float('inf')

    """
    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            running_loss = + loss.item() * images.size(0)

        loss_values.append(running_loss / len(train_dataset))

    plt.figure()
    plt.plot(loss_values)

    #plt.savefig('foo.png')
    plt.savefig('foo.png', bbox_inches='tight')
    """

    """
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0

        for data in dataLoader:
            images, labels = data

            outputs = model(images)
            loss = criterion_label(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloaders['train'])
        losses.append(epoch_loss)
    """

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    genGen.eval()

    for param in genGen.parameters():
        param.requires_grad = False

    losses_NIKlosses = []

    loLosses_NIKlosses = []
    loLosses_NIKlosses2 = []

    # loLosses_NIKlosses2 = []
    loLosses_NIKlosses3 = []

    # for itr in range(1, args.nepochs + 1):

    # for itr in range(1, args.nepochs + 1):
    for itr in range(1, 1 + 1):
        runningLoss_NIKrunningLoss = 0.0

        # for i, (x, y) in enumerate(train_loader):

        # for i, (x, y) in enumerate(train_loader):
        for i, (x, y) in enumerate(test_loader):
            # x = x.to(device)

            # print(x)
            # print(x.shape)

            # print(y)
            # print(y.item())

            # for i21 in range(len(y)):
            #    if y[i21] == 0 and i21 == 0:
            #        y[i21] = y[i21+1]
            #        x[i21, :, :, :] = x[i21+1, :, :, :]
            #    elif y[i21] == 0:
            #        y[i21] = y[i21 - 1]
            #        x[i21, :, :, :] = x[i21 - 1, :, :, :]

            # if i > 0:
            #    if y.item() == 0:
            #        y = y_prevPrev
            #        x = x_prevPrev

            '''
            for i21 in range(len(y)):
                if y[i21] == 0 and i21 == 0:
                    y[i21] = y[i21+1]
                    x[i21, :, :, :] = x[i21+1, :, :, :]
                elif y[i21] == 0:
                    y[i21] = y[i21 - 1]
                    x[i21, :, :, :] = x[i21 - 1, :, :, :]
            '''

            x = x.to(device)
            print(i)

            # print(x.shape)
            # asdfsadfs

            genFGen2 = x
            # lossGen, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(genFGen2, args, model, ggenFGen2, x)

            # use: val-batchsize
            ggenFGen2 = torch.randn([args.val_batchsize, nrand], device=device)

            # ggenFGen2 = torch.randn([args.batchsize, nrand], device=device)
            # ggenFGen2 = torch.randn([args.batchsize, nrand], device=device, requires_grad=True)

            # with torch.no_grad():
            #    _, firstOnly_lossGen, _, _ = use_loss_fn2(genFGen2, args, model, ggenFGen2, x)

            # loLosses_NIKlosses.append(firstOnly_lossGen.item())

            # print(firstOnly_lossGen)
            # print(loLosses_NIKlosses)

            with torch.no_grad():
                firstOnly_lossGen2 = computeLoss(x, model)

            # print(y)
            # print(y.item())

            # print(firstOnly_lossGen2)
            # print(firstOnly_lossGen2.item())

            # asdfasdfs

            loLosses_NIKlosses2.append(firstOnly_lossGen2.item())

            # print(y)
            # print(y.item())

            # if y.item() == 0:
            # if y.item() == 1:

            #if y.item() == 1:
            if y.item() == 0:
                loLosses_NIKlosses3.append(0)

                # print(y)
                # print(y.item())

            else:
                loLosses_NIKlosses3.append(1)

            # loLosses_NIKlosses3.append(1)

            '''
            if y.item() == 0:
                loLosses_NIKlosses3.append(0)

                #print(y)
                #print(y.item())

            else:
                loLosses_NIKlosses3.append(1)
            '''

            # print(y)
            # print(y.item())

            # print(firstOnly_lossGen2)
            # print(loLosses_NIKlosses2)

            # x_prevPrev = x
            # y_prevPrev = y

    # print(loLosses_NIKlosses)
    # print(loLosses_NIKlosses2)

    # print(loLosses_NIKlosses2)
    # print(loLosses_NIKlosses3)

    # use: test_loLoader
    # now use test_loLoader

    """
    for itr in range(1, 1 + 1):
        runningLoss_NIKrunningLoss = 0.0

        # for i, (x, y) in enumerate(train_loader):

        # for i, (x, y) in enumerate(train_loader):
        for i, (x, y) in enumerate(test_loLoader):
            # x = x.to(device)

            # for i21 in range(len(y)):
            #    if y[i21] == 0 and i21 == 0:
            #        y[i21] = y[i21+1]
            #        x[i21, :, :, :] = x[i21+1, :, :, :]
            #    elif y[i21] == 0:
            #        y[i21] = y[i21 - 1]
            #        x[i21, :, :, :] = x[i21 - 1, :, :, :]

            # if i > 0:
            #    if y.item() == 0:
            #        y = y_prevPrev
            #        x = x_prevPrev

            '''
            for i21 in range(len(y)):
                if y[i21] == 0 and i21 == 0:
                    y[i21] = y[i21+1]
                    x[i21, :, :, :] = x[i21+1, :, :, :]
                elif y[i21] == 0:
                    y[i21] = y[i21 - 1]
                    x[i21, :, :, :] = x[i21 - 1, :, :, :]
            '''

            x = x.to(device)
            print(i)

            # print(x.shape)
            # asdfsadfs

            genFGen2 = x
            # lossGen, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(genFGen2, args, model, ggenFGen2, x)

            # use: val-batchsize
            ggenFGen2 = torch.randn([args.val_batchsize, nrand], device=device)

            # ggenFGen2 = torch.randn([args.batchsize, nrand], device=device)
            # ggenFGen2 = torch.randn([args.batchsize, nrand], device=device, requires_grad=True)

            # with torch.no_grad():
            #    _, firstOnly_lossGen, _, _ = use_loss_fn2(genFGen2, args, model, ggenFGen2, x)

            # loLosses_NIKlosses.append(firstOnly_lossGen.item())

            # print(firstOnly_lossGen)
            # print(loLosses_NIKlosses)

            with torch.no_grad():
                firstOnly_lossGen2 = computeLoss(x, model)

            loLosses_NIKlosses2.append(firstOnly_lossGen2.item())

            # print(y)
            # print(y.item())

            loLosses_NIKlosses3.append(0)
    """

    # print(loLosses_NIKlosses)
    # print(loLosses_NIKlosses2)

    # print(loLosses_NIKlosses2)
    # print(loLosses_NIKlosses3)

    import numpy as np
    # import pandas as pd
    import matplotlib.pyplot as plt
    # import seaborn as sns

    # ROC curve and auc score
    from sklearn.datasets import make_classification
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score

    # def plot_roc_curve(fpr, tpr):

    # def plot_roc_curve(fpr, tpr):
    def plot_roc_curve(fpr, tpr, auroc21):
        # plt.plot(fpr, tpr, color='orange', label='ROC')

        # plt.plot(fpr, tpr, color='orange', label='ROC')
        # plt.plot(fpr, tpr, color='orange', label='ROC')

        # plt.plot(fpr, tpr, color='orange', label='ROC (AUROC = {0:.3f})'.format(auroc21))
        # plt.plot(fpr, tpr, color='orange', label='ROC (AUROC = {0:.3f})'.format(auroc21))

        # plt.plot(fpr, tpr, color='orange', label='ROC (AUROC = {0:.3f})'.format(auroc21))
        plt.plot(fpr, tpr, color='orange', label='ROC (AUROC = {0:.4f})'.format(auroc21))

        # plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()

        # plt.savefig('ROC_MainROC.png', bbox_inches='tight')
        # plt.savefig('nikROC_MainROC.png', bbox_inches='tight')

        # plt.savefig('nikROC_MainROC.png', bbox_inches='tight')
        # plt.savefig('nikNikROC_MainROC.png', bbox_inches='tight')

        # plt.savefig('nikNikROC_MainROC.png', bbox_inches='tight')
        # plt.savefig('nikNikNikROC_MainROC.png', bbox_inches='tight')

        # plt.savefig('nikNikNikROC_MainROC.png', bbox_inches='tight')
        # plt.savefig('nik000NikNikROC_MainROC.png', bbox_inches='tight')

        # plt.savefig('nik000NikNikROC_MainROC.png', bbox_inches='tight')
        plt.savefig('cifarForFor0For0_MainROC.png', bbox_inches='tight')

        # plt.show()
        # plt.pause(99)

        # plt.savefig('ROC_MainROC.png', bbox_inches='tight')
        # plt.savefig('mainMainROC_MainROC.png', bbox_inches='tight')

        # plt.savefig('mainMainROC_MainROC.png', bbox_inches='tight')
        # plt.savefig('nikMainMainROC_MainROC.png', bbox_inches='tight')

        # plt.savefig('nikMainMainROC_MainROC.png', bbox_inches='tight')
        # plt.savefig('nikNikMainMainROC_MainROC.png', bbox_inches='tight')

        # plt.pause(9)
        # plt.ion()

    # print(loLossNoChange)
    # asdfkdfs

    # print(loLoss2)
    # print(loLossNoChange)

    # loLoss2 is 0 and 1
    # loLossNoChange is probability

    # loLoss2 = ground truth 0 and 1
    # roc_curve(loLoss2, loLossNoChange)

    # loLoss2 is the ground truth 0 and 1
    # loLossNoChange is the predicted probabilities

    # loLossNoChange = predicted probabilities
    loLossNoChange = loLosses_NIKlosses2

    # loLoss2 = ground truth 0 and 1
    loLoss2 = loLosses_NIKlosses3

    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score

    # print(average_precision_score(loLoss2, loLossNoChange))
    precision, recall, thresholds = precision_recall_curve(loLoss2, loLossNoChange)

    print(average_precision_score(loLoss2, loLossNoChange))
    print('')

    print(precision)
    print(recall)

    print('')
    print(thresholds)

    # def plot_pr_curve(fpr, tpr):

    # def plot_pr_curve(fpr, tpr):
    def plot_pr_curve(fpr, tpr, auroc21):
        # plt.plot(fpr, tpr, color='orange', label='PR')

        # plt.plot(fpr, tpr, color='orange', label='PR')
        # plt.plot(tpr, fpr, color='orange', label='PR')

        # plt.plot(fpr, tpr, color='orange', label='ROC (AUROC = {0:.3f})'.format(auroc21))
        # plt.plot(tpr, fpr, color='orange', label='PR (AUPRC = {0:.3f})'.format(auroc21))

        # plt.plot(tpr, fpr, color='orange', label='PR (AUPRC = {0:.3f})'.format(auroc21))
        plt.plot(tpr, fpr, color='orange', label='PR (AUPRC = {0:.4f})'.format(auroc21))

        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')

        plt.xlabel('Recall')
        plt.ylabel('Precision')

        plt.title('Precision Recall (PR) Curve')
        plt.legend()

        # plt.savefig('ROC_MainROC.png', bbox_inches='tight')
        # plt.savefig('nikPR_MainPR.png', bbox_inches='tight')

        # plt.savefig('nikPR_MainPR.png', bbox_inches='tight')
        # plt.savefig('nikNikPR_MainPR.png', bbox_inches='tight')

        # plt.savefig('nikNikPR_MainPR.png', bbox_inches='tight')
        # plt.savefig('nikNikNikPR_MainPR.png', bbox_inches='tight')

        # plt.savefig('nikNikNikPR_MainPR.png', bbox_inches='tight')
        # plt.savefig('nik000NikNikPR_MainPR.png', bbox_inches='tight')

        # plt.savefig('nik000NikNikPR_MainPR.png', bbox_inches='tight')
        plt.savefig('cifarFor0For0_MainPR.png', bbox_inches='tight')

        # plt.savefig('22Jan2020foFo.png', bbox_inches='tight')
        # plt.savefig('000000000000000fffffffffffffffoooFoo.png', bbox_inches='tight')

        # plt.show()
        # plt.pause(99)

        # plt.savefig('ROC_MainROC.png', bbox_inches='tight')
        # plt.savefig('mainMainROC_MainROC.png', bbox_inches='tight')

        # plt.savefig('mainMainROC_MainROC.png', bbox_inches='tight')
        # plt.savefig('nikMainMainPR_MainPR.png', bbox_inches='tight')

        # plt.savefig('nikMainMainPR_MainPR.png', bbox_inches='tight')
        # plt.savefig('nikNikMainMainPR_MainPR.png', bbox_inches='tight')

        # plt.pause(9)
        # plt.ion()

    # plot_pr_curve(precision, recall)

    # plot_pr_curve(precision, recall)
    plot_pr_curve(precision, recall, average_precision_score(loLoss2, loLossNoChange))

    # plot_pr_curve(precision, recall)
    plt.figure()

    print('')
    print(average_precision_score(loLoss2, loLossNoChange))

    print('')

    # probs = loLossNoChange
    fpr, tpr, thresholds = roc_curve(loLoss2, loLossNoChange)

    print(fpr)
    print(tpr)

    print('')
    print(thresholds)

    # fpr, tpr, thresholds = roc_curve(loLoss2, probs)
    # plot_roc_curve(fpr, tpr)

    # plot_roc_curve(fpr, tpr)

    # plot_roc_curve(fpr, tpr)
    # plot_roc_curve(fpr, tpr)

    # plot_roc_curve(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc_score(loLoss2, loLossNoChange))

    # print(roc_auc_score(fpr, tpr))
    # print(sklearn.metrics.auc(fpr, tpr))

    print('')
    print(roc_auc_score(loLoss2, loLossNoChange))

    from sklearn.metrics import auc
    # roc_auc = auc(fpr, tpr)

    print(auc(fpr, tpr))
    # roc_auc = auc(fpr, tpr)

    print('')
    # roc_auc = auc(fpr, tpr)

    '''
    plt.figure()
    #plt.plot(fpr[2], tpr[2], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot(fpr[2], tpr[2], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')
    #plt.legend(loc="lower right")

    plt.legend(loc="lower right")
    plt.show()
    '''

    def plot_roc_curve2(fpr, tpr, auroc21, fpr2, tpr2, auroc212):
        # plt.plot(fpr, tpr, color='orange', label='ROC (AUROC = {0:.4f})'.format(auroc21))

        # plt.plot(tpr, fpr, color='blue', label='PR (AUPRC = {0:.4f})'.format(auroc21))
        # plt.plot(tpr2, fpr2, color='blue', label='PR (AUPRC = {0:.4f})'.format(auroc212))

        plt.plot(fpr, tpr, color='orange', label='ROC (AUROC = {0:.4f})'.format(auroc21))
        plt.plot(tpr2, fpr2, color='blue', label='PR (AUPRC = {0:.4f})'.format(auroc212))

        # plt.plot(fpr, tpr, color='orange', label='ROC (AUROC = {0:.4f})'.format(auroc21))
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

        plt.xlabel('False Positive Rate (and Recall)')
        plt.ylabel('True Positive Rate (and Precision)')

        plt.title('ROC and PR Curves')
        plt.legend()

        # plt.plot(tpr, fpr, color='orange', label='PR (AUPRC = {0:.4f})'.format(auroc21))

        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')

        # plt.xlabel('Recall')
        # plt.ylabel('Precision')

        # plt.title('Precision Recall (PR) Curve')
        # plt.legend()

        # plt.savefig('nik00000NikNikROC_MainROC.png', bbox_inches='tight')

        # plt.savefig('nik00000NikNikROC_MainROC.png', bbox_inches='tight')
        # plt.savefig('nikNik00000nikNikNikROC_MainROC.png', bbox_inches='tight')

        # plt.savefig('nikNik00000nikNikNikROC_MainROC.png', bbox_inches='tight')
        plt.savefig('cifarFor0For0For0_MainROCPR.png', bbox_inches='tight')

    plt.figure()
    # plot_roc_curve2(fpr, tpr, roc_auc_score(loLoss2, loLossNoChange))

    # use: precision, recall, average_precision_score(loLoss2, loLossNoChange)
    # plot_roc_curve2(fpr, tpr, roc_auc_score(loLoss2, loLossNoChange), precision, recall, average_precision_score(loLoss2, loLossNoChange))

    plot_roc_curve2(fpr, tpr, roc_auc_score(loLoss2, loLossNoChange), precision, recall,
                    average_precision_score(loLoss2, loLossNoChange))

    # 0.7657142857142857
    # 0.7657142857142857

    # 0.7714285714285714
    # 0.7947712113075085

    # 0.7658408636296418

    # Data_j for MNIST digit j
    # ResFlow: See if p_g(x) works



    asdktdftksa



    for itr in range(1, args.nepochs + 1):
        runningLoss_NIKrunningLoss = 0.0

        for i, (x, y) in enumerate(train_loader):
            # print(x.shape)
            # print(y.shape)

            # print(y)

            for i21 in range(len(y)):
                if y[i21] == 0 and i21 == 0:
                    y[i21] = y[i21 + 1]
                    x[i21, :, :, :] = x[i21 + 1, :, :, :]
                elif y[i21] == 0:
                    y[i21] = y[i21 - 1]
                    x[i21, :, :, :] = x[i21 - 1, :, :, :]

            # print(y)
            # asdfsf

            # print(x.shape)
            # print(y.shape)

            '''
            if (itr==1) and (i==0):
                print((torch.mean(genGen.lin1.weight) + torch.mean(genGen.lin2.weight) + torch.mean(
                    genGen.dc1.weight) + torch.mean(genGen.dc2.weight) + torch.mean(
                    genGen.dc3.weight) + torch.mean(genGen.lin1.bias) + torch.mean(
                    genGen.lin2.bias) + torch.mean(
                    genGen.dc1.bias) + torch.mean(genGen.dc2.bias) + torch.mean(genGen.dc3.bias)).item())

                print((torch.mean(genGen.lin1bn.weight) + torch.mean(genGen.lin2bn.weight) + torch.mean(
                    genGen.dc1bn.weight) + torch.mean(genGen.dc2bn.weight) + torch.mean(
                    genGen.dc3.weight) + torch.mean(genGen.lin1bn.bias) + torch.mean(
                    genGen.lin2bn.bias) + torch.mean(
                    genGen.dc1bn.bias) + torch.mean(genGen.dc2bn.bias) + torch.mean(genGen.dc3.bias)).item())

                adkfdf
            '''

            # print((torch.mean(genGen.lin1.weight.grad) + torch.mean(genGen.lin2.weight.grad) + torch.mean(
            #                 genGen.dc1.weight.grad) + torch.mean(genGen.dc2.weight.grad) + torch.mean(
            #                 genGen.dc3.weight.grad) + torch.mean(genGen.lin1.bias.grad) + torch.mean(
            #                 genGen.lin2.bias.grad) + torch.mean(
            #                 genGen.dc1.bias.grad) + torch.mean(genGen.dc2.bias.grad) + torch.mean(genGen.dc3.bias.grad)).item())

            # print((torch.mean(genGen.lin1.weight.data) + torch.mean(genGen.lin2.weight.data) + torch.mean(
            #    genGen.dc1.weight.data) + torch.mean(genGen.dc2.weight.data) + torch.mean(
            #    genGen.dc3.weight.data) + torch.mean(genGen.lin1.bias.data) + torch.mean(
            #    genGen.lin2.bias.data) + torch.mean(
            #    genGen.dc1.bias.data) + torch.mean(genGen.dc2.bias.data) + torch.mean(genGen.dc3.bias.data)).item())

            '''
            print((torch.mean(genGen.lin1.weight) + torch.mean(genGen.lin2.weight) + torch.mean(
                genGen.dc1.weight) + torch.mean(genGen.dc2.weight) + torch.mean(
                genGen.dc3.weight) + torch.mean(genGen.lin1.bias) + torch.mean(
                genGen.lin2.bias) + torch.mean(
                genGen.dc1.bias) + torch.mean(genGen.dc2.bias) + torch.mean(genGen.dc3.bias)).item())
            '''

            # if (itr==1) and (i==1):
            #    print((torch.mean(genGen.lin1.weight.grad) + torch.mean(genGen.lin2.weight.grad) + torch.mean(
            #                     genGen.dc1.weight.grad) + torch.mean(genGen.dc2.weight.grad) + torch.mean(
            #                     genGen.dc3.weight.grad) + torch.mean(genGen.lin1.bias.grad) + torch.mean(
            #                     genGen.lin2.bias.grad) + torch.mean(
            #                     genGen.dc1.bias.grad) + torch.mean(genGen.dc2.bias.grad) + torch.mean(genGen.dc3.bias.grad)).item())

            # global_itr = epoch * len(train_loader) + i
            # update_lr(optimizer, global_itr)

            # Training procedure:
            # for each sample x:
            #   compute z = f(x)
            #   maximize log p(x) = log p(z) - log |det df/dx|

            x = x.to(device)

            # optimizer.zero_grad()
            optimizerGen.zero_grad()

            # beta = min(1, itr / args.annealing_iters) if args.annealing_iters > 0 else 1.
            # loss, logpz, delta_logp = compute_loss(args, model, beta=beta)

            # genFGen2 = genGen.forward(torch.cuda.FloatTensor(args.batch_size, 2).normal_())
            # genFGen2 = genGen.forward(torch.FloatTensor(args.batch_size, 2).normal_().to(device))

            # (?)
            # genFGen2 = genGen.forward(torch.FloatTensor(args.batch_size, 2).normal_().to(device))
            # (?)

            # (?)
            # genFGen2 = genGen.forward(torch.randn([args.batch_size, 2], device=device))
            # genFGen2 = genGen.forward(torch.randn([args.batch_size, 2], device = device, requires_grad = True))
            # (?)

            # genFGen2 = genGen.forward(torch.randn([args.batch_size, 2], device=device, requires_grad=True))

            # genFGen2 = genGen.forward(torch.randn([args.batch_size, 2], device=device, requires_grad=True))
            # genFGen2 = genGen.forward(torch.randn([args.batch_size, 2], device=device, requires_grad=True))

            # ggenFGen2 = torch.randn(nrand, device=device, requires_grad=True)
            # genFGen2 = genGen.forward(ggenFGen2)

            # genFGen2 = genGen.forward(ggenFGen2)
            # genFGen2 = genGen.forward(ggenFGen2.unsqueeze(0))

            # ggenFGen2 = torch.randn([args.batch_size, 2], device=device, requires_grad=True)
            # ggenFGen2 = torch.randn([args.batchsize, nrand], device=device, requires_grad=True)

            # ggenFGen2 = torch.randn([args.batchsize, nrand], device=device, requires_grad=True)

            # ggenFGen2 = torch.randn([args.batchsize, nrand], device=device, requires_grad=True)
            # ggenFGen2 = torch.randn([100, 128*8], device=device, requires_grad=True)

            # z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            # z_ = Variable(z_.cuda())
            # G_result = G(z_)

            # ggenFGen2 = torch.randn([100, 128 * 8], device=device, requires_grad=True)
            # ggenFGen2 = torch.randn([args.batchsize, 100], device=device, requires_grad=True)
            # ggenFGen2 = torch.randn([args.batchsize, 100], device=device, requires_grad=True).view(-1, 100, 1, 1)

            # ggenFGen2 = torch.randn([args.batchsize, 100], device=device, requires_grad=True)

            # ggenFGen2 = torch.randn([args.batchsize, 100], device=device, requires_grad=True)
            # ggenFGen2 = torch.randn([args.batchsize, nrand], device=device, requires_grad=True)

            # ggenFGen2 = torch.randn([args.batchsize, nrand], device=device, requires_grad=True)
            # ggenFGen2 = torch.randn([args.batchsize, 28*28], device=device, requires_grad=True)

            # ggenFGen2 = torch.randn([args.batchsize, 28 * 28], device=device, requires_grad=True)
            ggenFGen2 = torch.randn([args.batchsize, nrand], device=device, requires_grad=True)

            # genFGen2 = genGen.forward(ggenFGen2)
            genFGen2 = genGen.forward(ggenFGen2)

            # genFGen2.retain_grad()

            # genFGen2.retain_grad()
            genFGen2.retain_grad()

            # genFGen2 = genGen.forward(ggenFGen2)
            # genFGen2 = genGen.forward(ggenFGen2[None, ...])

            # output = net(V(torch.Tensor([x[None, ...]]))
            # genFGen2 = genGen.forward(torch.Tensor(ggenFGen2[None, ...]))

            # (?)
            # with torch.no_grad():
            #    genGen.eval()
            #    genFGen2 = genGen.forward(torch.FloatTensor(args.batch_size, 2).normal_().to(device))
            #    genGen.train()
            # (?)

            # (?)
            # with torch.no_grad():
            #    genGen.eval()
            #    genFGen2 = genGen.forward(torch.randn([args.batch_size, 2], device=device))
            #    genGen.train()
            # (?)

            # genFGen2 = genGen.forward(torch.cuda.FloatTensor(args.batch_size, 2).normal_())
            # lossGen = loss_fn2(genFGen2, args, model)

            # lossGen = loss_fn2(genFGen2, args, model)

            # lossGen = loss_fn2(genFGen2, args, model)
            # lossGen = use_loss_fn2(genFGen2, args, model)

            # lossGen = use_loss_fn2(genFGen2, args, model)
            # lossGen, xData = use_loss_fn2(genFGen2, args, model)

            # lossGen, xData = use_loss_fn2(genFGen2, args, model)
            # lossGen, xData = use_loss_fn2(genFGen2, args, model, mu, S)

            # lossGen, xData = use_loss_fn2(genFGen2, args, model, mu, S)
            # lossGen, xData = use_loss_fn2(genFGen2, args, model, ggenFGen2, toUse_storeAll, toUse_storeAll2)

            # lossGen, xData = use_loss_fn2(genFGen2, args, model, ggenFGen2, toUse_storeAll, toUse_storeAll2)

            # if (itr-1) % 10 == 0:
            #    lossGen, xData = use_loss_fn2(genFGen2, args, model, ggenFGen2, toUse_storeAll, toUse_storeAll2)
            # else:
            #    lossGen = use_loss_fn3(genFGen2, args, model, ggenFGen2, toUse_storeAll, toUse_storeAll2, xData)

            """
            print('')
            print(genFGen2.shape)
            #print(ggenFGen2.shape)

            #print(ggenFGen2.shape)
            print('')
            """

            # lossGen, xData = use_loss_fn2(genFGen2, args, model, ggenFGen2, toUse_storeAll, toUse_storeAll2)

            # lossGen, xData = use_loss_fn2(genFGen2, args, model, ggenFGen2, toUse_storeAll, toUse_storeAll2)
            # lossGen, xData = use_loss_fn2(genFGen2, args, model, ggenFGen2)

            # print(x.shape)
            # print(genFGen2.shape)

            # print(genFGen2.shape)
            # asdfasdf

            # genFGen2 = genFGen2.view(-1, 1, 28, 28)

            # genFGen2 = genFGen2.view(-1, 1, 28, 28)
            # genFGen2 = genFGen2.view(-1, 1, 28, 28)

            '''
            #genFGen2 = genFGen2.view(-1, 1, 28, 28)

            #genFGen2 = genFGen2.view(-1, 1, 28, 28)
            genFGen2 = genFGen2.view(-1, 1, 28, 28)
            '''

            # print(genFGen2)
            # adfasdfs

            # print(x.shape)
            # print(genFGen2.shape)

            # print(xData.shape)
            # print(genFGen2.shape)

            # import matplotlib
            # matplotlib.use('Agg')
            # import matplotlib.pyplot as plt

            # matplotlib.use("TkAgg")
            # import matplotlib.image as mpimg

            # import cv2
            # print(matplotlib.get_backend())

            # matplotlib.rcParams["backend"] = "TkAgg"
            # plt.switch_backend("TkAgg")

            """
            import tkinter
            import matplotlib
            matplotlib.use('TKAgg')
            import matplotlib.pyplot as plt

            import matplotlib.image as mpimg

            #imageStore = xData[0,:,:,:].squeeze().cpu().numpy()
            imageStore = genFGen2[0, :, :, :].squeeze().cpu().detach().numpy()

            #print(np.shape(imageStore))

            plt.figure()
            plt.imshow(imageStore)

            plt.savefig('baseBase21')

            plt.ion()
            plt.show()
            plt.pause(9)

            #cv2.imshow("Image", imageStore)
            #cv2.imshow("Gray", gray)
            #cv2.waitKey(0)
            """

            # pilTrans = transforms.ToTensor()
            # plt.imshow(xData[1, :])

            # lossGen = use_loss_fn2(genFGen2, args, model, ggenFGen2, x)

            # lossGen = use_loss_fn2(genFGen2, args, model, ggenFGen2, x)
            # lossGen, firstOnly_lossGen, secondOnly_lossGen = use_loss_fn2(genFGen2, args, model, ggenFGen2, x)

            # lossGen, firstOnly_lossGen, secondOnly_lossGen = use_loss_fn2(genFGen2, args, model, ggenFGen2, x)
            # lossGen, _, _ = use_loss_fn2(genFGen2, args, model, ggenFGen2, x)

            # lossGen, firstOnly_lossGen, secondOnly_lossGen = use_loss_fn2(genFGen2, args, model, ggenFGen2, x)
            # lossGen, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(genFGen2, args, model, ggenFGen2, x)

            lossGen, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(genFGen2, args, model,
                                                                                             ggenFGen2, x)

            # lossGen = use_loss_fn2(genFGen2, args, model, ggenFGen2, x)
            # lossGen = use_loss_fn2(genFGen2, args, model, ggenFGen2, x)

            # lossGen = use_loss_fn2(genFGen2, args, model, ggenFGen2, x)

            # lossGen = loss_fn2(genFGen2, args, model)
            # lossGen = loss_fn2(genFGen2, args, model)

            # xData = toy_data.inf_train_gen(args.data, batch_size=args.batch_size)
            # xData = torch.from_numpy(xData).type(torch.float32).to(device)

            '''
            print('')
            print(lossGen.item())

            print(firstOnly_lossGen.item())
            print(secondOnly_lossGen.item())
            print(thirdOnly_lossGen.item())

            print('')
            print(lossGen.grad)

            print(firstOnly_lossGen.grad)
            print(secondOnly_lossGen.grad)
            print(thirdOnly_lossGen.grad)

            print('')
            '''

            # loss_meter.update(loss.item())
            # logpz_meter.update(logpz.item())

            loss2_meter.update(lossGen.item())

            filoss2_meter.update(firstOnly_lossGen.item())
            seloss2_meter.update(secondOnly_lossGen.item())
            thloss2_meter.update(thirdOnly_lossGen.item())

            arfiloss2_meter.append(firstOnly_lossGen.item())
            arseloss2_meter.append(secondOnly_lossGen.item())
            arthloss2_meter.append(thirdOnly_lossGen.item())

            # filoss2_meter = utils.RunningAverageMeter(0.93)
            # seloss2_meter = utils.RunningAverageMeter(0.93)
            # thloss2_meter = utils.RunningAverageMeter(0.93)

            # gr1loss2_meter = utils.RunningAverageMeter(0.93)
            # gr2loss2_meter = utils.RunningAverageMeter(0.93)

            # delta_logp_meter.update(delta_logp.item())

            # loss.backward()
            # lossGen.backward()

            # print(lossGen.grad)
            # print(firstOnly_lossGen.grad)

            # print(genFGen2.grad)

            lossGen.backward()

            # lossGen.backward(create_graph=True)
            # lossGen.backward()

            # print(lossGen.grad)
            # print(firstOnly_lossGen.grad)

            # print(firstOnly_lossGen.grad.item())
            # print(secondOnly_lossGen.grad.item())

            # print(thirdOnly_lossGen.grad.item())
            # print('')

            # print(lossGen.grad.item())
            # asdfasdf

            """
            print('')
            print(lossGen.item())

            print(firstOnly_lossGen.item())
            print(secondOnly_lossGen.item())
            print(thirdOnly_lossGen.item())

            print('')
            print(lossGen.grad)

            print(firstOnly_lossGen.grad)
            print(secondOnly_lossGen.grad)
            print(thirdOnly_lossGen.grad)

            print('')
            """

            # print((torch.mean(genGen.lin1.weight.grad) + torch.mean(genGen.lin2.weight.grad) + torch.mean(
            #    genGen.dc1.weight.grad) + torch.mean(genGen.dc2.weight.grad) + torch.mean(genGen.dc3.weight.grad)).item())

            # print((torch.mean(genGen.lin1.bias.grad) + torch.mean(genGen.lin2.bias.grad) + torch.mean(
            #    genGen.dc1.bias.grad) + torch.mean(genGen.dc2.bias.grad) + torch.mean(genGen.dc3.bias.grad)).item())

            # print('')

            # gr1loss2_meter = utils.RunningAverageMeter(0.93)
            # gr2loss2_meter = utils.RunningAverageMeter(0.93)

            # gr1loss2_meter.update((torch.mean(genGen.lin1.weight.grad) + torch.mean(genGen.lin2.weight.grad) + torch.mean(
            #    genGen.dc1.weight.grad) + torch.mean(genGen.dc2.weight.grad) + torch.mean(
            #    genGen.dc3.weight.grad) + torch.mean(genGen.lin1.bias.grad) + torch.mean(genGen.lin2.bias.grad) + torch.mean(
            #    genGen.dc1.bias.grad) + torch.mean(genGen.dc2.bias.grad) + torch.mean(genGen.dc3.bias.grad)).item())

            '''
            gr1loss2_meter.update(
                (torch.mean(genGen.lin1.weight.grad) + torch.mean(genGen.lin2.weight.grad) + torch.mean(
                    genGen.dc1.weight.grad) + torch.mean(genGen.dc2.weight.grad) + torch.mean(
                    genGen.dc3.weight.grad) + torch.mean(genGen.lin1.bias.grad) + torch.mean(
                    genGen.lin2.bias.grad) + torch.mean(
                    genGen.dc1.bias.grad) + torch.mean(genGen.dc2.bias.grad) + torch.mean(genGen.dc3.bias.grad)).item())
            '''

            """
            gr1loss2_meter.update(
                (torch.mean(genGen.fc1.weight.grad) + torch.mean(genGen.fc2.weight.grad) + torch.mean(
                    genGen.fc3.weight.grad) + torch.mean(genGen.fc4.weight.grad) + torch.mean(
                    genGen.fc1.bias.grad) + torch.mean(
                    genGen.fc2.bias.grad) + torch.mean(
                    genGen.fc3.bias.grad) + torch.mean(genGen.fc4.bias.grad)).item())
            """

            gr1loss2_meter.update(
                (torch.mean(genGen.lin1.weight.grad) + torch.mean(genGen.lin2.weight.grad) + torch.mean(
                    genGen.dc1.weight.grad) + torch.mean(genGen.dc2.weight.grad) + torch.mean(
                    genGen.dc3.weight.grad) + torch.mean(genGen.lin1.bias.grad) + torch.mean(
                    genGen.lin2.bias.grad) + torch.mean(
                    genGen.dc1.bias.grad) + torch.mean(genGen.dc2.bias.grad) + torch.mean(genGen.dc3.bias.grad)).item())

            gr2loss2_meter.update(torch.mean(ggenFGen2.grad).item())
            # gr3loss2_meter.update(torch.mean(genFGen2.grad).item())

            # gr3loss2_meter.update(torch.mean(genFGen2.grad).item())

            # gr3loss2_meter.update(torch.mean(genFGen2.grad).item())
            # gr3loss2_meter.update(torch.mean(genFGen2.grad).item())

            if genFGen2.grad is not None:
                gr3loss2_meter.update(torch.mean(genFGen2.grad).item())
            else:
                gr3loss2_meter.update(0.0)

            argr1loss2_meter.append(gr1loss2_meter.val)
            argr2loss2_meter.append(gr2loss2_meter.val)
            argr3loss2_meter.append(gr3loss2_meter.val)

            # print((torch.mean(genGen.lin1.weight.grad) + torch.mean(genGen.lin2.weight.grad) + torch.mean(
            #    genGen.dc1.weight.grad) + torch.mean(genGen.dc2.weight.grad) + torch.mean(
            #    genGen.dc3.weight.grad) + torch.mean(genGen.lin1.bias.grad) + torch.mean(genGen.lin2.bias.grad) + torch.mean(
            #    genGen.dc1.bias.grad) + torch.mean(genGen.dc2.bias.grad) + torch.mean(genGen.dc3.bias.grad)).item())

            print(gr1loss2_meter.val)

            # print((torch.mean(genGen.lin1.weight.grad) + torch.mean(genGen.lin2.weight.grad) + torch.mean(
            #                 genGen.dc1.weight.grad) + torch.mean(genGen.dc2.weight.grad) + torch.mean(
            #                 genGen.dc3.weight.grad) + torch.mean(genGen.lin1.bias.grad) + torch.mean(
            #                 genGen.lin2.bias.grad) + torch.mean(
            #                 genGen.dc1.bias.grad) + torch.mean(genGen.dc2.bias.grad) + torch.mean(genGen.dc3.bias.grad)).item())

            # print((torch.mean(genGen.lin1.weight.data) + torch.mean(genGen.lin2.weight.data) + torch.mean(
            #    genGen.dc1.weight.data) + torch.mean(genGen.dc2.weight.data) + torch.mean(
            #    genGen.dc3.weight.data) + torch.mean(genGen.lin1.bias.data) + torch.mean(
            #    genGen.lin2.bias.data) + torch.mean(
            #    genGen.dc1.bias.data) + torch.mean(genGen.dc2.bias.data) + torch.mean(genGen.dc3.bias.data)).item())

            # print((torch.mean(genGen.lin1.weight) + torch.mean(genGen.lin2.weight) + torch.mean(
            #    genGen.dc1.weight) + torch.mean(genGen.dc2.weight) + torch.mean(
            #    genGen.dc3.weight) + torch.mean(genGen.lin1.bias) + torch.mean(
            #    genGen.lin2.bias) + torch.mean(
            #    genGen.dc1.bias) + torch.mean(genGen.dc2.bias) + torch.mean(genGen.dc3.bias)).item())

            print(torch.mean(ggenFGen2.grad).item())
            # print(torch.mean(genFGen2.grad).item())

            # print(torch.mean(genFGen2.grad).item())

            # print(torch.mean(genFGen2.grad).item())
            # print(torch.mean(genFGen2.grad).item())

            if genFGen2.grad is not None:
                print(torch.mean(genFGen2.grad).item())
            else:
                print(0.0)

            # print(x.grad)
            # print(genFGen2.grad)

            # print(genFGen2.grad)
            # print(genGen.parameters().tolist())

            print('')

            # genFGen2.retain_grad()
            # print(genFGen2.grad)

            # torch.cuda.synchronize()

            # torch.cuda.synchronize()
            # torch.cuda.synchronize()

            # if args.learn_p and itr > args.annealing_iters:
            #    compute_p_grads(model)

            # optimizer.step()
            optimizerGen.step()

            # running_loss += loss.item() * images.size(0)

            # running_loss += loss.item() * images.size(0)
            runningLoss_NIKrunningLoss += lossGen.item() * x.size(0)

            # update_lipschitz(model, args.n_lipschitz_iters)
            time2_meter.update(time.time() - end)

            # logger.info('Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f})'.format(
            #        itr, time2_meter.val, time2_meter.avg, loss2_meter.val, loss2_meter.avg))

            # logger.info(
            #    'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f})'.format(itr, time2_meter.val, time2_meter.avg,
            #                                                                     loss2_meter.val, loss2_meter.avg))

            # logger.info(
            #    'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f})'.format(itr, time2_meter.val, time2_meter.avg,
            #                                                                     loss2_meter.val, loss2_meter.avg))

            """
            logger.info(
                'Itr {:04d} | Tm {:.4f}({:.4f}) | L {:.6f}({:.6f}) | L0 {:.4f}({:.4f}) | L1 {:.6f}({:.6f})  | L2 {:.6f}({:.6f}) | G1 {:.6f}({:.6f})  | G2 {:.6f}({:.6f})'.format(
                    itr, time2_meter.val, time2_meter.avg,
                    loss2_meter.val, loss2_meter.avg, filoss2_meter.val, filoss2_meter.avg, seloss2_meter.val,
                    seloss2_meter.avg, thloss2_meter.val, thloss2_meter.avg, gr1loss2_meter.val, gr1loss2_meter.avg,
                    gr2loss2_meter.val, gr2loss2_meter.avg))
            """

            logger.info(
                'Itr {:}, {:} | Tm {:.2f}({:.2f}) | L {:.6f}({:.6f}) | L0 {:.6f}({:.6f}) | L1 {:.6f}({:.6f})  | L2 {:.6f}({:.6f}) | G1 {:.6f}({:.6f})  | G2 {:.6f}({:.6f})  | G3 {:.6f}({:.6f})'.format(
                    itr, i, time2_meter.val, time2_meter.avg,
                    loss2_meter.val, loss2_meter.avg, filoss2_meter.val, filoss2_meter.avg, seloss2_meter.val,
                    seloss2_meter.avg, thloss2_meter.val, thloss2_meter.avg, gr1loss2_meter.val, gr1loss2_meter.avg,
                    gr2loss2_meter.val, gr2loss2_meter.avg, gr3loss2_meter.val, gr3loss2_meter.avg))

            # if (itr-1) % 250 == 0:
            #    logger.info('Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f})'.format(itr, time2_meter.val, time2_meter.avg, loss2_meter.val, loss2_meter.avg))

            if lossGen < best_loss:
                best_loss = lossGen

                # utils.save_checkpoint({'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(), 'args': args},
                #                      os.path.join(args.save, 'mainMainMyBest'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, 'mainMainMainMyMyBest'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, 'maainMaainMaainMyMyMyBest'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, 'maaiinMaaiinMaaiinMyMyMyMyBest'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, 'maaiinMaaiinMaaiinMyMyMyMyMyBeBest'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, 'maaiiinMaaiiinMyMyMyMyMyBeBest'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, 'maaiiinMaaiiinMyMyBest'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, 'maaiiinMaaiiinMyMyMyBest9'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, 'maaiiinMaaiiinMaaiiinMyBest99'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, 'maaiiinNikNikMaaiiinMaaiiinBest'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, 'maaiiinNikNik99MaaiiinMaaiiinBest'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, 'maaiiinNikNik99MaaiiinMaaiiinBeBe'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, '0mainNikNik99MaaiiinMaaiiinBeBe'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, '00mainNikNik99MaaiiinMaaiiinBeBe'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, '0000mmainNikNik99MaaiiinMaaiiinBeBe'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, '00000mmmainNikNik99MaaiiinMaaiiinBeBe'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, '000000mmmmainNikNik99MaaiiinMaaiiinBeBe'), args.nepochs)

                utils.save_checkpoint(
                    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                     'args': args},
                    os.path.join(args.save,
                                 '000000000000000000000mmmmmmmmmmmmmmmmmmaaaaaiinNikNik99MaaiiinMaaiiinBeBe'),
                    args.nepochs)

            # scheduler.step(lossGen)

            # scheduler.step(lossGen)
            # scheduler.step(lossGen)

            if (itr - 1) % 1000 == 0:
                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, 'maaiiinNikNik99MaaiiinMaaiiinBe2Be2'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, '0mainNikNik99MaaiiinMaaiiinBe2Be2'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, '00mainNikNik99MaaiiinMaaiiinBe2Be2'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, '000mmainNikNik99MaaiiinMaaiiinBe2Be2'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, '0000mmmainNikNik99MaaiiinMaaiiinBe2Be2'), args.nepochs)

                # utils.save_checkpoint(
                #    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                #     'args': args},
                #    os.path.join(args.save, '00000mmmmainNikNik99MaaiiinMaaiiinBe2Be2'), args.nepochs)

                utils.save_checkpoint(
                    {'state_dict': genGen.state_dict(), 'optimizer_state_dict': optimizerGen.state_dict(),
                     'args': args},
                    os.path.join(args.save,
                                 '00000000000000000000mmmmmmmmmmmmmmmmmmaaaaaiinNikNik99MaaiiinMaaiiinBe2Be2'),
                    args.nepochs)

            # scheduler.step()

            # scheduler.step()
            # scheduler.step()

            """
            logger.info('Current LR {}'.format(optimizer.param_groups[0]['lr']))

            train(epoch, model)
            lipschitz_constants.append(get_lipschitz_constants(model))

            ords.append(get_ords(model))
            logger.info('Lipsh: {}'.format(pretty_repr(lipschitz_constants[-1])))
            logger.info('Order: {}'.format(pretty_repr(ords[-1])))

            if args.ema_val:
                test_bpd = validate(epoch, model, ema)
            else:
                test_bpd = validate(epoch, model)

            if args.scheduler and scheduler is not None:
                scheduler.step()

            if test_bpd < best_test_bpd:
                best_test_bpd = test_bpd
                utils.save_checkpoint({
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': args,
                    'ema': ema,
                    'test_bpd': test_bpd,
                }, os.path.join(args.save, 'models'), epoch, last_checkpoints, num_checkpoints=5)
            """

            # torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            # }, os.path.join(args.save, 'models', 'theTheTheTheTheTheTheTheTheMostRecent.pth'))

        # losses_NIKlosses.append(runningLoss_NIKrunningLoss / len())
        losses_NIKlosses.append(runningLoss_NIKrunningLoss / len(datasets.MNIST(
            args.dataroot, train=True, transform=transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                add_noise,
            ])
        )))

        import matplotlib.pyplot as plt
        # plt.figure()

        plt.plot(losses_NIKlosses)
        # plt.savefig('gggoooGGoo.png', bbox_inches='tight')

        # plt.savefig('gggoooGGoo.png', bbox_inches='tight')
        # plt.savefig('00ggggoooGGoo.png', bbox_inches='tight')

        # plt.savefig('000gggggoooGGoo.png', bbox_inches='tight')
        # plt.savefig('0000ggggggoooGGoo.png', bbox_inches='tight')

        # plt.savefig('00000gggggggoooGGoo.png', bbox_inches='tight')
        # plt.savefig('000000ggggggggoooGGoo.png', bbox_inches='tight')

        # plt.savefig('0000000gggggggggoooGGoo.png', bbox_inches='tight')
        # plt.savefig('00000000ggggggggggoooGGoo.png', bbox_inches='tight')

        # plt.savefig('000000000gggggggggggoooGGoo.png', bbox_inches='tight')
        # plt.savefig('0000000000ggggggggggggoooGGoo.png', bbox_inches='tight')

        # plt.savefig('00000000000gggggggggggggoooGGoo.png', bbox_inches='tight')
        # plt.savefig('000000000000ggggggggggggggoooGGoo.png', bbox_inches='tight')

        # plt.savefig('0000000000000gggggggggggggggoooGGoo.png', bbox_inches='tight')
        # plt.savefig('00000000000000ggggggggggggggggoooGGoo.png', bbox_inches='tight')
        plt.savefig('000000000000000ggggggggggggggggoooGGoo.png', bbox_inches='tight')

    # plt.figure()
    plt.plot(losses_NIKlosses)

    # plt.savefig('ffoooFoo.png', bbox_inches='tight')
    # plt.savefig('00fffoooFoo.png', bbox_inches='tight')

    # plt.savefig('000ffffoooFoo.png', bbox_inches='tight')
    # plt.savefig('0000fffffoooFoo.png', bbox_inches='tight')

    # plt.savefig('00000ffffffoooFoo.png', bbox_inches='tight')
    # plt.savefig('000000fffffffoooFoo.png', bbox_inches='tight')

    # plt.savefig('0000000ffffffffoooFoo.png', bbox_inches='tight')
    # plt.savefig('00000000fffffffffoooFoo.png', bbox_inches='tight')

    # plt.savefig('000000000ffffffffffoooFoo.png', bbox_inches='tight')
    # plt.savefig('0000000000fffffffffffoooFoo.png', bbox_inches='tight')

    # plt.savefig('00000000000ffffffffffffoooFoo.png', bbox_inches='tight')
    # plt.savefig('000000000000fffffffffffffoooFoo.png', bbox_inches='tight')

    # plt.savefig('0000000000000ffffffffffffffoooFoo.png', bbox_inches='tight')
    # plt.savefig('00000000000000fffffffffffffffoooFoo.png', bbox_inches='tight')
    plt.savefig('000000000000000fffffffffffffffoooFoo.png', bbox_inches='tight')


if __name__ == '__main__':
    main()

