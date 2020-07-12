import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

#import os
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
    '--data', type=str, default='mnist', choices=[
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
#parser.add_argument('--imagesize', type=int, default=32)

#parser.add_argument('--imagesize', type=int, default=32)
parser.add_argument('--imagesize', type=int, default=28)

parser.add_argument('--nbits', type=int, default=8)  # Only used for celebahq
parser.add_argument('--block', type=str, choices=['resblock', 'coupling'], default='resblock')

parser.add_argument('--coeff', type=float, default=0.98)
parser.add_argument('--vnorms', type=str, default='2222')

#parser.add_argument('--n-lipschitz-iters', type=int, default=None)
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

parser.add_argument('--nepochs', help='Number of epochs for training', type=int, default=1000)
#parser.add_argument('--batchsize', help='Minibatch size', type=int, default=64)

#parser.add_argument('--batchsize', help='Minibatch size', type=int, default=64)
#parser.add_argument('--batchsize', help='Minibatch size', type=int, default=32)

#parser.add_argument('--batchsize', help='Minibatch size', type=int, default=32)
parser.add_argument('--batchsize', help='Minibatch size', type=int, default=32)

#parser.add_argument('--batchsize', help='Minibatch size', type=int, default=64)
#parser.add_argument('--batchsize', help='Minibatch size', type=int, default=64)

parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--wd', help='Weight decay', type=float, default=0)

parser.add_argument('--warmup-iters', type=int, default=1000)
parser.add_argument('--annealing-iters', type=int, default=0)

parser.add_argument('--save', help='directory to save results', type=str, default='experiment1')
#parser.add_argument('--val-batchsize', help='minibatch size', type=int, default=200)

#parser.add_argument('--val-batchsize', help='minibatch size', type=int, default=200)
parser.add_argument('--val-batchsize', help='minibatch size', type=int, default=32)

#parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--seed', type=int, default=None)

parser.add_argument('--ema-val', type=eval, choices=[True, False], default=True)
parser.add_argument('--update-freq', type=int, default=1)

parser.add_argument('--task', type=str, choices=['density', 'classification', 'hybrid'], default='density')
parser.add_argument('--scale-dim', type=eval, choices=[True, False], default=False)

parser.add_argument('--rcrop-pad-mode', type=str, choices=['constant', 'reflect'], default='reflect')
parser.add_argument('--padding-dist', type=str, choices=['uniform', 'gaussian'], default='uniform')

#parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--begin-epoch', type=int, default=0)

#parser.add_argument('--resume', type=str, default=None)

#parser.add_argument('--resume', type=str, default=None)
#parser.add_argument('--resume', type=str, default='./mnist_resflow_16-16-16.pth')

#parser.add_argument('--resume', type=str, default=None)
#parser.add_argument('--resume', type=str, default='./experiment1/models/most_recent.pth')

#parser.add_argument('--resume', type=str, default=None)
#parser.add_argument('--resume', type=str, default=None)

#torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mostMostMostRecent.pth'))

#parser.add_argument('--resume', type=str, default='./experiment1/models/mostMostMostRecent.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmoostMostMostRecent.pth')

# mmoostMostMostRecent
# use: mmoostMostMostRecent

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmoostMostMostRecent.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmoostMostMostRecent.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmoostMostMostRecent.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmooostMostMostRecent.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmooostMostMostRecent.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmoooostMostMostRecent.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmoooostMostMostRecent.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmooooostMostMostRecent.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmooooostMostMostRecent.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent2.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent2.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent4.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent4.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent4.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent4.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent4.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent4.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent6.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/moosstttMoosstttMoosstttMoosstttRecent4.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent6.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent7.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent7.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent8.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent8.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent9.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent9.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent99.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent99.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent99.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent99.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent9999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent9999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent99999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent99999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent99999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent99999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent9999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent9999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent9999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmmoooooostMostMostRecent9999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmooooostMostMostRecent999999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmooooostMostMostRecent999999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmooooostMostMostRecent999999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmmooooostMostMostRecent999999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmoooostMostMostRecent9999999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmmoooostMostMostRecent9999999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmooostMostMostRecent99999999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmooostMostMostRecent99999999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmooostMostMostRecent99999999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmmooostMostMostRecent99999999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mmoostMostMostRecent999999999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mmoostMostMostRecent999999999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mostMostMostRecent9999999999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mostMostMostRecent9999999999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mostMostMostRecent9999999999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mostMostMostRecent9999999999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mostMostMostReRecent99999999999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mostMostMostReRecent99999999999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mostMostMostReReRecent999999999999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mostMostMostReReRecent999999999999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mostMostMostRecRecRecent9999999999999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mostMostMostRecRecRecent9999999999999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/mostMostMostRecRecRecRecent99999999999999999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/mostMostMostRecRecRecRecent99999999999999999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/moRec99.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/moRec99.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/moRec99.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/moRec99.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/moRec9999.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/moRec9999.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/moRec99rec99.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/moRec99rec99.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/moRe991re991re991.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/moRe991re991re991.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/moRe977re977re977re977.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/moRe991re991re991.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/moMoRe9777re9777re9777re9777.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/moMoRe9777re9777re9777re9777.pth')
#parser.add_argument('--resume', type=str, default='./experiment1/models/moMoMoRee9777ree9777ree9777ree9777.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/moMoMoRee9777ree9777ree9777ree9777.pth')

#parser.add_argument('--resume', type=str, default='./experiment1/models/moMoMoRee9777ree9777ree9777ree9777.pth')
parser.add_argument('--resume', type=str, default='./experiment1/models/moMoMoMoRRee9777rree9777rree9777rree9777.pth')

#torch.save({
#            'state_dict': model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'args': args,
#            'ema': ema,
#            'test_bpd': test_bpd,
#        }, os.path.join(args.save, 'models', 'mmmmmooooostMostMostRecent.pth'))

#parser.add_argument('--nworkers', type=int, default=4)

#parser.add_argument('--nworkers', type=int, default=4)
#parser.add_argument('--nworkers', type=int, default=2)

#parser.add_argument('--nworkers', type=int, default=2)
parser.add_argument('--nworkers', type=int, default=1)

#parser.add_argument('--nworkers', type=int, default=4)
#parser.add_argument('--nworkers', type=int, default=4)

parser.add_argument('--print-freq', help='Print progress every so iterations', type=int, default=100)
parser.add_argument('--vis-freq', help='Visualize progress every so iterations', type=int, default=500)

args = parser.parse_args()

# Random seed

# Random seed
if args.seed is None:
    args.seed = np.random.randint(100000)

utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

# logger
logger.info(args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024**3)))
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
        x = torch.floor(x / 2**(8 - args.nbits))
        x = x / 2**args.nbits
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
if args.data == 'cifar10':
    im_dim = 3
    n_classes = 10
    if args.task in ['classification', 'hybrid']:

        # Classification-specific preprocessing.
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
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.dataroot, train=False, transform=transform_test),
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.nworkers,
    )
elif args.data == 'mnist':
    im_dim = 1
    init_layer = layers.LogitTransform(1e-6)
    n_classes = 10
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
                lambda x: add_noise(x, nvals=2**args.nbits),
            ])
        ), batch_size=args.batchsize, shuffle=True, num_workers=args.nworkers
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CelebAHQ(
            train=False, transform=transforms.Compose([
                reduce_bits,
                lambda x: add_noise(x, nvals=2**args.nbits),
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

#print(dataset_size)
#print(len(test_loader.dataset))

#print(len(train_loader.dataset.mnist.data))
#print(len(train_loader.dataset.mnist.train_data))
#print(len(train_loader.dataset.mnist.train_labels))

#print(len(train_loader.dataset.mnist.test_data))
#print(len(train_loader.dataset.mnist.test_labels))

#print(len(test_loader.dataset.mnist.test_data))
#print(len(test_loader.dataset.mnist.test_labels))

if args.squeeze_first:
    input_size = (input_size[0], input_size[1] * 4, input_size[2] // 2, input_size[3] // 2)
    squeeze_layer = layers.SqueezeLayer(2)

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


#logger.info(model)
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

    #args = checkpt['args']
    #logger.info(args)

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


def compute_loss(x, model, model_copy, beta=1.0):
    bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(n_classes).to(x)
    logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)

    if args.data == 'celeba_5bit':
        nvals = 32
    elif args.data == 'celebahq':
        nvals = 2**args.nbits
    else:
        nvals = 256

    x, logpu = add_padding(x, nvals)

    if args.squeeze_first:
        x = squeeze_layer(x)

    if args.task == 'hybrid':
        z_logp, logits_tensor = model(x.view(-1, *input_size[1:]), 0, classify=True)
        z, delta_logp = z_logp
    elif args.task == 'density':
        #z, delta_logp = model(x.view(-1, *input_size[1:]), 0)

        #z, delta_logp = model(x.view(-1, *input_size[1:]), 0)
        #z, delta_logp = model(x.view(-1, *input_size[1:]), 0)

        #model = model.to(device)

        z, delta_logp = model(x.view(-1, *input_size[1:]), 0)

        #model = model.to(device)

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


def compute_loss_original(x, model, beta=1.0):
    bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(n_classes).to(x)
    logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)

    if args.data == 'celeba_5bit':
        nvals = 32
    elif args.data == 'celebahq':
        nvals = 2**args.nbits
    else:
        nvals = 256

    x, logpu = add_padding(x, nvals)

    if args.squeeze_first:
        x = squeeze_layer(x)

    if args.task == 'hybrid':
        z_logp, logits_tensor = model(x.view(-1, *input_size[1:]), 0, classify=True)
        z, delta_logp = z_logp
    elif args.task == 'density':
        #z, delta_logp = model(x.view(-1, *input_size[1:]), 0)

        #z, delta_logp = model(x.view(-1, *input_size[1:]), 0)
        #z, delta_logp = model(x.view(-1, *input_size[1:]), 0)

        #model = model.to(device)

        #print(x)
        #print(x.shape)
        #adfas

        z, delta_logp = model(x.view(-1, *input_size[1:]), 0)

        #model = model.to(device)

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


def compute_loss2(x, model, beta=1.0):
    #bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(n_classes).to(x)

    bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(n_classes).to(x)
    logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)

    if args.data == 'celeba_5bit':
        nvals = 32
    elif args.data == 'celebahq':
        nvals = 2**args.nbits
    else:
        nvals = 256

    #x, logpu = add_padding(x, nvals)

    #x, logpu = add_padding(x, nvals)
    #x, logpu = add_padding(x, nvals)

    #x, logpu = add_padding(x, nvals)
    #_, logpu = add_padding(x, nvals)

    x, logpu = add_padding(x, nvals)

    #x, logpu = add_padding(x, nvals)
    #_, logpu = add_padding(x, nvals)

    #print(args.squeeze_first)
    #dasfasdf

    if args.squeeze_first:
        x = squeeze_layer(x)

    #print(args.task)

    #print(x.shape)
    #print((x.view(-1, *input_size[1:])).shape)

    if args.task == 'hybrid':
        z_logp, logits_tensor = model(x.view(-1, *input_size[1:]), 0, classify=True)
        z, delta_logp = z_logp

    elif args.task == 'density':
        #z, delta_logp = model(x.view(-1, *input_size[1:]), 0)

        #z, delta_logp = model(x.view(-1, *input_size[1:]), 0)
        #z, delta_logp = model(x.view(-1, *input_size[1:]), 0)

        #z, delta_logp = model(x.view(-1, *input_size[1:]), 0)

        #z, delta_logp = model(x.view(-1, *input_size[1:]), 0)
        #z, delta_logp = model(x.view(-1, *input_size[1:]), 0.0)

        #prev_delta_logp = delta_logp

        #prev_delta_logp = delta_logp
        #z, delta_logp = model(x.view(-1, *input_size[1:]), 0.0)

        #z, delta_logp = model(x.view(-1, *input_size[1:]), 0.0)
        #z, delta_logp = model(x.view(-1, *input_size[1:]), -1e-7)

        #print(x)
        #adfas

        #print(x.shape)
        #asdf

        #x.requires_grad = True

        #z, delta_logp = model(x.view(-1, *input_size[1:]), 0.0)
        z, delta_logp = model(x.view(-1, *input_size[1:]), 0.0)

        #if torch.isnan(torch.mean(z)):
        #   z = x.view(-1, x.squeeze().shape[1]**2)

        """
        if torch.isnan(torch.mean(delta_logp)):
            delta_logp = prev_delta_logp

        if torch.isnan(torch.mean(z)):
          z = x.view(-1, x.squeeze().shape[1]**2)
        """

        #asfdgsd

        #print(x.shape)
        #print(z.shape)
        #print(delta_logp.shape)

        #print('')
        #print(x)
        #print(z)
        #print(delta_logp)

        #z[torch.isnan(z)] = 0.0

        #print(x.shape)
        #print(x)

        """
        print(x.shape)
        print(torch.mean(x))

        #print(z.shape)
        #print(z)

        print(z.shape)
        print(torch.mean(z))
        """

        #t = torch.ones((2, 3, 4))
        #t.size()
        #torch.Size([2, 3, 4])
        #t.view(-1, 12).size()
        #torch.Size([2, 12])

        #print(z.shape)
        #print(z)

        #if torch.isnan(torch.mean(z)):
        #    z = x.view(-1, x.squeeze().shape[1]**2)

        #print(z.shape)
        #print(z)

        #print(z)
        #print(delta_logp)

        #asdfkdfs

        #print(z)
        #asdfa

        #print('')
        #print(z)

        #z[torch.isnan(z)] = 0.0
        #delta_logp[torch.isnan(delta_logp)] = 0.0

        #z[z != z] = 0.0
        #delta_logp[delta_logp != delta_logp] = 0.0

    elif args.task == 'classification':
        z, logits_tensor = model(x.view(-1, *input_size[1:]), classify=True)

    if args.task in ['density', 'hybrid']:
        # log p(z)

        # log p(z)
        logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)

        # log p(x)
        logpx = logpz - beta * delta_logp - np.log(nvals) * (
            args.imagesize * args.imagesize * (im_dim + args.padding)) - logpu

        #bits_per_dim = -torch.mean(logpx) / (args.imagesize * args.imagesize * im_dim) / np.log(2)

        #bits_per_dim = -torch.mean(logpx) / (args.imagesize * args.imagesize * im_dim) / np.log(2)
        #bits_per_dim = torch.mean(torch.exp(logpx)) / (args.imagesize * args.imagesize * im_dim) / np.log(2)

        #bits_per_dim = torch.mean(torch.exp(logpx)) / (args.imagesize * args.imagesize * im_dim) / np.log(2)

        #print(torch.mean(logpx))
        #print(torch.mean(torch.exp(logpx)))

        #bits_per_dim = torch.mean(torch.exp(logpx)) / (args.imagesize * args.imagesize * im_dim) / np.log(2)
        bits_per_dim = torch.mean(torch.exp(logpx)) / (args.imagesize * args.imagesize * im_dim) / np.log(2)

        #bits_per_dim = torch.mean(torch.exp(logpx)) / (args.imagesize * args.imagesize * im_dim) / np.log(2)
        #bits_per_dim = torch.mean(torch.exp(logpx))

        #logpz = torch.mean(logpz).detach()
        #logpz = torch.mean(torch.exp(logpx)).detach()

        #logpz = torch.mean(torch.exp(logpx)).detach()

        #logpz = torch.mean(logpz).detach()
        #delta_logp = torch.mean(-delta_logp).detach()

        #delta_logp = torch.mean(-delta_logp).detach()

    #return bits_per_dim, logits_tensor, logpz, delta_logp
    #return logpz

    #return logpz
    return bits_per_dim


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

    #first_term_loss = compute_loss2(genFGen2, args, model)
    #first_term_loss2 = compute_loss2(genFGen2, args, model)
    #first_term_loss = torch.log(first_term_loss2 / (1.0 - first_term_loss2))

    #first_term_loss = compute_loss2(genFGen2, args, model)

    #first_term_loss = compute_loss2(genFGen2, args, model)
    #first_term_loss = compute_loss2(genFGen2, args, model)

    #print('')
    #print(first_term_loss)

    #mu = torch.from_numpy(np.array([2.805741, -0.00889241], dtype="float32")).to(device)
    #S = torch.from_numpy(np.array([[pow(0.3442525,2), 0.0], [0.0, pow(0.35358343,2)]], dtype="float32")).to(device)

    #mu = torch.from_numpy(np.array([2.8093171, 1.2994107e-03], dtype="float32")).to(device)
    #S = torch.from_numpy(np.array([[pow(0.35840544, 2), 0.0], [0.0, pow(0.34766033, 2)]], dtype="float32")).to(device)

    #mu = torch.from_numpy(np.array([0.0, 0.0], dtype="float32")).to(device)
    #S = torch.from_numpy(np.array([[pow(1.0,2), 0.0], [0.0, pow(1.0,2)]], dtype="float32")).to(device)

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

    #print(first_term_loss)
    #first_term_loss = compute_loss2(genFGen2, args, model)

    #print(genFGen2)
    #dasfasdfs

    #first_term_loss = compute_loss2(genFGen2, args, model)
    #first_term_loss = compute_loss2(genFGen2, model)

    #print(xData.shape)
    #print(genFGen2.shape)

    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    imageStore = xData[0,:,:,:].squeeze().cpu().numpy()
    #imageStore = genFGen2[0, :, :, :].squeeze().cpu().detach().numpy()

    plt.imshow(imageStore)
    plt.show()
    """

    #pilTrans = transforms.ToTensor()
    #plt.imshow(xData[1, :])

    #first_term_loss = compute_loss2(genFGen2, model)
    #first_term_loss = compute_loss2(xData, model)

    #first_term_loss = compute_loss2(genFGen2, model)

    #first_term_loss = compute_loss2(genFGen2, model)
    #first_term_loss = compute_loss2(genFGen2, model)

    #first_term_loss = compute_loss2(genFGen2, model)

    #first_term_loss = compute_loss2(genFGen2, model)
    #first_term_loss = compute_loss2(xData, model)

    #print(xData)
    #print(genFGen2)

    #first_term_loss = compute_loss2(genFGen2, model)
    #first_term_loss = compute_loss2(genFGen2, model)

    first_term_loss = compute_loss2(genFGen2, model)
    #first_term_loss = compute_loss_original(genFGen2, model)

    #first_term_loss.retain_grad()

    #first_term_loss.retain_grad()
    first_term_loss.retain_grad()

    #print(first_term_loss)
    #print('')

    #print(first_term_loss)

    """
    second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    for i in range(args.batch_size):
        second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_() ** 2
        second_term_loss32[i] = torch.min(second_term_loss22)
    second_term_loss2 = torch.mean(second_term_loss32)
    """

    #print(first_term_loss)
    #print('')

    #print('')
    #print(compute_loss2(mu.unsqueeze(0), args, model))

    #print(torch.exp(toUse_storeAll.log_prob(mu)))
    #print('')

    #first_term_loss = storeAll

    #xData = toy_data.inf_train_gen(args.data, batch_size=args.batch_size)
    #xData = torch.from_numpy(xData).type(torch.float32).to(device)

    #print(xData.shape)
    #print(torch.mean(xData))
    #print(torch.std(xData))

    #xData = torch.empty((args.batch_size, 2), device=device)
    #xData[:args.batch_size//2, :] = toUse_storeAll.sample((args.batch_size//2,)) # .sample_n(args.batch_size // 2)
    #xData[args.batch_size//2:, :] = toUse_storeAll2.sample((args.batch_size//2,)) # .sample_n(args.batch_size//2)

    """
    xData = torch.empty((args.batch_sizeM, 2), device=device)
    xData[:args.batch_sizeM // 2, :] = toUse_storeAll.sample((args.batch_sizeM // 2,))  # .sample_n(args.batch_size // 2)
    xData[args.batch_sizeM // 2:, :] = toUse_storeAll2.sample((args.batch_sizeM // 2,))  # .sample_n(args.batch_size//2)
    """

    #xData = torch.empty((args.batch_size, 2)).normal_(mean=[2.82507515, 1.92882611e-04 + 0.8], std=0.5)
    #xData[args.batch_size//2:,:] = torch.empty((args.batch_size, 2)).normal_(mean=4, std=0.5)

    #mu = torch.from_numpy(np.array([2.82507515, 1.92882611e-04 + 0.8], dtype="float32")).to(device)
    #S = torch.from_numpy(np.array([[pow(0.07166782, 2), 0.0], [0.0, pow(0.06917527, 2)]], dtype="float32")).to(device)
    #mu2 = torch.from_numpy(np.array([2.82507515, 1.92882611e-04 - 0.8], dtype="float32")).to(device)
    #toUse_storeAll = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=S)
    #toUse_storeAll2 = torch.distributions.MultivariateNormal(loc=mu2, covariance_matrix=S)

    #print(xData.shape)
    #print(torch.mean(xData))
    #print(torch.std(xData))

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

    #args.batch_size = 2
    #genFGen2 = torch.from_numpy(np.array([[3, 0], [2, 0]], dtype="float32")).to(device)
    #xData = torch.from_numpy(np.array([[1, 0], [0, 1]], dtype="float32")).to(device)

    #import timeit
    #start = timeit.default_timer()
    #stop = timeit.default_timer()
    #print('Time: ', stop - start)

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

    #import timeit
    #start = timeit.default_timer()
    #stop = timeit.default_timer()
    #print('Time: ', stop - start)

    #print('')
    #print(second_term_loss2)

    #distances = torch.norm(vertices - point_locs, p=2, dim=1)
    #distances = torch.sqrt((vertices - point_locs).pow(2).sum(1))

    #import timeit
    #start = timeit.default_timer()
    #stop = timeit.default_timer()
    #print('Time: ', stop - start)

    #second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    second_term_loss32 = torch.empty(args.batchsize, device=device, requires_grad=False)
    #for i in range(args.batch_size):
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

        #print(torch.mean(torch.sqrt((genFGen2[i, :] - xData).view(args.batchsize, -1).pow(2).sum(1))))
        #print(torch.mean(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchsize, -1).pow(2).sum(1))))
        #print(torch.mean(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1))))
        #print('')

        #print(torch.sqrt((genFGen2[i, :] - xData).view(args.batchsize, -1).pow(2).sum(1)))
        #print(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchsize, -1).pow(2).sum(1)))
        #print(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1)))
        #print('')

        #second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p='fro', dim=1).requires_grad_()
        #second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()
        #second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()**2
        #second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1))**2
        #second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_()**2

        #second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2

        #second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2
        second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).view(args.batchsize, -1).pow(2).sum(1)).requires_grad_() ** 2

        #second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()
        #print(second_term_loss22.shape)
        second_term_loss32[i] = torch.min(second_term_loss22)
    #print(second_term_loss32)
    #print(second_term_loss32.shape)
    #print(torch.norm(genFGen2 - xData, p=None, dim=0).shape)
    #second_term_loss22 = torch.min(second_term_loss32)
    #print(second_term_loss22)
    #print(second_term_loss22.shape)
    #second_term_loss2 = torch.mean(second_term_loss32)
    #second_term_loss2 = 0.3 * torch.mean(second_term_loss32)
    #second_term_loss2 = 3.0 * torch.mean(second_term_loss32)
    #second_term_loss2 = 7.62939453125 * torch.mean(second_term_loss32)
    #print(second_term_loss2)
    #print(second_term_loss2.shape)

    #second_term_loss2 = 0.3 * torch.mean(second_term_loss32)

    #second_term_loss2 = 0.3 * torch.mean(second_term_loss32)
    #second_term_loss2 = 0.001 * torch.mean(second_term_loss32)

    #second_term_loss2 = 0.001 * torch.mean(second_term_loss32)

    #second_term_loss2 = 0.001 * torch.mean(second_term_loss32)
    second_term_loss2 = torch.mean(second_term_loss32)

    #second_term_loss2.retain_grad()

    #second_term_loss2.retain_grad()
    second_term_loss2.retain_grad()

    #import timeit
    #start = timeit.default_timer()

    #stop = timeit.default_timer()
    #print('Time: ', stop - start)

    #print(second_term_loss2)
    #print('')

    #print('')
    #print(first_term_loss)

    #print(second_term_loss2)
    #print('')

    #print(first_term_loss)
    #print(second_term_loss2)

    #second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    #for i in range(args.batch_size):
    #    second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_()**2
    #    second_term_loss32[i] = torch.min(second_term_loss22)
    #second_term_loss2 = torch.mean(second_term_loss32)

    #print(genFGen2.shape)
    #print(genFGen3.shape)

    #print(xData.shape)
    #print('')

    #third_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    third_term_loss32 = torch.empty(args.batchsize, device=device, requires_grad=False)
    #for i in range(args.batch_size):
    for i in range(args.batchsize):
        #print(xData.shape)
        #print(genFGen2.shape)

        #print(genFGen3.shape)
        #print('')

        #print(xData.squeeze().shape)
        #print(genFGen2.squeeze().shape)
        #print('')

        #print((genFGen2[i, :] - xData).pow(2).sum(1).shape)
        #print((genFGen2[i, :] - genFGen2).pow(2).sum(1).shape)

        #print((genFGen2[i, :].squeeze() - xData.squeeze()).pow(2).sum(1).shape)
        #print((genFGen2[i, :].squeeze() - genFGen2.squeeze()).pow(2).sum(1).shape)

        #print((genFGen3[i, :] - genFGen3).pow(2).sum(1).shape)
        #print('')

        #print(torch.norm(genFGen2[i, :] - xData, p=None, dim=2).shape)
        #print(torch.norm(genFGen2[i, :] - genFGen2, p=None, dim=2).shape)

        #print(torch.norm(genFGen2[i, :].squeeze() - xData.squeeze(), p=None, dim=2).shape)
        #print(torch.norm(genFGen2[i, :].squeeze() - genFGen2.squeeze(), p=None, dim=2).shape)

        #print(torch.norm(genFGen3[i, :] - genFGen3, p=None, dim=1).shape)
        #print('')

        #a = torch.randn(64, 3, 32, 32)
        #a = a.view(64, -1)
        #b = torch.norm(a, p=2, dim=1)

        #a = torch.randn(64, 1, 28, 28)
        #a = a.view(64, -1)
        #b = torch.norm(a, p=2, dim=1)

        #print((genFGen2[i, :] - xData).view(args.batchsize,-1).pow(2).sum(1).shape)
        #print((genFGen2[i, :] - genFGen2).view(args.batchsize,-1).pow(2).sum(1).shape)
        #print('')

        #print(torch.norm((genFGen2[i, :] - xData).view(args.batchsize,-1), p=None, dim=1).shape)
        #print(torch.norm((genFGen2[i, :] - genFGen2).view(args.batchsize,-1), p=None, dim=1).shape)
        #print('')

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

        #print(torch.mean(torch.sqrt((genFGen2[i, :] - xData).view(args.batchsize, -1).pow(2).sum(1))))
        #print(torch.mean(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchsize, -1).pow(2).sum(1))))
        #print(torch.mean(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1))))
        #print('')

        #print(torch.sqrt((genFGen2[i, :] - xData).view(args.batchsize, -1).pow(2).sum(1)))
        #print(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchsize, -1).pow(2).sum(1)))
        #print(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1)))
        #print('')

        #third_term_loss22 = (torch.norm(genFGen3[i, :] - genFGen3, p=None, dim=1).requires_grad_()) / (
        #            1.0e-17 + torch.norm(genFGen2[i, :] - genFGen2, p=None, dim=1).requires_grad_())
        #third_term_loss22 = (torch.sqrt(1e-17 + (genFGen3[i, :] - genFGen3).pow(2).sum(1)).requires_grad_()) / (
        #        1e-17 + torch.sqrt(1e-17 + (genFGen2[i, :] - genFGen2).pow(2).sum(1)).requires_grad_())

        #third_term_loss22 = (torch.sqrt(1e-17 + (genFGen3[i, :] - genFGen3).pow(2).sum(1)).requires_grad_()) / (
        #        1e-17 + torch.sqrt(1e-17 + (genFGen2[i, :] - genFGen2).pow(2).sum(1)).requires_grad_())

        third_term_loss22 = (torch.sqrt(1e-17 + (genFGen3[i, :] - genFGen3).pow(2).sum(1)).requires_grad_()) / (
                1e-17 + torch.sqrt(1e-17 + (genFGen2[i, :] - xData).view(args.batchsize, -1).pow(2).sum(1)).requires_grad_())

        third_term_loss32[i] = torch.mean(third_term_loss22)
    #third_term_loss12 = torch.mean(third_term_loss32)
    #third_term_loss12 = 0.01 * torch.mean(third_term_loss32)
    #third_term_loss12 = 0.025 * torch.mean(third_term_loss32)
    #third_term_loss12 = 0.25 * torch.mean(third_term_loss32)
    #third_term_loss12 = 0.1 * torch.mean(third_term_loss32)

    #third_term_loss12 = 0.25 * torch.mean(third_term_loss32)

    #third_term_loss12 = 0.25 * torch.mean(third_term_loss32)
    #third_term_loss12 = 0.1 * torch.mean(third_term_loss32)

    #third_term_loss12 = 0.1 * torch.mean(third_term_loss32)

    #third_term_loss12 = 0.1 * torch.mean(third_term_loss32)
    third_term_loss12 = torch.mean(third_term_loss32)

    #third_term_loss12.retain_grad()

    #third_term_loss12.retain_grad()
    third_term_loss12.retain_grad()

    #print(third_term_loss12)
    #print('')

    #return first_term_loss + second_term_loss2
    #return first_term_loss + second_term_loss2, xData
    #return first_term_loss + second_term_loss2 + third_term_loss12, xData

    #return first_term_loss + second_term_loss2 + third_term_loss12, xData
    #return first_term_loss + second_term_loss2 + third_term_loss12

    #print(first_term_loss)
    #print(second_term_loss2)

    #print(third_term_loss12)
    #print('')

    #torch.set_printoptions(sci_mode=False)

    #print(first_term_loss)
    #print('')

    """
    #print(torch.isnan(first_term_loss))
    if torch.isnan(first_term_loss):
        first_term_loss = 0.0
    """

    #print(first_term_loss)
    #print('')

    #return first_term_loss + second_term_loss2 + third_term_loss12
    #return first_term_loss + second_term_loss2 + third_term_loss12

    #print(second_term_loss2)
    #print(third_term_loss12)

    #if torch.isnan(first_term_loss):
    #    return second_term_loss2 + third_term_loss12
    #else:
    #    return first_term_loss + second_term_loss2 + third_term_loss12

    #return first_term_loss + second_term_loss2 + third_term_loss12

    #return first_term_loss + second_term_loss2 + third_term_loss12
    #return first_term_loss + second_term_loss2 + third_term_loss12

    #return first_term_loss + second_term_loss2 + third_term_loss12
    #return first_term_loss + second_term_loss2 + third_term_loss12, first_term_loss, second_term_loss2

    print('')
    print(first_term_loss.item())

    print(second_term_loss2.item())
    print(third_term_loss12.item())

    #print('')
    #print(first_term_loss.grad)

    #print(second_term_loss2.grad)
    #print(third_term_loss12.grad)

    print('')

    #total_totTotalLoss = first_term_loss * second_term_loss2 * third_term_loss12
    #total_totTotalLoss = first_term_loss + second_term_loss2 + third_term_loss12

    #total_totTotalLoss = first_term_loss + second_term_loss2 + third_term_loss12

    #total_totTotalLoss = first_term_loss + second_term_loss2 + third_term_loss12
    #total_totTotalLoss = first_term_loss + 0.001 * second_term_loss2 + 0.1 * third_term_loss12

    #total_totTotalLoss = first_term_loss + 0.001 * second_term_loss2 + 0.1 * third_term_loss12

    #total_totTotalLoss = first_term_loss + 0.001 * second_term_loss2 + 0.1 * third_term_loss12
    #total_totTotalLoss = first_term_loss + 0.001 * second_term_loss2 + 10.0 * third_term_loss12

    #total_totTotalLoss = first_term_loss + 0.001 * second_term_loss2 + 0.1 * third_term_loss12
    #total_totTotalLoss = first_term_loss + 10.0 * second_term_loss2 + 0.1 * third_term_loss12

    #total_totTotalLoss = first_term_loss + 0.001 * second_term_loss2 + 0.1 * third_term_loss12

    #total_totTotalLoss = first_term_loss + 0.001 * second_term_loss2 + 0.1 * third_term_loss12
    total_totTotalLoss = first_term_loss + 1.0 * second_term_loss2 + 0.1 * third_term_loss12

    #total_totTotalLoss.retain_grad()

    #total_totTotalLoss.retain_grad()
    total_totTotalLoss.retain_grad()

    #return first_term_loss + second_term_loss2 + third_term_loss12, first_term_loss, second_term_loss2
    #return first_term_loss + second_term_loss2 + third_term_loss12, first_term_loss, second_term_loss2, third_term_loss12

    #return first_term_loss + second_term_loss2 + third_term_loss12, first_term_loss, second_term_loss2, third_term_loss12
    return total_totTotalLoss, first_term_loss, second_term_loss2, third_term_loss12


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


def train(epoch, model, model_copy):

    model = parallelize(model)
    model.train()

    #model.train()

    total = 0
    correct = 0

    end = time.time()

    for i, (x, y) in enumerate(train_loader):

        #print(x.shape)
        #print(y.shape)

        #print(y)

        for i21 in range(len(y)):
            if y[i21] == 0 and i21 == 0:
                y[i21] = y[i21+1]
                x[i21, :, :, :] = x[i21+1, :, :, :]
            elif y[i21] == 0:
                y[i21] = y[i21 - 1]
                x[i21, :, :, :] = x[i21 - 1, :, :, :]

        #print(y)
        #asdfsf

        global_itr = epoch * len(train_loader) + i
        update_lr(optimizer, global_itr)

        # Training procedure:
        # for each sample x:
        #   compute z = f(x)
        #   maximize log p(x) = log p(z) - log |det df/dx|

        x = x.to(device)

        #model = model.to(device)

        beta = beta = min(1, global_itr / args.annealing_iters) if args.annealing_iters > 0 else 1.
        #bpd, logits, logpz, neg_delta_logp = compute_loss(x, model, model_copy, beta=beta)

        #bpd, logits, logpz, neg_delta_logp = compute_loss(x, model, model_copy, beta=beta)

        #bpd, logits, logpz, neg_delta_logp = compute_loss(x, model, model_copy, beta=beta)
        #bpd, logits, logpz, neg_delta_logp = compute_loss(x, model, model_copy, beta=beta)

        #bpd, logits, logpz, neg_delta_logp = compute_loss(x, model, beta=beta)

        #bpd, logits, logpz, neg_delta_logp = compute_loss(x, model, beta=beta)
        #bpd, logits, logpz, neg_delta_logp = compute_loss(x, model, beta=beta)

        #rand_z = torch.randn(args.batchsize, (im_dim + args.padding) * args.imagesize * args.imagesize).to(device)
        #bpd = use_loss_fn2(model(rand_z, inverse=True).view(-1, *input_size[1:]), args, model_copy, rand_z, x)

        #fixed_z = standard_normal_sample([min(32, args.batchsize),
        #                                  (im_dim + args.padding) * args.imagesize * args.imagesize]).to(device)

        # use: model(fixed_z, inverse=True).view(-1, *input_size[1:])
        #bpd = use_loss_fn2(model(fixed_z, inverse=True).view(-1, *input_size[1:]), args, model_copy, fixed_z, x)

        # use: torch.randn([args.batchsize, nrand], device=device, requires_grad=True)
        #rand_z = torch.randn(args.batchsize, (im_dim + args.padding) * args.imagesize * args.imagesize).to(device)

        #rand_z = torch.randn(args.batchsize, (im_dim + args.padding) * args.imagesize * args.imagesize).to(device)
        #rand_z2 = torch.randn([args.batchsize, (im_dim + args.padding) * args.imagesize * args.imagesize], device=device, requires_grad=True)

        #print(rand_z)
        #print(rand_z2)

        #print(rand_z.shape)
        #print(rand_z2.shape)

        #rand_z = torch.randn([args.batchsize, 200],
        #                     device=device, requires_grad=True)

        #rand_z = torch.randn([args.batchsize, (im_dim + args.padding) * args.imagesize * args.imagesize],
        #                      device=device, requires_grad=True)

        #fixFixed_x = model(rand_z, inverse=True).view(-1, *input_size[1:])
        #bpd = use_loss_fn2(fixFixed_x, args, model_copy, rand_z, x)

        #fixFixed_x = model(rand_z.detach(), inverse=True).view(-1, *input_size[1:])
        #bpd = use_loss_fn2(fixFixed_x, args, model_copy, rand_z, x)

        #from torch.autograd import Variable
        #fixFixed_x = model(Variable(rand_z.detach(), requires_grad=True), inverse=True).view(-1, *input_size[1:])

        #fixFixed_x = model(rand_z.detach(), inverse=True).view(-1, *input_size[1:])
        #fixFixed_x = model(rand_z.detach(), inverse=True).view(-1, *input_size[1:])

        #with torch.no_grad():
        #    model.eval()
        #    fixFixed_x = model(rand_z.detach(), inverse=True).view(-1, *input_size[1:])
        #    model.train()

        #with torch.no_grad():
        #    fixFixed_x = model(rand_z, inverse=True).view(-1, *input_size[1:])

        #z, delta_logp = model(x.view(-1, *input_size[1:]), 0)
        #fixFixed_x = model(rand_z, 0, inverse=True).view(-1, *input_size[1:])

        model_copy.eval()

        for param in model_copy.parameters():
            param.requires_grad = False

        #model.eval()
        #with torch.no_grad():
        #    fixFixed_x = model(rand_z, inverse=True).view(-1, *input_size[1:])
        #model.train()
        #bpd = use_loss_fn2(fixFixed_x, args, model_copy, rand_z, x)

        #fixFixed_x = model(rand_z, inverse=True).view(-1, *input_size[1:])
        #bpd = use_loss_fn2(fixFixed_x, args, model_copy, rand_z, x)

        #fixFixed_x = model.module.inverse(rand_z).view(-1, *input_size[1:])
        #bpd = use_loss_fn2(fixFixed_x, args, model_copy, rand_z, x)

        '''
        model.eval()
        with torch.no_grad():
            fixFixed_x = model.module.inverse(rand_z).view(-1, *input_size[1:])
        model.train()
        #bpd = use_loss_fn2(fixFixed_x, args, model_copy, rand_z, x)

        #lossGen, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(genFGen2, args, model, ggenFGen2, x)
        bpd, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(fixFixed_x, args, model_copy, rand_z, x)
        '''

        """
        model.eval()
        with torch.no_grad():
            fixFixed_x = model.module.inverse(rand_z).view(-1, *input_size[1:])
        model.train()
        #from torch.autograd import Variable
        #fixFixed_x = Variable(fixFixed_x, requires_grad=True)
        bpd, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(fixFixed_x, args, model_copy, rand_z, x)

        #fixFixed_x = model.module.inverse(rand_z).view(-1, *input_size[1:])
        #bpd, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(fixFixed_x, args, model_copy, rand_z, x)
        """

        """
        teTemp_logpz = standard_normal_logprob(rand_z).view(rand_z.size(0), -1).sum(1, keepdim=True)
        fixFixed_x = model.module.inverse(rand_z, teTemp_logpz).view(-1, *input_size[1:])

        bpd, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(fixFixed_x, args, model_copy,
                                                                                     rand_z, x)
        """

        #from torch.autograd import Variable
        #rand_z = Variable(rand_z, requires_grad=True)

        #fixFixed_x = model.module.inverse(rand_z).view(-1, *input_size[1:])
        #bpd, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(fixFixed_x, args, model_copy, rand_z, x)

        rand_z = torch.randn([args.batchsize, (im_dim + args.padding) * args.imagesize * args.imagesize],
                             device=device, requires_grad=True)

        fixFixed_x = model.module.inverse(rand_z).view(-1, *input_size[1:])
        bpd, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(fixFixed_x, args, model_copy, rand_z, x)

        logpz = firstOnly_lossGen
        neg_delta_logp = secondOnly_lossGen

        #model.eval()
        #with torch.no_grad():
        #    fixFixed_x = model.module.inverse(rand_z.detach()).view(-1, *input_size[1:])
        #model.train()
        #bpd, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(fixFixed_x, args, model_copy, rand_z, x)

        #teTemp_logpz = standard_normal_logprob(rand_z).view(rand_z.size(0), -1).sum(1, keepdim=True)
        #fixFixed_x = model.module.inverse(rand_z, teTemp_logpz).view(-1, *input_size[1:])
        #bpd, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(fixFixed_x, args, model_copy, rand_z, x)

        #fixFixed_x = model(rand_z, inverse=True).view(-1, *input_size[1:])
        #bpd = use_loss_fn2(fixFixed_x, args, model_copy, rand_z, x)

        #fixFixed_x = model(rand_z.view(-1, *input_size[1:]))
        #fixFixed_x = model(rand_z, inverse=True).view(-1, *input_size[1:])

        #fixFixed_x = model(rand_z, inverse=True).view(-1, *input_size[1:])
        #fixFixed_x = model(rand_z, 0, inverse=True).view(-1, *input_size[1:])

        #fixFixed_x = model(rand_z, inverse=True).view(-1, *input_size[1:])
        #bpd = use_loss_fn2(fixFixed_x, args, model_copy, rand_z, x)

        #rand_z = torch.randn(args.batchsize, (im_dim + args.padding) * args.imagesize * args.imagesize).to(device)
        #fake_imgs = model(rand_z, inverse=True).view(-1, *input_size[1:])

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

            # Compute accuracy
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

        #print(i)
        #adfa

        #if i % args.vis_freq == 0:
        if (i % args.vis_freq == 0) and (i > 0):
            #visualize(epoch, model, i, x)

            #visualize(epoch, model, i, x)
            visualize(epoch, model, i, x, bpd.item())

            #torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            #}, os.path.join(args.save, 'models', 'moostMoostMoostRecent.pth'))

            #torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            #}, os.path.join(args.save, 'models', 'moosstMoosstMoosstRecent.pth'))

            #torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            #}, os.path.join(args.save, 'models', 'moossttMoossttMoossttRecent.pth'))

            #torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            #}, os.path.join(args.save, 'models', 'moossttMoossttMoossttMoossttRecent.pth'))

            #torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            #}, os.path.join(args.save, 'models', 'moosstttMoosstttMoosstttMoosstttRecent.pth'))

            #torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            #}, os.path.join(args.save, 'models', 'moosstttMoosstttMoosstttMoosstttRecent2.pth'))

            #torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            #}, os.path.join(args.save, 'models', 'moosstttMoosstttMoosstttMoosstttRecent3.pth'))

            #torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            #}, os.path.join(args.save, 'models', 'moosstttMoosstttMoosstttMoosstttRecent4.pth'))

            #torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            #}, os.path.join(args.save, 'models', 'moosstttMoosstttMoosstttMoosstttRecent5.pth'))

            #torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            #}, os.path.join(args.save, 'models', 'moosstttMoosstttMoosstttMoosstttRecent6.pth'))

            #torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            #}, os.path.join(args.save, 'models', 'moosstttMoosstttMoosstttMoosstttRecent7.pth'))

            #torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            #}, os.path.join(args.save, 'models', 'moosstttMoosstttMoosstttMoosstttRecent8.pth'))

            #torch.save({
            #    'state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'args': args,
            #    'ema': ema,
            #    'test_bpd': test_bpd,
            #}, os.path.join(args.save, 'models', 'moosstttMoosstttMoosstttMoosstttRecent9.pth'))

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


def visualize(epoch, model, itr, real_imgs, lossLoss41):
    model.eval()
    utils.makedirs(os.path.join(args.save, 'imgs'))
    real_imgs = real_imgs[:32]
    _real_imgs = real_imgs

    if args.data == 'celeba_5bit':
        nvals = 32
    elif args.data == 'celebahq':
        nvals = 2**args.nbits
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

        #filename = os.path.join(args.save, 'imgs', 'e{:03d}_i{:06d}.png'.format(epoch, itr))

        #filename = os.path.join(args.save, 'imgs', 'e{:03d}_i{:06d}.png'.format(epoch, itr))
        #filename = os.path.join(args.save, 'imgs', 'e{:03d}_i{:06d}.png'.format(epoch, itr))

        #filename = os.path.join(args.save, 'imgs', 'e{:03d}_i{:06d}.png'.format(epoch, itr))
        #filename = os.path.join(args.save, 'imgs', 'ee{:03d}i{:06d}.png'.format(epoch, itr))

        #filename = os.path.join(args.save, 'imgs', 'ee{:03d}i{:06d}.png'.format(epoch, itr))
        filename = os.path.join(args.save, 'imgs', 'LF{:}B{:}Ep{:03d}It{:06d}.png'.format(lossLoss41, args.batchsize, epoch, itr))

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

    """
    model_copy = type(mymodel)() # get a new instance
    model_copy.load_state_dict(mymodel.state_dict()) # copy weights

    #model_clone = model_cls(**kwargs)
    #model_clone.load_state_dict(copy.deepcopy(original_model.state_dict()))
    """

    # https://pytorch.org/docs/0.3.0/notes/serialization.html
    # use: https://pytorch.org/docs/0.3.0/notes/serialization.html

    '''
    model_copy = type(mymodel)() # get a new instance
    model_copy.load_state_dict(mymodel.state_dict()) # copy weights
    '''

    model_copy = model # get a new instance
    #model_copy.load_state_dict(model.state_dict()) # copy weights

    #model_copy.load_state_dict(model.state_dict()) # copy weights

    #model_copy.load_state_dict(model.state_dict()) # copy weights
    model_copy.load_state_dict(model.state_dict()) # copy weights

    #model_copy.detach()
    model_copy.eval()

    for param in model_copy.parameters():
        param.requires_grad = False

    # (1) torch.save(the_model.state_dict(), PATH)
    # the_model = TheModelClass(*args, **kwargs)
    # the_model.load_state_dict(torch.load(PATH))

    # or (2) torch.save(the_model, PATH)
    # the_model = torch.load(PATH)

    # if args.resume:
    #     validate(args.begin_epoch - 1, model, ema)

    # if args.resume:
    #     validate(args.begin_epoch - 1, model)

    for epoch in range(args.begin_epoch, args.nepochs):

        logger.info('Current LR {}'.format(optimizer.param_groups[0]['lr']))

        train(epoch, model, model_copy)
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

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'most_recent.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mostMostMostRecent.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmooooostMostMostRecent.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmmoooooostMostMostRecent.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmmoooooostMostMostRecent2.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmmoooooostMostMostRecent3.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmmoooooostMostMostRecent4.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmmoooooostMostMostRecent5.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmmoooooostMostMostRecent6.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmmoooooostMostMostRecent7.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmmoooooostMostMostRecent8.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmmoooooostMostMostRecent9.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmmoooooostMostMostRecent99.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmmoooooostMostMostRecent999.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmmoooooostMostMostRecent9999.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmmoooooostMostMostRecent99999.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmmoooooostMostMostRecent999999.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmmoooooostMostMostRecent9999999.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmooooostMostMostRecent99999999.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmmooooostMostMostRecent999999999.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmmoooostMostMostRecent9999999999.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmmooostMostMostRecent99999999999.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mmoostMostMostRecent999999999999.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mostMostMostRecent9999999999999.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mostMostMostReRecent99999999999999.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mostMostMostReReRecent999999999999999.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mostMostMostRecRecRecent9999999999999999.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'mostMostMostRecRecRecRecent99999999999999999.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'moRec99.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'moRec9999.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'moRec99rec99.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'moRe991re991re991.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'moRe977re977re977re977.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'moMoRe9777re9777re9777re9777.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'moMoMoRee9777ree9777ree9777ree9777.pth'))

        #torch.save({
        #    'state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'args': args,
        #    'ema': ema,
        #    'test_bpd': test_bpd,
        #}, os.path.join(args.save, 'models', 'moMoMoMoRRee9777rree9777rree9777rree9777.pth'))

        torch.save({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args,
            'ema': ema,
            'test_bpd': test_bpd,
        }, os.path.join(args.save, 'models', '00moMoMoMoRRee98rree98rree98rree98.pth'))


if __name__ == '__main__':
    main()

