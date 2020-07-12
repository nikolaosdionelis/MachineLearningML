import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

#import os
import time
import argparse

import sys
import torch

print('')
print(sys.version)
print(torch.__version__)

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('')
import math
import numpy as np

import lib.utils as utils
import lib.layers as layers
import lib.optimizers as optim

import lib.toy_data as toy_data
import lib.layers.base as base_layers
from lib.visualize_flow import visualize_transform
from lib.visualize_flow import visualize_transform2

ACTIVATION_FNS = {
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh,
    'elu': torch.nn.ELU,
    'selu': torch.nn.SELU,
    'fullsort': base_layers.FullSort,
    'maxmin': base_layers.MaxMin,
    'swish': base_layers.Swish,
    'lcube': base_layers.LipschitzCube,
}

parser = argparse.ArgumentParser()

# Use 8 Gaussians
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='8gaussians')

parser.add_argument('--arch', choices=['iresnet', 'realnvp'], default='iresnet')
parser.add_argument('--coeff', type=float, default=0.9)

parser.add_argument('--vnorms', type=str, default='222222')
parser.add_argument('--n-lipschitz-iters', type=int, default=5)

parser.add_argument('--atol', type=float, default=None)
parser.add_argument('--rtol', type=float, default=None)

parser.add_argument('--learn-p', type=eval, choices=[True, False], default=False)
parser.add_argument('--mixed', type=eval, choices=[True, False], default=True)

parser.add_argument('--dims', type=str, default='128-128-128-128')
parser.add_argument('--act', type=str, choices=ACTIVATION_FNS.keys(), default='swish')

parser.add_argument('--nblocks', type=int, default=100)
parser.add_argument('--brute-force', type=eval, choices=[True, False], default=False)

parser.add_argument('--actnorm', type=eval, choices=[True, False], default=False)
parser.add_argument('--batchnorm', type=eval, choices=[True, False], default=False)

parser.add_argument('--exact-trace', type=eval, choices=[True, False], default=False)
parser.add_argument('--n-power-series', type=int, default=None)

parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='geometric')

parser.add_argument('--niters', type=int, default=5000)
parser.add_argument('--niters2', type=int, default=1501)

parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--test_batch_size', type=int, default=500)

parser.add_argument('--weight-decay', type=float, default=1e-5)
parser.add_argument('--annealing-iters', type=int, default=0)

parser.add_argument('--save', type=str, default='experiments/')

parser.add_argument('--viz_freq', type=int, default=1000)
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--log_freq', type=int, default=1000)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()

# logger
utils.makedirs(args.save)

logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

print('')
print(device)
print(device.type)

print('')
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def standard_normal_sample(size):
    return torch.randn(size)

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

def compute_loss(args, model, batch_size=None, beta=1.):
    if batch_size is None:
        batch_size = args.batch_size

    # load data
    x = toy_data.inf_train_gen(args.data, batch_size=batch_size)
    x = torch.from_numpy(x).type(torch.float32).to(device)

    zero = torch.zeros(x.shape[0], 1).to(x)

    # transform to z
    z, delta_logp = model(x, zero)

    # compute log p(z)
    logpz = standard_normal_logprob(z).sum(1, keepdim=True)

    logpx = logpz - beta * delta_logp
    loss = -torch.mean(logpx)

    return loss, torch.mean(logpz), torch.mean(-delta_logp)

# x is a Tensor => batch_size x 2
def compute_loss2(x, args, model, batch_size=None, beta=1.):
   if batch_size is None:
       batch_size = args.batch_size

   zero = torch.zeros(x.shape[0], 1).to(x)

   # transform to z
   z, delta_logp = model(x, zero)

   # x is a Tensor => batch_size x 2
   # x is the same as args.data => batch_size x 2

   # compute log p(z)
   logpz = standard_normal_logprob(z).sum(1, keepdim=True)

   # compute log p(x)
   logpx = logpz - beta * delta_logp

   #return torch.mean(logpx)
   return torch.mean(torch.exp(logpx))

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable

# GAN Model
class Generator(nn.Module):
   def __init__(self, nhidden):
       super(Generator, self).__init__()

       self.lin1 = nn.Linear(2, nhidden)
       self.lin2 = nn.Linear(nhidden, nhidden)
       self.lin3 = nn.Linear(nhidden, 2)

   def forward(self, z):
       h = F.relu(self.lin1(z))
       x2 = F.relu(self.lin2(h))

       x = self.lin3(x2)
       return x

# # FenceGAN Model
#def get_generative():
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
    first_term_loss = compute_loss2(genFGen2, args, model)

    #print('')
    #print(first_term_loss)

    #import math
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

    second_term_loss = torch.empty(size=(len(genFGen2), len(xData))).to(device)
    for i in range(len(genFGen2)):
        for j in range(len(xData)):
            second_term_loss[i, j] = torch.dist(genFGen2[i,:], xData[j,:], 2)
    second_term_loss, _ = torch.min(second_term_loss, 1)
    second_term_loss = torch.mean(second_term_loss)
    second_term_loss *= 100.0 / (args.batch_size**2)

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

    #print(third_term_loss)
    #print('')

    print('')
    #asdfsfa

    #return first_term_loss + second_term_loss + third_term_loss
    return first_term_loss + second_term_loss

def parse_vnorms():
    ps = []

    for p in args.vnorms:
        if p == 'f':
            ps.append(float('inf'))

        else:
            ps.append(float(p))

    return ps[:-1], ps[1:]

def compute_p_grads(model):
    scales = 0.
    nlayers = 0

    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            scales = scales + m.compute_one_iter()
            nlayers += 1

    scales.mul(1 / nlayers).mul(0.01).backward()

    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            if m.domain.grad is not None and torch.isnan(m.domain.grad):
                m.domain.grad = None

def build_nnet(dims, activation_fn=torch.nn.ReLU):
    nnet = []

    domains, codomains = parse_vnorms()
    if args.learn_p:
        if args.mixed:
            domains = [torch.nn.Parameter(torch.tensor(0.)) for _ in domains]
        else:
            domains = [torch.nn.Parameter(torch.tensor(0.))] * len(domains)
        codomains = domains[1:] + [domains[0]]
    for i, (in_dim, out_dim, domain, codomain) in enumerate(zip(dims[:-1], dims[1:], domains, codomains)):
        nnet.append(activation_fn())
        nnet.append(
            base_layers.get_linear(
                in_dim,
                out_dim,
                coeff=args.coeff,
                n_iterations=args.n_lipschitz_iters,
                atol=args.atol,
                rtol=args.rtol,
                domain=domain,
                codomain=codomain,
                zero_init=(out_dim == 2),
            )
        )
    return torch.nn.Sequential(*nnet)

def update_lipschitz(model, n_iterations):
    for m in model.modules():
        if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
            m.compute_weight(update=True, n_iterations=n_iterations)

        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            m.compute_weight(update=True, n_iterations=n_iterations)

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
    return('[[' + ','.join(list(map(lambda i: '{}'.format(i), a))) + ']]')

if __name__ == '__main__':
    activation_fn = ACTIVATION_FNS[args.act]

    if args.arch == 'iresnet':
        dims = [2] + list(map(int, args.dims.split('-'))) + [2]

        blocks = []
        if args.actnorm: blocks.append(layers.ActNorm1d(2))

        for _ in range(args.nblocks):
            blocks.append(
                layers.iResBlock(
                    build_nnet(dims, activation_fn),
                    n_dist=args.n_dist,
                    n_power_series=args.n_power_series,
                    exact_trace=args.exact_trace,
                    brute_force=args.brute_force,
                    n_samples=args.n_samples,
                    neumann_grad=False,
                    grad_in_forward=False,
                )
            )

            if args.actnorm: blocks.append(layers.ActNorm1d(2))
            if args.batchnorm: blocks.append(layers.MovingBatchNorm1d(2))

        model = layers.SequentialFlow(blocks).to(device)

    elif args.arch == 'realnvp':
        blocks = []

        for _ in range(args.nblocks):
            blocks.append(layers.CouplingLayer(2, swap=False))
            blocks.append(layers.CouplingLayer(2, swap=True))

            if args.actnorm: blocks.append(layers.ActNorm1d(2))
            if args.batchnorm: blocks.append(layers.MovingBatchNorm1d(2))

        model = layers.SequentialFlow(blocks).to(device)

    #logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    hiddenLayers = 8
    genGen = Generator(hiddenLayers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizerGen = optim.Adam(genGen.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)

    logpz_meter = utils.RunningAverageMeter(0.93)
    delta_logp_meter = utils.RunningAverageMeter(0.93)

    end = time.time()
    best_loss = float('inf')

    #model = torch.nn.DataParallel(model)
    model.train()

    for itr in range(1, 2):
        optimizer.zero_grad()
        #optimizerGen.zero_grad()

        beta = min(1, itr / args.annealing_iters) if args.annealing_iters > 0 else 1.
        loss, logpz, delta_logp = compute_loss(args, model, beta=beta)

        #genFGen2 = genGen.forward(torch.cuda.FloatTensor(args.batch_size, 2).normal_())
        #lossGen = loss_fn2(genFGen2, args, model)

        #plt.figure()
        #plt.plot(genFGen2[:, 0].cpu().detach().numpy(), genFGen2[:, 1].cpu().detach().numpy(), 'o')

        ##plt.ion()
        #plt.show()
        #plt.pause(1)

        loss_meter.update(loss.item())
        logpz_meter.update(logpz.item())

        delta_logp_meter.update(delta_logp.item())

        loss.backward()
        #lossGen.backward()

        if args.learn_p and itr > args.annealing_iters:
            compute_p_grads(model)

        optimizer.step()
        #optimizerGen.step()

        update_lipschitz(model, args.n_lipschitz_iters)
        time_meter.update(time.time() - end)

        logger.info(
            'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f})'
            ' | Logp(z) {:.6f}({:.6f}) | DeltaLogp {:.6f}({:.6f})'.format(
                itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, logpz_meter.val, logpz_meter.avg,
                delta_logp_meter.val, delta_logp_meter.avg
            )
        )

        if itr % args.val_freq == 0 or itr == args.niters:
            update_lipschitz(model, 200)
            with torch.no_grad():
                model.eval()
                test_loss, test_logpz, test_delta_logp = compute_loss(args, model, batch_size=args.test_batch_size)
                log_message = (
                    '[TEST] Iter {:04d} | Test Loss {:.6f} '
                    '| Test Logp(z) {:.6f} | Test DeltaLogp {:.6f}'.format(
                        itr, test_loss.item(), test_logpz.item(), test_delta_logp.item()
                    )
                )
                logger.info(log_message)

                logger.info('Ords: {}'.format(pretty_repr(get_ords(model))))

                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    utils.makedirs(args.save)
                    torch.save({
                        'args': args,
                        'state_dict': model.state_dict(),
                    }, os.path.join(args.save, 'checkpt.pth'))
                model.train()

        if itr % args.viz_freq == 0:
            with torch.no_grad():
                model.eval()

                p_samples = toy_data.inf_train_gen(args.data, batch_size=20000)
                sample_fn, density_fn = model.inverse, model.forward

                plt.figure(figsize=(9, 3))
                visualize_transform(p_samples, torch.randn, standard_normal_logprob, transform=sample_fn,
                                    inverse_transform=density_fn, samples=True, npts=400, device=device)

                fig_filename = os.path.join(args.save, 'figs', '{:04d}.jpg'.format(itr))
                print('')

                print(fig_filename)
                print('')

                utils.makedirs(os.path.dirname(fig_filename))
                plt.savefig(fig_filename)

                #plt.ion()
                plt.show()
                plt.pause(0.5)

                plt.close()
                model.train()

            end = time.time()

    logger.info('Training 1 has finished.')

    #utils.save_checkpoint({'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
    #                       'args': args}, os.path.join(args.save, 'models'), args.niters)

    #utils.save_checkpoint({'state_dict': model.state_dict()}, os.path.join(args.save, 'models'), args.niters)
    #adsfgdsgsdfdsaa

    #checkpoint = torch.load(os.path.join(os.path.join(args.save, 'models'), 'checkpt-%04d.pth' % args.niters))
    checkpoint = torch.load(os.path.join(os.path.join(args.save, 'models'), 'checkpt-%04d.pth' % args.niters), map_location = torch.device('cpu'))

    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #args.load_state_dict(checkpoint['args'])

    time2_meter = utils.RunningAverageMeter(0.93)
    loss2_meter = utils.RunningAverageMeter(0.93)

    end = time.time()
    best_loss = float('inf')

    for itr in range(1, args.niters2 + 1):
        #optimizer.zero_grad()
        optimizerGen.zero_grad()

        #beta = min(1, itr / args.annealing_iters) if args.annealing_iters > 0 else 1.
        #loss, logpz, delta_logp = compute_loss(args, model, beta=beta)

        #genFGen2 = genGen.forward(torch.cuda.FloatTensor(args.batch_size, 2).normal_())
        genFGen2 = genGen.forward(torch.FloatTensor(args.batch_size, 2).normal_().to(device))

        #genFGen2 = genGen.forward(torch.cuda.FloatTensor(args.batch_size, 2).normal_())
        lossGen = loss_fn2(genFGen2, args, model)

        xData = toy_data.inf_train_gen(args.data, batch_size=args.batch_size)
        xData = torch.from_numpy(xData).type(torch.float32).to(device)

        if (itr-1)%20 == 0:
            plt.figure()

            plt.plot(xData[:, 0].cpu().squeeze().numpy(), xData[:, 1].cpu().squeeze().numpy(), '+r')
            plt.plot(genFGen2[:, 0].cpu().detach().numpy(), genFGen2[:, 1].cpu().detach().numpy(), 'ob')

            plt.grid()
            plt.xlim(-1.0, 4.5)
            plt.ylim(-1.5, 1.5)

            fig_filename = os.path.join(args.save, 'figs', 'ffii{:04d}.jpg'.format(itr))
            print('')

            print(fig_filename)
            print('')

            utils.makedirs(os.path.dirname(fig_filename))
            plt.savefig(fig_filename)

            #plt.ion()
            #plt.show()
            #plt.pause(0.1)
            #plt.close()

            utils.save_checkpoint({'state_dict': genGen.state_dict()}, os.path.join(args.save, 'myModels'), itr)
            #utils.save_checkpoint({'state_dict': genGen.state_dict()}, os.path.join(args.save, 'myModels'), args.niters2)

        #loss_meter.update(loss.item())
        #logpz_meter.update(logpz.item())

        loss2_meter.update(lossGen.item())

        #delta_logp_meter.update(delta_logp.item())

        #loss.backward()
        lossGen.backward()

        #if args.learn_p and itr > args.annealing_iters:
        #    compute_p_grads(model)

        #optimizer.step()
        optimizerGen.step()

        #update_lipschitz(model, args.n_lipschitz_iters)
        time2_meter.update(time.time() - end)

        logger.info('Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f})'.format(
                itr, time2_meter.val, time2_meter.avg, loss2_meter.val, loss2_meter.avg))

        #if itr % args.val_freq == 0 or itr == args.niters:
        #    update_lipschitz(model, 200)
        #    with torch.no_grad():
        #        model.eval()
        #        test_loss, test_logpz, test_delta_logp = compute_loss(args, model, batch_size=args.test_batch_size)
        #        log_message = (
        #            '[TEST] Iter {:04d} | Test Loss {:.6f} '
        #            '| Test Logp(z) {:.6f} | Test DeltaLogp {:.6f}'.format(
        #                itr, test_loss.item(), test_logpz.item(), test_delta_logp.item()
        #            )
        #        )
        #        logger.info(log_message)

        #        logger.info('Ords: {}'.format(pretty_repr(get_ords(model))))

        #        if test_loss.item() < best_loss:
        #            best_loss = test_loss.item()
        #            utils.makedirs(args.save)
        #            torch.save({
        #                'args': args,
        #                'state_dict': model.state_dict(),
        #            }, os.path.join(args.save, 'checkpt.pth'))
        #        model.train()

        #if itr == 1 or itr % args.viz_freq == 0:
        #    with torch.no_grad():
        #        model.eval()

        #        p_samples = toy_data.inf_train_gen(args.data, batch_size=20000)
        #        sample_fn, density_fn = model.inverse, model.forward

        #        plt.figure(figsize=(9, 3))
        #        visualize_transform(p_samples, torch.randn, standard_normal_logprob, transform=sample_fn,
        #                            inverse_transform=density_fn, samples=True, npts=400, device=device)

        #        fig_filename = os.path.join(args.save, 'figs', '{:04d}.jpg'.format(itr))
        #        print('')

        #        print(fig_filename)
        #        print('')

        #        utils.makedirs(os.path.dirname(fig_filename))
        #        plt.savefig(fig_filename)

        #        #plt.ion()
        #        plt.show()
        #        plt.pause(0.5)

        #        plt.close()
        #        model.train()

        #    end = time.time()

    logger.info('Training 2 has finished.')

