import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.utils as vutils

#import os
#import seaborn as sns

import pickle
import math 

import utils 
import hmc 

from torch.distributions.normal import Normal

real_label = 1
fake_label = 0
criterion = nn.BCELoss()
criterion_mse = nn.MSELoss()

def dcgan(dat, netG, netD, args):
    device = args.device
    X_training = dat['X_train'].to(device)
    fixed_noise = torch.randn(args.num_gen_images, args.nz, 1, 1, device=device)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, 0.999)) 
    for epoch in range(1, args.epochs+1):
        for i in range(0, len(X_training), args.batchSize):
            netD.zero_grad()
            stop = min(args.batchSize, len(X_training[i:]))
            real_cpu = X_training[i:i+stop].to(device)

            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z)))

            netG.zero_grad()
            label.fill_(real_label) 
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            ## log performance
            if i % args.log == 0:
                print('Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
                        % (epoch, args.epochs, i, len(X_training), errD.data, errG.data, D_x, D_G_z1, D_G_z2))

        print('*'*100)
        print('End of epoch {}'.format(epoch))
        print('*'*100)

        if epoch % args.save_imgs_every == 0:
            fake = netG(fixed_noise).detach()
            vutils.save_image(fake, '%s/dcgan_%s_fake_epoch_%03d.png' % (args.results_folder, args.dataset, epoch), normalize=True, nrow=20) 

        if epoch % args.save_ckpt_every == 0:
            torch.save(netG.state_dict(), os.path.join(args.results_folder, 'netG_dcgan_%s_epoch_%s.pth'%(args.dataset, epoch)))


def presgan(dat, netG, netD, log_sigma, args):
    device = args.device
    X_training = dat['X_train'].to(device)
    fixed_noise = torch.randn(args.num_gen_images, args.nz, 1, 1, device=device)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, 0.999)) 
    sigma_optimizer = optim.Adam([log_sigma], lr=args.sigma_lr, betas=(args.beta1, 0.999))
    if args.restrict_sigma:
        logsigma_min = math.log(math.exp(args.sigma_min) - 1.0)
        logsigma_max = math.log(math.exp(args.sigma_max) - 1.0)
    stepsize = args.stepsize_num / args.nz

    #print(dat['X_train'].shape)
    #print(dat['Y_train'].shape)

    #print(X_training.shape)
    #asdfsadfsfs

    #print(dat['Y_train'])
    #asdfsads

    # use: X_training
    # also use: dat['Y_train']

    losses_NIKlosses = []

    x = X_training
    y = dat['Y_train'].to(device)

    netG.eval()

    for param in netG.parameters():
        param.requires_grad = False

    #netG.eval()

    #for param in netG.parameters():
    #    param.requires_grad = False

    #print(x.shape)
    #print(y.shape)

    #asdfasfsfs

    for itr in range(1, 1 + 1):
        runningLoss_NIKrunningLoss = 0.0

        #for i, (x, y) in enumerate(X_training):
        for i in range(len(X_training)):
            # print(x.shape)
            # print(y.shape)

            # print(y)

            x = x.to(device)

            # args.batchsize = 1024
            # args.batchsize = 16384

            # args.batchsize = 1024
            #args.batchSize = 2048

            #args.batchSize = 2048
            #args.batchSize = 150

            #args.batchSize = 2048
            args.batchSize = 2*2048

            # args.batchsize = 1024
            # ggenFGen2 = torch.randn([args.batchsize, nrand], device=device, requires_grad=True)

            # genFGen2 = genGen.forward(ggenFGen2)
            # genFGen2 = genGen.forward(ggenFGen2)

            # ggenFGen2 = torch.randn([args.batchsize, nrand], device=device, requires_grad=True)
            # genFGen2 = genGen.forward(ggenFGen2)

            # ggenFGen2 = torch.randn([args.batchsize, nrand], device=device)
            # genFGen2 = genGen.forward(ggenFGen2)

            with torch.no_grad():
                ggenFGen2 = torch.randn([args.batchSize, 100, 1, 1], device=device)
                genFGen2 = netG.forward(ggenFGen2)

            # print(x.shape)
            # print(y.shape)

            # print(genFGen2.shape)
            # print(args.batchsize)

            # for i21 in range(len(y)):
            #    if y[i21] == 0 and i21 == 0:
            #        y[i21] = y[i21+1]
            #        x[i21, :, :, :] = x[i21+1, :, :, :]
            #    elif y[i21] == 0:
            #        y[i21] = y[i21 - 1]
            #        x[i21, :, :, :] = x[i21 - 1, :, :, :]

            # y2 = []
            x2 = []
            for i21 in range(len(y)):
                if y[i21] == 1:
                    # y2.append(y[i21])
                    x2.append(x[i21, :, :, :])

            x2 = torch.stack(x2)
            # y2 = torch.stack(y2)

            # y3 = []
            x3 = []
            for i21 in range(len(y)):
                if y[i21] == 2:
                    # y3.append(y[i21])
                    x3.append(x[i21, :, :, :])
            
            x3 = torch.stack(x3)
            # y3 = torch.stack(y3)

            # y4 = []
            x4 = []
            for i21 in range(len(y)):
                if y[i21] == 3:
                    # y4.append(y[i21])
                    x4.append(x[i21, :, :, :])

            x4 = torch.stack(x4)
            # y4 = torch.stack(y4)

            # print(x2.shape)
            # print(x3.shape)
            # print(x4.shape)

            # print(y2.shape)
            # print(y3.shape)
            # print(y4.shape)

            # y5 = []
            x5 = []
            for i21 in range(len(y)):
                if y[i21] == 4:
                    # y5.append(y[i21])
                    x5.append(x[i21, :, :, :])

            x5 = torch.stack(x5)
            # y5 = torch.stack(y5)

            # y6 = []
            x6 = []
            for i21 in range(len(y)):
                if y[i21] == 5:
                    # y6.append(y[i21])
                    x6.append(x[i21, :, :, :])

            x6 = torch.stack(x6)
            # y6 = torch.stack(y6)

            # y7 = []
            x7 = []
            for i21 in range(len(y)):
                if y[i21] == 6:
                    # y7.append(y[i21])
                    x7.append(x[i21, :, :, :])

            x7 = torch.stack(x7)
            # y7 = torch.stack(y7)

            # y8 = []
            x8 = []
            for i21 in range(len(y)):
                if y[i21] == 7:
                    # y8.append(y[i21])
                    x8.append(x[i21, :, :, :])

            x8 = torch.stack(x8)
            # y8 = torch.stack(y8)

            # y9 = []
            x9 = []
            for i21 in range(len(y)):
                if y[i21] == 8:
                    # y9.append(y[i21])
                    x9.append(x[i21, :, :, :])

            x9 = torch.stack(x9)
            # y9 = torch.stack(y9)

            # y99 = []
            x99 = []
            for i21 in range(len(y)):
                if y[i21] == 9:
                    # y99.append(y[i21])
                    x99.append(x[i21, :, :, :])

            x99 = torch.stack(x99)
            # y99 = torch.stack(y99)

            x999 = []
            for i21 in range(len(y)):
                if y[i21] == 0:
                    x999.append(x[i21, :, :, :])
            x999 = torch.stack(x999)

            # print(x9.shape)
            # print(x99.shape)
            # print(genFGen2.shape)

            #print(x999.shape)
            #asdfasdfs

            genFGen2 = genFGen2.view(-1, 64 * 64)

            x9 = x9.view(-1, 64 * 64)
            x99 = x99.view(-1, 64 * 64)

            # print(x9.shape)
            # print(x99.shape)
            # print(genFGen2.shape)

            # x99 = x99.view(-1, 64 * 64)
            x999 = x999.view(-1, 64 * 64)

            x8 = x8.view(-1, 64 * 64)
            x7 = x7.view(-1, 64 * 64)

            x6 = x6.view(-1, 64 * 64)
            x5 = x5.view(-1, 64 * 64)

            x4 = x4.view(-1, 64 * 64)
            #x3 = x3.view(-1, 64 * 64)

            #x3 = x3.view(-1, 64 * 64)

            #x3 = x3.view(-1, 64 * 64)
            #x3 = x3.view(-1, 64 * 64)

            x3 = x3.view(-1, 64 * 64)

            x2 = x2.view(-1, 64 * 64)
            # x8 = x8.view(-1, 64 * 64)

            # print(args.batchsize)
            # print(genFGen2.shape)

            with torch.no_grad():

                # second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
                # second_term_loss32 = torch.empty(args.batchsize, device=device, requires_grad=False)
                second_term_loss32 = torch.empty(args.batchSize, device=device)
                # for i in range(args.batch_size):
                for i in range(args.batchSize):
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

                    # print(i)

                    # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2
                    # second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2

                    # second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2

                    # second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2
                    # second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2

                    # second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2
                    # second_term_loss22 = torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2

                    # second_term_losss22 = torch.sqrt((genFGen2[i, :] - x9).pow(2).sum(1)) ** 2
                    # second_term_lossss22 = torch.sqrt((genFGen2[i, :] - x8).pow(2).sum(1)) ** 2

                    # second_term_losssss22 = torch.sqrt((genFGen2[i, :] - x7).pow(2).sum(1)) ** 2
                    # second_term_lossssss22 = torch.sqrt((genFGen2[i, :] - x6).pow(2).sum(1)) ** 2

                    # second_term_losssssss22 = torch.sqrt((genFGen2[i, :] - x5).pow(2).sum(1)) ** 2
                    # second_term_lossssssss22 = torch.sqrt((genFGen2[i, :] - x4).pow(2).sum(1)) ** 2

                    # second_term_losssssssss22 = torch.sqrt((genFGen2[i, :] - x3).pow(2).sum(1)) ** 2
                    # second_term_lossssssssss22 = torch.sqrt((genFGen2[i, :] - x2).pow(2).sum(1)) ** 2

                    # print(x99.shape)
                    # print(genFGen2[i, :].shape)

                    secondSecSec_term_loss32 = torch.empty(10, device=device)
                    # secondSecSec_term_loss32[8] = torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2

                    # secondSecSec_term_loss32[8] = torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2
                    # secondSecSecSec_term_loss32 = torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2

                    # secondSecSec_term_loss32[8] = torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2

                    # secondSecSec_term_loss32[8] = torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2
                    #secondSecSec_term_loss32[8] = torch.min(torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2)

                    #secondSecSec_term_loss32[7] = torch.min(torch.sqrt((genFGen2[i, :] - x9).pow(2).sum(1)) ** 2)
                    #secondSecSec_term_loss32[6] = torch.min(torch.sqrt((genFGen2[i, :] - x8).pow(2).sum(1)) ** 2)

                    #secondSecSec_term_loss32[5] = torch.min(torch.sqrt((genFGen2[i, :] - x7).pow(2).sum(1)) ** 2)
                    #secondSecSec_term_loss32[4] = torch.min(torch.sqrt((genFGen2[i, :] - x6).pow(2).sum(1)) ** 2)

                    #secondSecSec_term_loss32[3] = torch.min(torch.sqrt((genFGen2[i, :] - x5).pow(2).sum(1)) ** 2)
                    #secondSecSec_term_loss32[2] = torch.min(torch.sqrt((genFGen2[i, :] - x4).pow(2).sum(1)) ** 2)

                    #secondSecSec_term_loss32[1] = torch.min(torch.sqrt((genFGen2[i, :] - x3).pow(2).sum(1)) ** 2)
                    #secondSecSec_term_loss32[0] = torch.min(torch.sqrt((genFGen2[i, :] - x2).pow(2).sum(1)) ** 2)

                    # print(secondSecSec_term_loss32)
                    # print(torch.min(torch.sqrt((genFGen2[i, :] - x999).pow(2).sum(1)) ** 2))

                    # use: x999
                    secondSecSec_term_loss32[0] = torch.min(torch.sqrt((genFGen2[i, :] - x999).pow(2).sum(1)) ** 2)

                    secondSecSec_term_loss32[1] = torch.min(torch.sqrt((genFGen2[i, :] - x2).pow(2).sum(1)) ** 2)
                    secondSecSec_term_loss32[2] = torch.min(torch.sqrt((genFGen2[i, :] - x3).pow(2).sum(1)) ** 2)

                    secondSecSec_term_loss32[3] = torch.min(torch.sqrt((genFGen2[i, :] - x4).pow(2).sum(1)) ** 2)
                    secondSecSec_term_loss32[4] = torch.min(torch.sqrt((genFGen2[i, :] - x5).pow(2).sum(1)) ** 2)

                    secondSecSec_term_loss32[5] = torch.min(torch.sqrt((genFGen2[i, :] - x6).pow(2).sum(1)) ** 2)
                    secondSecSec_term_loss32[6] = torch.min(torch.sqrt((genFGen2[i, :] - x7).pow(2).sum(1)) ** 2)

                    secondSecSec_term_loss32[7] = torch.min(torch.sqrt((genFGen2[i, :] - x8).pow(2).sum(1)) ** 2)
                    secondSecSec_term_loss32[8] = torch.min(torch.sqrt((genFGen2[i, :] - x9).pow(2).sum(1)) ** 2)

                    #secondSecSec_term_loss32[8] = torch.min(torch.sqrt((genFGen2[i, :] - x9).pow(2).sum(1)) ** 2)
                    secondSecSec_term_loss32[9] = torch.min(torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2)

                    # asdfasdfs

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
                    # second_term_loss32[i] = torch.min(second_term_loss22)

                    # second_term_loss32[i] = torch.min(second_term_loss22)
                    # second_term_loss32[i] = torch.argmin(secondSecSec_term_loss32)

                    # second_term_loss32[i] = torch.argmin(secondSecSec_term_loss32)
                    second_term_loss32[i] = torch.argmin(secondSecSec_term_loss32)

                    # second_term_loss32[i] = torch.min(second_term_loss22)

                    # second_term_loss32[i] = torch.min(second_term_loss22)
                    # second_term_loss32[i] = torch.min(second_term_loss22)
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
                # second_term_loss2 = torch.mean(second_term_loss32)

                # second_term_loss2 = torch.mean(second_term_loss32)

                # second_term_loss2 = torch.mean(second_term_loss32)
                # second_term_loss2 = torch.mean(second_term_loss32)

                # print(second_term_loss2)
                # asdfasfd

                # second_term_loss2.retain_grad()

                # second_term_loss2.retain_grad()
                # second_term_loss2.retain_grad()

                # (?)
                # second_term_loss2.retain_grad()
                # (?)

            # print(second_term_loss2)

            # tensor(89.3141, device='cuda:0')
            # print(second_term_loss2)

            # tensor(89.3141, device='cuda:0')
            # 0,1: tensor(89.3141, device='cuda:0')

            # 0,1: tensor(89.3141, device='cuda:0')
            # 0,2: tensor(63.0707, device='cuda:0')

            # 0,3: tensor(65.5907, device='cuda:0')
            # 0,4: tensor(74.6557, device='cuda:0')

            # 0,5: tensor(58.6006, device='cuda:0')
            # 0,6: tensor(57.5523, device='cuda:0')

            # 0,7: tensor(70.9559, device='cuda:0')
            # 0,8: tensor(64.4004, device='cuda:0')

            # 0,8: tensor(64.4004, device='cuda:0')
            # 0,9: tensor(62.5445, device='cuda:0')

            # print(second_term_loss2)

            # print(second_term_loss2)
            # print(second_term_loss2)

            # print(second_term_loss2)
            # print(second_term_loss32)

            import matplotlib.pyplot as plt
            # plt.plot(second_term_loss32)

            plt.plot(second_term_loss32.cpu())
            plt.savefig('saveSaSaSaSaveStore_second_term_loss32.png', bbox_inches='tight')

            counterFor0 = 0
            counterFor1 = 0
            counterFor2 = 0
            counterFor3 = 0
            counterFor4 = 0
            counterFor5 = 0
            counterFor6 = 0
            counterFor7 = 0
            counterFor8 = 0
            counterFor9 = 0
            for ii_loop21 in range(len(second_term_loss32)):
                if second_term_loss32[ii_loop21]==0:
                    counterFor0 += 1
                elif second_term_loss32[ii_loop21]==1:
                    counterFor1 += 1
                elif second_term_loss32[ii_loop21]==2:
                    counterFor2 += 1
                elif second_term_loss32[ii_loop21]==3:
                    counterFor3 += 1
                elif second_term_loss32[ii_loop21]==4:
                    counterFor4 += 1
                elif second_term_loss32[ii_loop21]==5:
                    counterFor5 += 1
                elif second_term_loss32[ii_loop21]==6:
                    counterFor6 += 1
                elif second_term_loss32[ii_loop21]==7:
                    counterFor7 += 1
                elif second_term_loss32[ii_loop21]==8:
                    counterFor8 += 1
                elif second_term_loss32[ii_loop21]==9:
                    counterFor9 += 1

            plt.figure()
            plt.plot(
                [counterFor0, counterFor1, counterFor2, counterFor3, counterFor4, counterFor5, counterFor6, counterFor7,
                 counterFor8, counterFor9])
            plt.savefig('saveSaSaveSaSaveSaSaveSaSaSaveStore_second_term_loss32.png', bbox_inches='tight')
            plt.savefig('NumberOfOccOccurences_vs_ClassesClusters.png', bbox_inches='tight')

            plt.figure()
            plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [counterFor0, counterFor1, counterFor2, counterFor3, counterFor4, counterFor5, counterFor6, counterFor7,
                 counterFor8, counterFor9], '--bo', linewidth=2, markersize=12)
            plt.ylabel('Number of modes')
            plt.xlabel('Modes')
            plt.savefig('NuNumberOfOccurences_vs_ClassesClusters.png', bbox_inches='tight')

            plt.figure()
            plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                     [counterFor0 / (
                                 counterFor0 + counterFor1 + counterFor2 + counterFor3 + counterFor4 + counterFor5 + counterFor6 + counterFor7 + counterFor8 + counterFor9),
                      counterFor1 / (
                                  counterFor0 + counterFor1 + counterFor2 + counterFor3 + counterFor4 + counterFor5 + counterFor6 + counterFor7 + counterFor8 + counterFor9),
                      counterFor2 / (
                                  counterFor0 + counterFor1 + counterFor2 + counterFor3 + counterFor4 + counterFor5 + counterFor6 + counterFor7 + counterFor8 + counterFor9),
                      counterFor3 / (
                                  counterFor0 + counterFor1 + counterFor2 + counterFor3 + counterFor4 + counterFor5 + counterFor6 + counterFor7 + counterFor8 + counterFor9),
                      counterFor4 / (
                                  counterFor0 + counterFor1 + counterFor2 + counterFor3 + counterFor4 + counterFor5 + counterFor6 + counterFor7 + counterFor8 + counterFor9),
                      counterFor5 / (
                                  counterFor0 + counterFor1 + counterFor2 + counterFor3 + counterFor4 + counterFor5 + counterFor6 + counterFor7 + counterFor8 + counterFor9),
                      counterFor6 / (
                                  counterFor0 + counterFor1 + counterFor2 + counterFor3 + counterFor4 + counterFor5 + counterFor6 + counterFor7 + counterFor8 + counterFor9),
                      counterFor7 / (
                                  counterFor0 + counterFor1 + counterFor2 + counterFor3 + counterFor4 + counterFor5 + counterFor6 + counterFor7 + counterFor8 + counterFor9),
                      counterFor8 / (
                                  counterFor0 + counterFor1 + counterFor2 + counterFor3 + counterFor4 + counterFor5 + counterFor6 + counterFor7 + counterFor8 + counterFor9),
                      counterFor9 / (
                                  counterFor0 + counterFor1 + counterFor2 + counterFor3 + counterFor4 + counterFor5 + counterFor6 + counterFor7 + counterFor8 + counterFor9)],
                     '--bo', linewidth=2, markersize=12)
            #plt.ylabel('Number of modes')
            plt.ylabel('Probability')
            plt.xlabel('Modes')
            plt.savefig('NumNumNumNumberOfOccurences_vs_ClassesClusters.png', bbox_inches='tight')

            asdfkfs
    
    asdfasdfasdf



    bsz = args.batchSize
    for epoch in range(1, args.epochs+1):
        for i in range(0, len(X_training), bsz): 
            sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)

            netD.zero_grad()
            stop = min(bsz, len(X_training[i:]))
            real_cpu = X_training[i:i+stop].to(device)

            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            noise_eta = torch.randn_like(real_cpu)
            noised_data = real_cpu + sigma_x.detach() * noise_eta
            out_real = netD(noised_data)
            errD_real = criterion(out_real, label)
            errD_real.backward()
            D_x = out_real.mean().item()

            # train with fake
            
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            mu_fake = netG(noise) 
            fake = mu_fake + sigma_x * noise_eta
            label.fill_(fake_label)
            out_fake = netD(fake.detach())
            errD_fake = criterion(out_fake, label)
            errD_fake.backward()
            D_G_z1 = out_fake.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # update G network: maximize log(D(G(z)))

            netG.zero_grad()
            sigma_optimizer.zero_grad()

            label.fill_(real_label)  
            gen_input = torch.randn(batch_size, args.nz, 1, 1, device=device)
            out = netG(gen_input)
            noise_eta = torch.randn_like(out)
            g_fake_data = out + noise_eta * sigma_x

            dg_fake_decision = netD(g_fake_data)
            g_error_gan = criterion(dg_fake_decision, label) 
            D_G_z2 = dg_fake_decision.mean().item()

            if args.lambda_ == 0:
                g_error_gan.backward()
                optimizerG.step() 
                sigma_optimizer.step()

            else:
                hmc_samples, acceptRate, stepsize = hmc.get_samples(
                    netG, g_fake_data.detach(), gen_input.clone(), sigma_x.detach(), args.burn_in, 
                        args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt, 
                            args.hmc_learning_rate, args.hmc_opt_accept)
                
                bsz, d = hmc_samples.size()
                mean_output = netG(hmc_samples.view(bsz, d, 1, 1).to(device))
                bsz = g_fake_data.size(0)

                mean_output_summed = torch.zeros_like(g_fake_data)
                for cnt in range(args.num_samples_posterior):
                    mean_output_summed = mean_output_summed + mean_output[cnt*bsz:(cnt+1)*bsz]
                mean_output_summed = mean_output_summed / args.num_samples_posterior  

                c = ((g_fake_data - mean_output_summed) / sigma_x**2).detach()
                #g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()

                #print(torch.mul(c, out + sigma_x * noise_eta).mean())
                #asdfasdf

                #g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()
                g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()

                g_error = g_error_gan - args.lambda_ * g_error_entropy
                g_error.backward()
                optimizerG.step() 
                sigma_optimizer.step()

            if args.restrict_sigma:
                log_sigma.data.clamp_(min=logsigma_min, max=logsigma_max)

            ## log performance
            if i % args.log == 0:
                print('Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
                        % (epoch, args.epochs, i, len(X_training), errD.data, g_error_gan.data, D_x, D_G_z1, D_G_z2))

        print('*'*100)
        print('End of epoch {}'.format(epoch))
        print('sigma min: {} .. sigma max: {}'.format(torch.min(sigma_x), torch.max(sigma_x)))
        print('*'*100)
        if args.lambda_ > 0:
            print('| MCMC diagnostics ====> | stepsize: {} | min ar: {} | mean ar: {} | max ar: {} |'.format(
                        stepsize, acceptRate.min().item(), acceptRate.mean().item(), acceptRate.max().item()))

        if epoch % args.save_imgs_every == 0:
            fake = netG(fixed_noise).detach()
            vutils.save_image(fake, '%s/presgan_%s_fake_epoch_%03d.png' % (args.results_folder, args.dataset, epoch), normalize=True, nrow=20) 

        if epoch % args.save_ckpt_every == 0:
            torch.save(netG.state_dict(), os.path.join(args.results_folder, 'netG_presgan_%s_epoch_%s.pth'%(args.dataset, epoch)))
            torch.save(log_sigma, os.path.join(args.results_folder, 'log_sigma_%s_%s.pth'%(args.dataset, epoch)))
            
