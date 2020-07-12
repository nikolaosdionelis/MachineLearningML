import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.utils as vutils

import seaborn as sns
import os 
import pickle 
import math 
import csv

import utils 
import hmc 

from torch.distributions.normal import Normal
from tensorboardX import SummaryWriter

torch.manual_seed(123)
real_label = 1
fake_label = 0
criterion = nn.BCELoss()
criterion_mse = nn.MSELoss()

def dcgan(dat, netG, netD, args):
    device = args.device
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        criterion.cuda()
        criterion_mse.cuda()
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
    writer = SummaryWriter(log_dir='tensorboard'+args.dataset)
    device = args.device
    if torch.cuda.is_available():
        print("cuda")
        netG.cuda()
        netD.cuda()
        criterion.cuda()
        criterion_mse.cuda()
    X_training = dat['X_train'].to(device) # [60000, 1, 64, 64]
    fixed_noise = torch.randn(args.num_gen_images, args.nz, 1, 1, device=device)
    torch.manual_seed(123)
    # NEW
    Y_training = dat['Y_train'].to(device)
    # NUM_CLASS = 10
    NUM_CLASS = args.n_classes

    optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, 0.999)) 
    sigma_optimizer = optim.Adam([log_sigma], lr=args.sigma_lr, betas=(args.beta1, 0.999))
    if args.restrict_sigma:
        logsigma_min = math.log(math.exp(args.sigma_min) - 1.0)
        logsigma_max = math.log(math.exp(args.sigma_max) - 1.0)
    #stepsize = args.stepsize_num / args.nz
    
    #bsz = args.batchSize

    #print(X_training.shape)

    #print(X_training.shape)
    #print(X_training.shape)

    #asdfasdfcscv

    stepsize = args.stepsize_num / args.nz

    Y_forY_training = dat['Y_train'].to(device)

    bsz = args.batchSize

    for epoch in range(1, args.epochs + 1):
        for i in range(0, len(X_training), bsz):
            # sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)

            # sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)
            # sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)

            stop = min(bsz, len(X_training[i:]))
            real_cpu = X_training[i:i + stop].to(device)
            y_real_cpu = Y_forY_training[i:i + stop].to(device)

            for sadgfjasj in range(len(y_real_cpu)):
               if (sadgfjasj>0) and (y_real_cpu[sadgfjasj]==2):
                   y_real_cpu[sadgfjasj] = y_real_cpu[sadgfjasj-1]
                   real_cpu[sadgfjasj, :] = real_cpu[sadgfjasj-1, :]
               elif (sadgfjasj==0) and (y_real_cpu[sadgfjasj]==2):
                   y_real_cpu[sadgfjasj] = y_real_cpu[sadgfjasj+1]
                   real_cpu[sadgfjasj, :] = real_cpu[sadgfjasj+1, :]

            X_training[i:i + stop] = real_cpu
            Y_forY_training[i:i + stop] = y_real_cpu

            #sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)

            #sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)
            #sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)

            #sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)
            sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)

            netD.zero_grad()
            stop = min(bsz, len(X_training[i:]))
            real_cpu = X_training[i:i + stop].to(device)

            '''
            for epoch in range(1, args.epochs+1):
                for i in range(0, len(X_training), bsz): # bsz = 64
        
                    sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)
        
                    netD.zero_grad()
                    stop = min(bsz, len(X_training[i:]))
                    real_cpu = X_training[i:i+stop].to(device) # [64, 1, 64, 64]
            '''

            batch_size = real_cpu.size(0)
            labelv = torch.full((batch_size,), real_label).to(device)

            # train discriminator on real (noised) data and real labels
            y_labels = Y_training[i:i+stop].to(device)
            y_one_hot = torch.FloatTensor(batch_size, NUM_CLASS).to(device) # adding cuda here 
            # print(batch_size, bsz, y_labels.size())
            y_one_hot = y_one_hot.zero_().scatter_(1, y_labels.view(batch_size, 1), 1).to(device)

            noise_eta = torch.randn_like(real_cpu).to(device)
            noised_data = real_cpu + sigma_x.detach() * noise_eta
            out_real = netD(noised_data, y_one_hot) #, y_one_hot_labels
            errD_real = criterion(out_real, labelv)
            errD_real.backward()
            D_x = out_real.mean().item()

            # make generator output image from random labels; make discriminator classify
            rand_y_one_hot = torch.FloatTensor(batch_size, NUM_CLASS).zero_().to(device) # adding cuda here 
            rand_y_one_hot.scatter_(1, torch.randint(0, NUM_CLASS, size=(batch_size,1), device = device), 1) # #rand_y_one_hot.scatter_(1, torch.from_numpy(np.random.randint(0, 10, size=(bsz,1))), 1)

            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            mu_fake = netG(noise, rand_y_one_hot) 
            fake = mu_fake + sigma_x * noise_eta
            labelv = labelv.fill_(fake_label).to(device)
            out_fake = netD(fake.detach(), rand_y_one_hot)
            errD_fake = criterion(out_fake, labelv)
            errD_fake.backward()
            D_G_z1 = out_fake.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # update G network: maximize log(D(G(z)))

            netG.zero_grad()
            sigma_optimizer.zero_grad()

            rand_y_one_hot = torch.FloatTensor(batch_size, NUM_CLASS).zero_().to(device)
            rand_y_one_hot = rand_y_one_hot.scatter_(1, torch.randint(0, NUM_CLASS, size=(batch_size,1), device = device), 1).to(device)
            labelv = labelv.fill_(real_label).to(device)  
            gen_input = torch.randn(batch_size, args.nz, 1, 1, device=device)
            out = netG(gen_input, rand_y_one_hot) # add rand y labels
            noise_eta = torch.randn_like(out)
            g_fake_data = out + noise_eta * sigma_x

            dg_fake_decision = netD(g_fake_data, rand_y_one_hot) # add rand y labels
            g_error_gan = criterion(dg_fake_decision, labelv) 
            D_G_z2 = dg_fake_decision.mean().item()

#             # TO TEST WITHOUT ENTROPY, SET: 
#             if epoch < 10 and args.lambda_ != 0 and args.dataset != 'mnist': 
#                 args.lambda_ = 0
#             elif epoch < 20 and args.lambda_ != 0 and args.dataset != 'mnist': 
#                 args.lambda_ = 0.0001
#             elif args.lambda_ != 0 and args.dataset != 'mnist':
#                 args.lambda_ = 0.0002

            if args.lambda_ == 0:
                g_error_gan.backward()
                optimizerG.step() 
                sigma_optimizer.step()

            else:
                # added y_tilde param (rand_y_one_hot)
                hmc_samples, hmc_labels, acceptRate, stepsize = hmc.get_samples(
                    netG, g_fake_data.detach(), rand_y_one_hot.detach(), gen_input.clone(), sigma_x.detach(), args.burn_in, 
                        args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt, 
                            args.hmc_learning_rate, args.hmc_opt_accept)
                
                bsz, d = hmc_samples.size()
                hmc_samples = hmc_samples.view(bsz, d, 1, 1).to(device)
                hmc_labels = hmc_labels.to(device)
                mean_output = netG(hmc_samples, hmc_labels)
                bsz = g_fake_data.size(0)

                mean_output_summed = torch.zeros_like(g_fake_data).to(device)
                for cnt in range(args.num_samples_posterior):
                    mean_output_summed = mean_output_summed + mean_output[cnt*bsz:(cnt+1)*bsz]
                mean_output_summed = mean_output_summed / args.num_samples_posterior  

                c = ((g_fake_data - mean_output_summed) / sigma_x**2).detach()
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
                with open('%s/log.csv' % args.results_folder, 'a') as f:
                    r = csv.writer(f)
                    # Loss_G, Loss_D, D(x), D(G(z))
                    r.writerow([g_error_gan.data, errD.data, D_x, D_G_z2])


            if i % (2*args.log) == 0:
                t_iter = (epoch*len(X_training)+i)/bsz
                writer.add_scalar('Loss_G', g_error_gan.data, t_iter)
                writer.add_scalar('Loss_D', errD.data, t_iter)
                writer.add_scalar('D(x)', D_x, t_iter)
                writer.add_scalar('D(G(z))', D_G_z2, t_iter)

        print('*'*100)
        print('End of epoch {}'.format(epoch))
        print('sigma min: {} .. sigma max: {}'.format(torch.min(sigma_x), torch.max(sigma_x)))
        print('*'*100)
        if args.lambda_ > 0:
            print('| MCMC diagnostics ====> | stepsize: {} | min ar: {} | mean ar: {} | max ar: {} |'.format(
                        stepsize, acceptRate.min().item(), acceptRate.mean().item(), acceptRate.max().item()))

        if epoch % args.save_imgs_every == 0:
            rand_y_one_hot = torch.FloatTensor(args.num_gen_images, NUM_CLASS).zero_().to(device) # adding cuda here
            rand_y_one_hot = rand_y_one_hot.scatter_(1, torch.randint(0, NUM_CLASS, size=(args.num_gen_images,1), device = device), 1).to(device) # #rand_y_one_hot.scatter_(1, torch.from_numpy(np.random.randint(0, 10, size=(bsz,1))), 1)
            fake = netG(fixed_noise, rand_y_one_hot).detach()

            vutils.save_image(fake, '%s/presgan_%s_fake_epoch_%03d.png' % (args.results_folder, args.dataset, epoch), normalize=True, nrow=20) 

        if epoch % args.save_ckpt_every == 0:
            torch.save(netG.state_dict(), os.path.join(args.results_folder, 'netG_presgan_%s_epoch_%s.pth'%(args.dataset, epoch)))
            torch.save(log_sigma, os.path.join(args.results_folder, 'log_sigma_%s_%s.pth'%(args.dataset, epoch)))
            torch.save(netD.state_dict(), os.path.join(args.results_folder, 'netD_presgan_%s_epoch_%s.pth'%(args.dataset, epoch)))
