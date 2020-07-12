import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.utils as vutils

import seaborn as sns
import os 
import pickle 
import math 

import utils 
import hmc 

import data
import datasets

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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


def use_loss_fn2(first_term_loss, genFGen2, args, model, genFGen3, xData):
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

    #print(genFGen2.shape)
    #print(xData.shape)

    #print(compute_loss2(genFGen2, model))
    #print(compute_loss2(xData, model))

    #print(compute_loss(xData, model))
    #print(compute_loss(xData, model).item())

    # (tensor(0.9740, device='cuda:0', grad_fn=<DivBackward0>), tensor([0.], device='cuda:0'),
    # tensor(-1139.7253, device='cuda:0'), tensor(4957.8486, device='cuda:0'))

    #print(computeLoss(genFGen2, model))
    #print(computeLoss(xData, model))

    #first_term_loss = compute_loss2(genFGen2, model)
    #first_term_loss = compute_loss2(genFGen2, model)

    #first_term_loss = compute_loss2(genFGen2, model)

    #first_term_loss = compute_loss2(genFGen2, model)
    #first_term_loss = computeLoss(genFGen2, model)

    #print(genFGen2.shape)
    #print(first_term_loss)

    #first_term_loss.retain_grad()

    #first_term_loss.retain_grad()
    #first_term_loss.retain_grad()

    # (?)
    #first_term_loss.retain_grad()
    # (?)

    #print(first_term_loss)
    #print('')

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

    #xData = xData.view(-1, 28 * 28)
    xData = xData.view(-1, 64 * 64)

    #xData = xData.view(-1, 28*28)
    #genFGen2 = genFGen2.view(-1, 28*28)

    #genFGen2 = genFGen2.view(-1, 28 * 28)
    genFGen2 = genFGen2.view(-1, 64 * 64)

    #genFGen2 = genFGen2.view(-1, 28*28)
    #genFGen3 = genFGen3.view(-1, 28*28)

    #genFGen3 = genFGen3.view(-1, 28 * 28)
    genFGen3 = genFGen3.squeeze()

    #print(genFGen3.shape)
    #asdfasdf

    #print(xData.shape)
    #print(genFGen2.shape)

    #print(genFGen3.shape)
    #asdfasdf

    device = args.device

    #print(device)
    #adfasdfs

    #genFGen3 = genFGen3.view(-1, 28 * 28)
    #genFGen3 = genFGen3.view(-1, 64 * 64)

    #xData = torch.transpose(xData, 0, 1)
    #genFGen2 = torch.transpose(genFGen2, 0, 1)

    #genFGen2 = torch.transpose(genFGen2, 0, 1)
    #genFGen3 = torch.transpose(genFGen3, 0, 1)

    #print(genFGen2.shape)
    #print(xData.shape)
    #print(genFGen3.shape)

    #print(genFGen2.shape)
    #print(xData.shape)

    #print(genFGen3.shape)
    #print(args.batchSize)

    #second_term_loss32 = torch.empty(args.batchSize, device=device, requires_grad=False)
    #for i in range(args.batchSize):
    #    second_term_loss32[i] = torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)
    #second_term_loss2 = torch.mean(second_term_loss32)

    #xData = xData[:15000,:]
    #xData.requires_grad = True

    #print(xData.shape)
    #asdfasfs

    #print(xData.shape)
    #adfasdfs

    #second_term_loss2 = torch.empty(1, device=device, requires_grad=False)
    second_term_loss2 = torch.zeros(1, device=device, requires_grad=False)
    #print(second_term_loss2)
    #asdfadsfs
    for i in range(args.batchSize):
        # print(i)

        # print((torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2))
        # print((torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2).shape)

        # asdfasdf
        # second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)

        # second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)
        # second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)

        #if i < 6:

        #if i < 6:
        #if i < 5:
        #    second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)
        #else:
        #    second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :].detach() - xData).pow(2).sum(1)) ** 2)

        #print(i)
        #second_term_loss2 += torch.min(torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_() ** 2)

        second_term_loss2 += torch.min(torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_() ** 2)

        #if i < 7:
        #    second_term_loss2 += torch.min(torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_() ** 2)
        #else:
        #    second_term_loss2 += torch.min(torch.norm(genFGen2[i, :].detach() - xData, p=None, dim=1).requires_grad_() ** 2)

        # second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)
        # second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :].detach() - xData).pow(2).sum(1)) ** 2)
        # try:
        #    second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)
        #    #break
        # except MemoryError:
        #    second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :].detach() - xData).pow(2).sum(1)) ** 2)
    #second_term_loss2 /= args.batchSize

    second_term_loss2 /= args.batchSize
    #second_term_loss2 = max(second_term_loss2, 1e-8)

    #print(second_term_loss2)
    #print(second_term_loss2.requires_grad)

    #asdfasdfs

    #second_term_loss2.backward()

    second_term_loss2 = second_term_loss2.squeeze()
    #second_term_loss2 = abs(second_term_loss2.squeeze())

    #second_term_loss2 = max(second_term_loss2, torch.tensor(1e-8))

    #if torch.isnan(second_term_loss2).any():
    #    second_term_loss2

    #print(torch.isnan(second_term_loss2).any())
    #asdfasdfs

    #print(second_term_loss2)
    #print(second_term_loss2.requires_grad)

    #print(second_term_loss2)
    #print(second_term_loss2.requires_grad)

    #asdfas

    '''
    second_term_loss2 = torch.empty(1, device=device, requires_grad=False)
    for i in range(args.batchSize):
        #print(i)

        #print((torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2))
        #print((torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2).shape)

        #asdfasdf
        #second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)

        #second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)
        #second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)

        if i<6:
            second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)
        else:
            second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :].detach() - xData).pow(2).sum(1)) ** 2)

        #second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)
        #second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :].detach() - xData).pow(2).sum(1)) ** 2)
        #try:
        #    second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)
        #    #break
        #except MemoryError:
        #    second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :].detach() - xData).pow(2).sum(1)) ** 2)
    second_term_loss2 /= args.batchSize

    #second_term_loss2.backward()

    print(second_term_loss2)
    print(second_term_loss2.requires_grad)
    '''

    '''
    # second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    second_term_loss32 = torch.empty(args.batchSize, device=device, requires_grad=False)
    # for i in range(args.batch_size):
    for i in range(args.batchSize):
        """
        print(torch.mean(torch.sqrt((genFGen2[i, :] - xData).view(args.batchSize, -1).pow(2).sum(1))))
        print(torch.mean(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchSize, -1).pow(2).sum(1))))
        print(torch.mean(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1))))
        print('')

        print(torch.mean(torch.norm((genFGen2[i, :] - xData).view(args.batchSize, -1), p=None, dim=1)))
        print(torch.mean(torch.norm((genFGen2[i, :] - genFGen2).view(args.batchSize, -1), p=None, dim=1)))
        print(torch.mean(torch.norm((genFGen3[i, :] - genFGen3), p=None, dim=1)))
        print('')
        """

        # print(torch.mean(torch.sqrt((genFGen2[i, :] - xData).view(args.batchSize, -1).pow(2).sum(1))))
        # print(torch.mean(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchSize, -1).pow(2).sum(1))))
        # print(torch.mean(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1))))
        # print('')

        # print(torch.sqrt((genFGen2[i, :] - xData).view(args.batchSize, -1).pow(2).sum(1)))
        # print(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchSize, -1).pow(2).sum(1)))
        # print(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1)))
        # print('')

        # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p='fro', dim=1).requires_grad_()
        # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()
        # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()**2
        # second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1))**2
        # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_()**2

        # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2

        # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2
        # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).view(args.batchSize, -1).pow(2).sum(1)).requires_grad_() ** 2

        # second_term_loss22 = torch.sqrt(
        #    1e-17 + (genFGen2[i, :] - xData).view(args.batchSize, -1).pow(2).sum(1)).requires_grad_() ** 2

        # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()**2
        # second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2

        # tempVarVar21 = genFGen2[i, :] - xData
        # print(tempVarVar21.shape)

        # print(xData.shape)
        # asdfsadf

        # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2
        # second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2

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

        # print(i)

        # second_term_loss32[i] = torch.min(second_term_loss22)
        #second_term_loss32[i] = torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)

        #second_term_loss32[i] = torch.min(torch.sqrt((genFGen2[i, :].detach() - xData).pow(2).sum(1)) ** 2)

        if i<6:
            second_term_loss32[i] = torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)
        else:
            second_term_loss32[i] = torch.min(torch.sqrt((genFGen2[i, :].detach() - xData).pow(2).sum(1)) ** 2)

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
    second_term_loss2 = torch.mean(second_term_loss32)

    print(second_term_loss2)
    print(second_term_loss2.requires_grad)

    asdfasfs
    '''

    # print(second_term_loss2)
    # asdfasfd

    #second_term_loss32 = torch.empty(args.batchSize, device=device, requires_grad=False)
    #for i in range(args.batchSize):
    #    second_term_loss32[i] = torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)
    #second_term_loss2 = torch.mean(second_term_loss32)

    '''
    second_term_loss32 = torch.empty(args.batchSize, device=device, requires_grad=False)
    for i in range(args.batchSize):
        second_term_loss32[i] = torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)
    second_term_loss2 = torch.mean(second_term_loss32)
    '''

    '''
    #second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
    second_term_loss32 = torch.empty(args.batchSize, device=device, requires_grad=False)
    #for i in range(args.batch_size):
    for i in range(args.batchSize):
        """
        print(torch.mean(torch.sqrt((genFGen2[i, :] - xData).view(args.batchSize, -1).pow(2).sum(1))))
        print(torch.mean(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchSize, -1).pow(2).sum(1))))
        print(torch.mean(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1))))
        print('')

        print(torch.mean(torch.norm((genFGen2[i, :] - xData).view(args.batchSize, -1), p=None, dim=1)))
        print(torch.mean(torch.norm((genFGen2[i, :] - genFGen2).view(args.batchSize, -1), p=None, dim=1)))
        print(torch.mean(torch.norm((genFGen3[i, :] - genFGen3), p=None, dim=1)))
        print('')
        """

        #print(torch.mean(torch.sqrt((genFGen2[i, :] - xData).view(args.batchSize, -1).pow(2).sum(1))))
        #print(torch.mean(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchSize, -1).pow(2).sum(1))))
        #print(torch.mean(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1))))
        #print('')

        #print(torch.sqrt((genFGen2[i, :] - xData).view(args.batchSize, -1).pow(2).sum(1)))
        #print(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchSize, -1).pow(2).sum(1)))
        #print(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1)))
        #print('')

        #second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p='fro', dim=1).requires_grad_()
        #second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()
        #second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()**2
        #second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1))**2
        #second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_()**2

        #second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2

        #second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2
        #second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).view(args.batchSize, -1).pow(2).sum(1)).requires_grad_() ** 2

        #second_term_loss22 = torch.sqrt(
        #    1e-17 + (genFGen2[i, :] - xData).view(args.batchSize, -1).pow(2).sum(1)).requires_grad_() ** 2

        #second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()**2
        #second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2

        #tempVarVar21 = genFGen2[i, :] - xData
        #print(tempVarVar21.shape)

        #print(xData.shape)
        #asdfsadf

        #second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2
        #second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2

        # 61562.1641
        # 4.7732

        #print(genFGen2[i, :].shape)
        #print(xData.shape)

        #tempVarVar21 = genFGen2[i, :] - xData
        #print(tempVarVar21.shape)

        #print(second_term_loss22.shape)
        #adsfasfs

        #second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()
        #print(second_term_loss22.shape)
        #second_term_loss32[i] = torch.min(second_term_loss22)

        #print(i)

        #second_term_loss32[i] = torch.min(second_term_loss22)
        second_term_loss32[i] = torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)

        #second_term_loss32[i] = torch.min(second_term_loss22)
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

    #print(second_term_loss2)
    #asdfasfd
    '''

    #second_term_loss2.retain_grad()

    #second_term_loss2.retain_grad()
    #second_term_loss2.retain_grad()

    # (?)
    #second_term_loss2.retain_grad()
    # (?)

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
    third_term_loss32 = torch.empty(args.batchSize, device=device, requires_grad=False)
    #for i in range(args.batch_size):
    for i in range(args.batchSize):
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

        #print((genFGen2[i, :] - xData).view(args.batchSize,-1).pow(2).sum(1).shape)
        #print((genFGen2[i, :] - genFGen2).view(args.batchSize,-1).pow(2).sum(1).shape)
        #print('')

        #print(torch.norm((genFGen2[i, :] - xData).view(args.batchSize,-1), p=None, dim=1).shape)
        #print(torch.norm((genFGen2[i, :] - genFGen2).view(args.batchSize,-1), p=None, dim=1).shape)
        #print('')

        """
        print(torch.mean(torch.sqrt((genFGen2[i, :] - xData).view(args.batchSize, -1).pow(2).sum(1))))
        print(torch.mean(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchSize, -1).pow(2).sum(1))))
        print(torch.mean(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1))))
        print('')

        print(torch.mean(torch.norm((genFGen2[i, :] - xData).view(args.batchSize, -1), p=None, dim=1)))
        print(torch.mean(torch.norm((genFGen2[i, :] - genFGen2).view(args.batchSize, -1), p=None, dim=1)))
        print(torch.mean(torch.norm((genFGen3[i, :] - genFGen3), p=None, dim=1)))
        print('')
        """

        #print(torch.mean(torch.sqrt((genFGen2[i, :] - xData).view(args.batchSize, -1).pow(2).sum(1))))
        #print(torch.mean(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchSize, -1).pow(2).sum(1))))
        #print(torch.mean(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1))))
        #print('')

        #print(torch.sqrt((genFGen2[i, :] - xData).view(args.batchSize, -1).pow(2).sum(1)))
        #print(torch.sqrt((genFGen2[i, :] - genFGen2).view(args.batchSize, -1).pow(2).sum(1)))
        #print(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1)))
        #print('')

        #third_term_loss22 = (torch.norm(genFGen3[i, :] - genFGen3, p=None, dim=1).requires_grad_()) / (
        #            1.0e-17 + torch.norm(genFGen2[i, :] - genFGen2, p=None, dim=1).requires_grad_())
        #third_term_loss22 = (torch.sqrt(1e-17 + (genFGen3[i, :] - genFGen3).pow(2).sum(1)).requires_grad_()) / (
        #        1e-17 + torch.sqrt(1e-17 + (genFGen2[i, :] - genFGen2).pow(2).sum(1)).requires_grad_())

        #third_term_loss22 = (torch.sqrt(1e-17 + (genFGen3[i, :] - genFGen3).pow(2).sum(1)).requires_grad_()) / (
        #        1e-17 + torch.sqrt(1e-17 + (genFGen2[i, :] - genFGen2).pow(2).sum(1)).requires_grad_())

        #hbdafj = genFGen3[i, :] - genFGen3
        #print(hbdafj.shape)

        #adfa = genFGen2[i, :] - xData
        #print(adfa.shape)

        #third_term_loss22 = (torch.sqrt(1e-17 + (genFGen3[i, :] - genFGen3).pow(2).sum(1)).requires_grad_()) / (
        #        1e-17 + torch.sqrt(1e-17 + (genFGen2[i, :] - genFGen2).view(args.batchSize, -1).pow(2).sum(1)).requires_grad_())

        #third_term_loss22 = (torch.sqrt(1e-17 + (genFGen3[i, :] - genFGen3).pow(2).sum(1)).requires_grad_()) / (
        #       1e-17 + torch.sqrt(1e-17 + (genFGen2[i, :] - genFGen2).view(args.batchSize, -1).pow(2).sum(1)).requires_grad_())

        #third_term_loss22 = (torch.sqrt(1e-17 + (genFGen3[i, :] - genFGen3).pow(2).sum(1)).requires_grad_()) / (
        #            1e-17 + torch.sqrt(1e-17 + (genFGen2[i, :] - genFGen2).pow(2).sum(1)).requires_grad_())

        #third_term_loss22 = (torch.norm(genFGen3[i, :] - genFGen3, p=None, dim=1).requires_grad_()) / (
        #            1.0e-17 + torch.norm(genFGen2[i, :] - genFGen2, p=None, dim=1).requires_grad_())

        third_term_loss22 = (torch.sqrt(1e-17 + (genFGen3[i, :] - genFGen3).pow(2).sum(1)).requires_grad_()) / (
                   1e-17 + torch.sqrt(1e-17 + (genFGen2[i, :] - genFGen2).pow(2).sum(1)).requires_grad_())

        #print(third_term_loss22.shape)

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

    # (?)
    #third_term_loss12 = torch.zeros(1, device=device, requires_grad=True)
    # (?)

    #third_term_loss32 = torch.zeros(1, device=device, requires_grad=True)

    #third_term_loss32 = torch.zeros(1, device=device, requires_grad=True)
    #third_term_loss32 = torch.zeros(1, device=device, requires_grad=True)

    #print(third_term_loss12)
    #adfdfasc

    #third_term_loss12.retain_grad()

    #third_term_loss12.retain_grad()
    #third_term_loss12.retain_grad()

    # (?)
    #third_term_loss12.retain_grad()
    # (?)

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

    #print('')
    #print(first_term_loss.item())

    #print(second_term_loss2.item())
    #print(third_term_loss12.item())

    #print('')
    #print(first_term_loss.grad)

    #print(second_term_loss2.grad)
    #print(third_term_loss12.grad)

    #print('')

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
    #total_totTotalLoss = first_term_loss + 1.0 * second_term_loss2 + 0.1 * third_term_loss12

    #total_totTotalLoss = first_term_loss + 1.0 * second_term_loss2 + 0.1 * third_term_loss12

    #print(first_term_loss)
    #print(first_term_loss.requires_grad)

    #print(second_term_loss2)
    #print(second_term_loss2.requires_grad)

    #print(third_term_loss12)
    #print(third_term_loss12.requires_grad)

    #print(first_term_loss.requires_grad)
    #print(second_term_loss2.requires_grad)
    #print(third_term_loss12.requires_grad)

    #print(first_term_loss)
    #print(second_term_loss2)

    #print(third_term_loss12)
    #asdfasdf

    #first_term_loss = first_term_loss * 0.000001
    #second_term_loss2 = second_term_loss2.squeeze()

    #second_term_loss2 = second_term_loss2 * 0.001
    #second_term_loss2 = second_term_loss2 * 0.01

    #second_term_loss2 = second_term_loss2.squeeze()

    #first_term_loss *= 100.0
    #second_term_loss2 *= 0.0001

    #print(first_term_loss)
    #print(first_term_loss.requires_grad)

    #print(second_term_loss2)
    #print(second_term_loss2.requires_grad)

    #print(third_term_loss12)
    #print(third_term_loss12.requires_grad)

    #asdfszdf

    #total_totTotalLoss = first_term_loss + 1.0 * second_term_loss2 + 0.1 * third_term_loss12
    #total_totTotalLoss = first_term_loss + 0.3 * second_term_loss2 + 0.025 * third_term_loss12

    #total_totTotalLoss = first_term_loss + 0.001 * second_term_loss2 + third_term_loss12
    #total_totTotalLoss = first_term_loss + second_term_loss2 + third_term_loss12

    #total_totTotalLoss = first_term_loss + 100.0*second_term_loss2 + third_term_loss12
    #total_totTotalLoss = first_term_loss + 100.0*second_term_loss2 + 0.1*third_term_loss12

    #total_totTotalLoss = first_term_loss + 0.01 * second_term_loss2 + 0.1 * third_term_loss12
    #total_totTotalLoss = first_term_loss + 0.01 * second_term_loss2 + 0.01 * third_term_loss12

    #total_totTotalLoss = first_term_loss + 0.01 * second_term_loss2 + 0.01 * third_term_loss12

    #total_totTotalLoss = first_term_loss + 0.01 * second_term_loss2 + 0.01 * third_term_loss12
    #total_totTotalLoss = first_term_loss + 0.1 * second_term_loss2 + 0.01 * third_term_loss12

    #total_totTotalLoss = first_term_loss + 0.1 * second_term_loss2 + 0.01 * third_term_loss12
    total_totTotalLoss = first_term_loss + 0.01 * second_term_loss2 + 0.01 * third_term_loss12

    #print(total_totTotalLoss)
    #print(total_totTotalLoss.requires_grad)

    #asdfsadf

    #total_totTotalLoss.retain_grad()

    #total_totTotalLoss.retain_grad()
    #total_totTotalLoss.retain_grad()

    #return first_term_loss + second_term_loss2 + third_term_loss12, first_term_loss, second_term_loss2
    #return first_term_loss + second_term_loss2 + third_term_loss12, first_term_loss, second_term_loss2, third_term_loss12

    #return first_term_loss + second_term_loss2 + third_term_loss12, first_term_loss, second_term_loss2, third_term_loss12
    return total_totTotalLoss, first_term_loss, second_term_loss2, third_term_loss12


def presgan(dat, netG, netD, log_sigma, args, netG2):
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
    
    #bsz = args.batchSize
    bsz = len(X_training)

    netG2.eval()

    for param in netG2.parameters():
        param.requires_grad = False

    #print(X_training.shape)
    #print(len(range(0, len(X_training), bsz)))

    #asdfasdf

    #netG.eval()

    #for param in netG.parameters():
    #    param.requires_grad = False

    #netG2.eval()

    #for param in netG2.parameters():
    #    param.requires_grad = False

    loss_theLoss = torch.empty(args.epochs, device=device)
    loss_theLoss0 = torch.empty(args.epochs, device=device)

    loss_theLoss1 = torch.empty(args.epochs, device=device)
    loss_theLoss2 = torch.empty(args.epochs, device=device)

    loss_theLoss3 = torch.empty(args.epochs, device=device)

    #args.epochs = 1

    #args.epochs = 1
    #args.epochs = 1

    #print(X_training.shape)
    #print(np.shape(X_training))

    """
    print(X_training.shape)
    print(dat['X_train'].to(device).shape)

    #print(dat['X_train'].to(device).shape)
    print(dat['Y_train'].to(device).shape)

    print(dat['X_test'].to(device).shape)
    print(dat['Y_test'].to(device).shape)
    """

    #X_training = dat['X_test'].to(device)
    #X_training = dat['X_train'].to(device)

    #X_training = dat['X_test'].to(device)

    #X_training = dat['X_test'].to(device)
    #X_training = dat['X_test'].to(device)

    #X_training = dat['X_test'].to(device)
    #X_training = stack_mnist(data_dir, num_training_sample, num_test_sample, imageSize).to(device)

    #dat = data.load_data(args.dataset, args.dataroot, args.batchSize,
    #                     device=device, imgsize=args.imageSize, Ntrain=args.Ntrain, Ntest=args.Ntest)

    #bsz = 1

    #bsz = 1
    #bsz = 1

    #print(X_training)
    #print(X_training.shape)

    #X_training, X_test, Y_training, Y_test = data.stack_mnist('./data/stackedmnist', args.Ntrain, args.Ntest, args.imageSize)
    #X_training, _, _, _ = data.stack_mnist('./data/stackedmnist', args.Ntrain, args.Ntest, args.imageSize)

    #X_training, _, _, _ = data.stack_mnist('./data/stackedmnist', args.Ntrain, args.Ntest, args.imageSize)

    #X_training, _, _, _ = data.stack_mnist('./data/stackedmnist', args.Ntrain, args.Ntest, args.imageSize)
    #_, X_training, _, _ = data.stack_mnist('./data/stackedmnist', args.Ntrain, args.Ntest, args.imageSize)

    #print(X_training.shape)

    #X_training = torch.mean(X_training, 1, keepdim=True)
    #print(X_training.shape)

    #from keras.datasets import fashion_mnist
    #(X_training, _), (_, _) = fashion_mnist.load_data()

    #import torchvision

    #im_dim = 1
    #init_layer = layers.LogitTransform(1e-6)
    #n_classes = 10

    #print(args.imageSize)

    #train_loader = datasets.FashionMNIST(
    #        args.dataroot, train=True, transform=transforms.Compose([
    #            transforms.ToTensor(), transforms.Resize(args.imageSize),
    #            ])
    #    )
    #test_loader = datasets.FashionMNIST(
    #        args.dataroot, train=False, transform=transforms.Compose([
    #            transforms.ToTensor(), transforms.Resize(args.imageSize),
    #            ])
    #    )

    """
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            args.dataroot, train=True, transform=transforms.Compose([
                transforms.Resize(args.imageSize),
                transforms.ToTensor(),
            ])
        ),
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=args.workers,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            args.dataroot, train=False, transform=transforms.Compose([
                transforms.Resize(args.imageSize),
                transforms.ToTensor(),
            ])
        ),
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=args.workers,
    )
    """

    #print(X_training.shape)
    #X_training = test_loader.fashionmnist.data

    #X_training = test_loader.fashionmnist.data
    #X_training = train_loader.fashionmnist.data

    """
    data_path = './data/'
    imgsize = args.imageSize

    nc = 1
    transform = transforms.Compose([
        transforms.Resize(imgsize),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    mnist = torchvision.datasets.FashionMNIST(root=data_path, download=True, transform=transform, train=True)
    train_loader = DataLoader(mnist, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
    X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
    Y_training = torch.zeros(len(train_loader))
    for i, x in enumerate(train_loader):
        X_training[i, :, :, :] = x[0]
        Y_training[i] = x[1]
        if i % 10000 == 0:
            print('Loading data... {}/{}'.format(i, len(train_loader)))

    mnist = torchvision.datasets.FashionMNIST(root=data_path, download=True, transform=transform, train=False)
    test_loader = DataLoader(mnist, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
    X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
    Y_test = torch.zeros(len(test_loader))
    for i, x in enumerate(test_loader):
        X_test[i, :, :, :] = x[0]
        Y_test[i] = x[1]
        if i % 1000 == 0:
            print('i: {}/{}'.format(i, len(test_loader)))

    Y_training = Y_training.type('torch.LongTensor')
    Y_test = Y_test.type('torch.LongTensor')

    dat = {'X_train': X_training, 'Y_train': Y_training, 'X_test': X_test, 'Y_test': Y_test, 'nc': nc}

    X_training = dat['X_test'].to(device)
    """

    #print(X_training.shape)
    #X_training = dat['X_train'].to(device)

    #X_training = dat['X_train'].to(device)
    #X_training = dat['X_test'].to(device)

    #X_training = torchvision.datasets.FashionMNIST('', train=False, transform=None, target_transform=None, download=True)
    #print(X_training.shape)

    #asdfasdfsdfs

    """
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import matplotlib.cbook as cbook
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    fig, ax = plt.subplots()
    im = ax.imshow(X_training[0, 0, :, :].cpu())

    plt.show()
    """

    #print(X_training.shape)
    #adsfasdfadsf

    #sdafadf

    print('')

    #print('')
    #print('')

    #print(X_training.shape)
    #asfasdfafas

    #sdafsafs

    #sdfasf
    #sdfsa

    #asdfsadf



    # 0 to 9 vs 1 to 9
    # Train G(z)1 to 9.
    #
    # Detect abnormal OoD datasets.
    # Fashion-MNIST OoD data datasets
    #
    # Write the code for detecting
    # abnormal OoD datasets such as Fashion-MNIST.
    #
    # Abnormal OoD datasets
    # Detect abnormal OoD datasets.

    #sdfasfasfsdf

    # Detect abnormal OoD datasets.
    # We detect abnormal OoD datasets.

    #netG.eval()

    #for param in netG.parameters():
    #   param.requires_grad = False

    #print('')

    #print(X_training.shape)
    #asdfasfasxz

    #print(X_training.shape)

    #print(X_training.shape)
    #print(X_training.shape)

    #asdfasdfsdfs

    #print(X_training.shape)
    #sdafasdfsaf

    #print(X_training.shape)
    #adsfasfassdf

    # import numpy as np
    # import torchvision
    #
    # from torch.utils.data import Dataset, DataLoader
    # from torchvision import transforms

    '''
    netG.eval()

    for param in netG.parameters():
        param.requires_grad = False
    '''

    """
    imgsize = args.imageSize
    #data_path = args.dataroot

    #print(args.dataroot)
    #asdfasdfassadf

    args.dataroot = 'dataset'

    data_path = args.dataroot
    #data_path = 'data2'

    nc = 1
    transform = transforms.Compose([
            transforms.Resize(imgsize), 
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]) 

    mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)
    train_loader = DataLoader(mnist, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
    X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
    Y_training = torch.zeros(len(train_loader))
    for i, x in enumerate(train_loader):
        X_training[i, :, :, :] = x[0]
        Y_training[i] = x[1]
        if i % 10000 == 0:
            print('Loading data... {}/{}'.format(i, len(train_loader)))

    mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=False)
    test_loader = DataLoader(mnist, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
    X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
    Y_test = torch.zeros(len(test_loader))
    for i, x in enumerate(test_loader):
        X_test[i, :, :, :] = x[0]
        Y_test[i] = x[1]
        if i % 1000 == 0:
            print('i: {}/{}'.format(i, len(test_loader)))

    Y_training = Y_training.type('torch.LongTensor')
    Y_test = Y_test.type('torch.LongTensor')

    dat = {'X_train': X_training, 'Y_train': Y_training, 'X_test': X_test, 'Y_test': Y_test, 'nc': nc}

    #X_training = dat['X_train'].to(device)
    X_training = dat['X_test'].to(device)

    #X_training = dat['X_train'].to(device)
    #print(X_training.shape)

    #X_training = dat['X_test'].to(device)
    #print(X_training.shape)

    #print(X_training.shape)
    #print(X_training.shape)
    """

    #adsfasfasdfs

    #asdfgasg
    #sadfdaxsz

    #sadfa
    #asdfas

    '''
    data_path = 'dataset2'
    imgsize = args.imageSize

    nc = 1
    transform = transforms.Compose([
            transforms.Resize(imgsize), 
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]) 

    mnist = torchvision.datasets.FashionMNIST(root=data_path, download=True, transform=transform, train=True)
    train_loader = DataLoader(mnist, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
    X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
    Y_training = torch.zeros(len(train_loader))
    for i, x in enumerate(train_loader):
        X_training[i, :, :, :] = x[0]
        Y_training[i] = x[1]
        if i % 10000 == 0:
            print('Loading data... {}/{}'.format(i, len(train_loader)))

    mnist = torchvision.datasets.FashionMNIST(root=data_path, download=True, transform=transform, train=False)
    test_loader = DataLoader(mnist, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
    X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
    Y_test = torch.zeros(len(test_loader))
    for i, x in enumerate(test_loader):
        X_test[i, :, :, :] = x[0]
        Y_test[i] = x[1]
        if i % 1000 == 0:
            print('i: {}/{}'.format(i, len(test_loader)))

    Y_training = Y_training.type('torch.LongTensor')
    Y_test = Y_test.type('torch.LongTensor')

    dat = {'X_train': X_training, 'Y_train': Y_training, 'X_test': X_test, 'Y_test': Y_test, 'nc': nc}

    #X_training = dat['X_train'].to(device)
    X_training = dat['X_test'].to(device)

    #X_training = dat['X_train'].to(device)
    #print(X_training.shape)

    #X_training = dat.['X_test'].to(device)
    #print(X_training.shape)

    #print(X_training.shape)
    #print(X_training.shape)
    '''

    #sadfadsfs

    #sadf
    #asdfa

    """
    data_path = 'dataset3'
    imgsize = args.imageSize

    nc = 1
    transform = transforms.Compose([
            transforms.Resize(imgsize), 
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]) 

    mnist = torchvision.datasets.KMNIST(root=data_path, download=True, transform=transform, train=True)
    train_loader = DataLoader(mnist, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
    X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
    Y_training = torch.zeros(len(train_loader))
    for i, x in enumerate(train_loader):
        X_training[i, :, :, :] = x[0]
        Y_training[i] = x[1]
        if i % 10000 == 0:
            print('Loading data... {}/{}'.format(i, len(train_loader)))

    mnist = torchvision.datasets.KMNIST(root=data_path, download=True, transform=transform, train=False)
    test_loader = DataLoader(mnist, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
    X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
    Y_test = torch.zeros(len(test_loader))
    for i, x in enumerate(test_loader):
        X_test[i, :, :, :] = x[0]
        Y_test[i] = x[1]
        if i % 1000 == 0:
            print('i: {}/{}'.format(i, len(test_loader)))

    Y_training = Y_training.type('torch.LongTensor')
    Y_test = Y_test.type('torch.LongTensor')

    dat = {'X_train': X_training, 'Y_train': Y_training, 'X_test': X_test, 'Y_test': Y_test, 'nc': nc}

    #X_training = dat['X_train'].to(device)
    X_training = dat['X_test'].to(device)

    #X_training = dat['X_train'].to(device)
    print(X_training.shape)

    #X_training = dat.['X_test'].to(device)
    #print(X_training.shape)
    """

    #asdfasdfas
    
    #adsfas
    #asdfasdf

    '''
    data_path = 'dataset4'
    imgsize = args.imageSize

    nc = 1
    transform = transforms.Compose([
            transforms.Resize(imgsize), 
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]) 

    mnist = torchvision.datasets.QMNIST(root=data_path, download=True, transform=transform, train=True)
    train_loader = DataLoader(mnist, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
    X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
    Y_training = torch.zeros(len(train_loader))
    for i, x in enumerate(train_loader):
        X_training[i, :, :, :] = x[0]
        Y_training[i] = x[1]
        if i % 10000 == 0:
            print('Loading data... {}/{}'.format(i, len(train_loader)))

    mnist = torchvision.datasets.QMNIST(root=data_path, download=True, transform=transform, train=False)
    test_loader = DataLoader(mnist, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
    X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
    Y_test = torch.zeros(len(test_loader))
    for i, x in enumerate(test_loader):
        X_test[i, :, :, :] = x[0]
        Y_test[i] = x[1]
        if i % 1000 == 0:
            print('i: {}/{}'.format(i, len(test_loader)))

    Y_training = Y_training.type('torch.LongTensor')
    Y_test = Y_test.type('torch.LongTensor')

    dat = {'X_train': X_training, 'Y_train': Y_training, 'X_test': X_test, 'Y_test': Y_test, 'nc': nc}

    #X_training = dat['X_train'].to(device)
    X_training = dat['X_test'].to(device)

    #X_training = dat['X_train'].to(device)
    print(X_training.shape)

    #X_training = dat.['X_test'].to(device)
    #print(X_training.shape)

    #print(X_training.shape)
    #print(X_training.shape)
    '''

    #sadfasfsadf
    
    #asdfas
    #asfaszd

    """
    nc = 1
    transform = transforms.Compose([
            transforms.Resize(imgsize), 
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]) 

    mnist = torchvision.datasets.EMNIST(root=data_path, download=True, transform=transform, train=True)
    train_loader = DataLoader(mnist, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
    X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
    Y_training = torch.zeros(len(train_loader))
    for i, x in enumerate(train_loader):
        X_training[i, :, :, :] = x[0]
        Y_training[i] = x[1]
        if i % 10000 == 0:
            print('Loading data... {}/{}'.format(i, len(train_loader)))

    mnist = torchvision.datasets.EMNIST(root=data_path, download=True, transform=transform, train=False)
    test_loader = DataLoader(mnist, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
    X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
    Y_test = torch.zeros(len(test_loader))
    for i, x in enumerate(test_loader):
        X_test[i, :, :, :] = x[0]
        Y_test[i] = x[1]
        if i % 1000 == 0:
            print('i: {}/{}'.format(i, len(test_loader)))

    Y_training = Y_training.type('torch.LongTensor')
    Y_test = Y_test.type('torch.LongTensor')

    dat = {'X_train': X_training, 'Y_train': Y_training, 'X_test': X_test, 'Y_test': Y_test, 'nc': nc}

    X_training = dat.['X_train'].to(device)
    print(X_training.shape)

    #X_training = dat.['X_test'].to(device)
    #print(X_training.shape)
    """

    """
    # dataset = datasets.MNIST(root='./data')
    # idx = dataset.train_labels==1
    # dataset.train_labels = dataset.train_labels[idx]
    # dataset.train_data = dataset.train_data[idx]
    
    #dataset = datasets.MNIST(root='./data')
    #idx = dataset.train_labels==1
    #dataset.train_labels = dataset.train_labels[idx]
    #dataset.train_data = dataset.train_data[idx]
    
    nc = 1
    transform = transforms.Compose([
            transforms.Resize(imgsize), 
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]) 

    #mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)
    
    #mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)
    mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)
    
    idx = mnist.train_labels==0
    mnist.train_labels = mnist.train_labels[idx]
    
    mnist.train_data = mnist.train_data[idx]
    #mnist.train_data = mnist.train_data[idx]
    
    #mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)
    train_loader = DataLoader(mnist, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
    X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
    Y_training = torch.zeros(len(train_loader))
    for i, x in enumerate(train_loader):
        X_training[i, :, :, :] = x[0]
        Y_training[i] = x[1]
        if i % 10000 == 0:
            print('Loading data... {}/{}'.format(i, len(train_loader)))

    #mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=False)
    
    #mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=False)
    mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=False)
    
    idx = mnist.train_labels==0
    mnist.train_labels = mnist.train_labels[idx]
    
    mnist.train_data = mnist.train_data[idx]
    #mnist.train_data = mnist.train_data[idx]
    
    #mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=False)
    test_loader = DataLoader(mnist, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
    X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
    Y_test = torch.zeros(len(test_loader))
    for i, x in enumerate(test_loader):
        X_test[i, :, :, :] = x[0]
        Y_test[i] = x[1]
        if i % 1000 == 0:
            print('i: {}/{}'.format(i, len(test_loader)))

    Y_training = Y_training.type('torch.LongTensor')
    Y_test = Y_test.type('torch.LongTensor')

    dat = {'X_train': X_training, 'Y_train': Y_training, 'X_test': X_test, 'Y_test': Y_test, 'nc': nc}

    X_training = dat.['X_train'].to(device)
    print(X_training.shape)

    #X_training = dat.['X_test'].to(device)
    #print(X_training.shape)
    """

    '''
    # dataset = datasets.MNIST(root='./data')
    # idx = dataset.train_labels==1
    # dataset.train_labels = dataset.train_labels[idx]
    # dataset.train_data = dataset.train_data[idx]
    
    #dataset = datasets.MNIST(root='./data')
    #idx = dataset.train_labels==1
    #dataset.train_labels = dataset.train_labels[idx]
    #dataset.train_data = dataset.train_data[idx]
    
    nc = 1
    transform = transforms.Compose([
            transforms.Resize(imgsize), 
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]) 

    #mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)
    
    #mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)
    mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)
    
    idx = mnist.train_labels!=0
    mnist.train_labels = mnist.train_labels[idx]
    
    mnist.train_data = mnist.train_data[idx]
    #mnist.train_data = mnist.train_data[idx]
    
    #mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)
    train_loader = DataLoader(mnist, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
    X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
    Y_training = torch.zeros(len(train_loader))
    for i, x in enumerate(train_loader):
        X_training[i, :, :, :] = x[0]
        Y_training[i] = x[1]
        if i % 10000 == 0:
            print('Loading data... {}/{}'.format(i, len(train_loader)))

    #mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=False)
    
    #mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=False)
    mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=False)
    
    idx = mnist.train_labels!=0
    mnist.train_labels = mnist.train_labels[idx]
    
    mnist.train_data = mnist.train_data[idx]
    #mnist.train_data = mnist.train_data[idx]
    
    #mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=False)
    test_loader = DataLoader(mnist, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
    X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
    Y_test = torch.zeros(len(test_loader))
    for i, x in enumerate(test_loader):
        X_test[i, :, :, :] = x[0]
        Y_test[i] = x[1]
        if i % 1000 == 0:
            print('i: {}/{}'.format(i, len(test_loader)))

    Y_training = Y_training.type('torch.LongTensor')
    Y_test = Y_test.type('torch.LongTensor')

    dat = {'X_train': X_training, 'Y_train': Y_training, 'X_test': X_test, 'Y_test': Y_test, 'nc': nc}

    X_training = dat.['X_train'].to(device)
    print(X_training.shape)

    #X_training = dat.['X_test'].to(device)
    #print(X_training.shape)
    '''

    """
    real_cpu = X_training

    print('')
    #print(real_cpu.shape)

    #print(real_cpu.shape)
    #print(real_cpu.shape)

    #print(real_cpu.shape)
    #print(real_cpu.shape)

    p_probP = torch.zeros(1, device=device)

    #varInIn = torch.randn(batch_size, args.nz, 1, 1, device=device)
    #varOutOut = netG(varInIn)
    
    # use: https://pytorch.org/docs/stable/torchvision/datasets.html
    # we use: https://pytorch.org/docs/stable/torchvision/datasets.html

    with torch.no_grad():
        #varInIn = torch.randn(batch_size, args.nz, 1, 1, device=device)
        #varInIn = torch.randn(1024, args.nz, 1, 1, device=device)
        varInIn = torch.randn(2048, args.nz, 1, 1, device=device)
        varOutOut = netG(varInIn)
        g_error, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(p_probP,
                                                                                     varOutOut, args, netG2,
                                                                                     varInIn,
                                                                                     real_cpu.to(device))
        # ROC and AUROC on L_1
        # AUROC on secondOnly_lossGen
    """

    '''
    #real_cpu = X_training

    #real_cpu = X_training
    real_cpu = X_training

    # print('')
    # print(real_cpu.shape)

    # print(real_cpu.shape)
    # print(real_cpu.shape)

    # print(real_cpu.shape)
    # print(real_cpu.shape)

    p_probP = torch.zeros(1, device=device)

    # varInIn = torch.randn(batch_size, args.nz, 1, 1, device=device)
    # varOutOut = netG(varInIn)

    # use: https://pytorch.org/docs/stable/torchvision/datasets.html
    # we use: https://pytorch.org/docs/stable/torchvision/datasets.html

    with torch.no_grad():
        # varInIn = torch.randn(batch_size, args.nz, 1, 1, device=device)
        # varInIn = torch.randn(1024, args.nz, 1, 1, device=device)
        varInIn = torch.randn(2048, args.nz, 1, 1, device=device)
        varOutOut = netG(varInIn)

        out = varOutOut
        # g_fake_data = varOutOut

        # gen_input = varInIn
        gen_input = torch.randn(10000, args.nz, 1, 1, device=device)

        # gen_input = varInIn
        sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)

        noise_eta = torch.randn_like(out)
        g_fake_data = out + noise_eta * sigma_x

        g_error, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(p_probP,
                                                                                         varOutOut, args, netG2,
                                                                                         varInIn,
                                                                                         real_cpu.to(device))
        # ROC and AUROC on L_1
        # AUROC on secondOnly_lossGen
    '''



    """
    with torch.no_grad():
        # varInIn = torch.randn(batch_size, args.nz, 1, 1, device=device)
        # varInIn = torch.randn(1024, args.nz, 1, 1, device=device)
        varInIn = torch.randn(2048, args.nz, 1, 1, device=device)
        varOutOut = netG(varInIn)

        out = varOutOut
        # g_fake_data = varOutOut

        # gen_input = varInIn
        gen_input = torch.randn(10000, args.nz, 1, 1, device=device)

        # gen_input = varInIn
        sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)

        noise_eta = torch.randn_like(out)
        g_fake_data = out + noise_eta * sigma_x

    # hmc_samples, acceptRate, stepsize, _ = hmc.get_samples(
    #    netG2, real_cpu.detach(), gen_input.clone(), sigma_x.detach(), args.burn_in,
    #    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
    #    args.hmc_learning_rate, args.hmc_opt_accept)
    
    #adsfasdfsf
    
    #asdfsdfs

    sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)

    out21 = out
    out = real_cpu

    g_fake_data21 = g_fake_data

    # out = netG(gen_input)
    noise_eta = torch.randn_like(out)
    g_fake_data = out + noise_eta * sigma_x

    hmc_samples, acceptRate, stepsize, _ = hmc.get_samples(
        netG, g_fake_data.detach(), gen_input.clone(), sigma_x.detach(), args.burn_in,
        args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
        args.hmc_learning_rate, args.hmc_opt_accept)

    bsz, d = hmc_samples.size()
    mean_output = netG(hmc_samples.view(bsz, d, 1, 1).to(device))
    bsz = g_fake_data.size(0)

    mean_output_summed = torch.zeros_like(g_fake_data)
    for cnt in range(args.num_samples_posterior):
        mean_output_summed = mean_output_summed + mean_output[cnt * bsz:(cnt + 1) * bsz]
    mean_output_summed = mean_output_summed / args.num_samples_posterior

    c = ((g_fake_data - mean_output_summed) / sigma_x ** 2).detach()
    g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()

    print(g_error_entropy)
    asdfasdfas

    # 11004.9570
    # 13257.9102

    out = out21
    g_fake_data = g_fake_data21

    with torch.no_grad():
        # bsz, d = hmc_samples.size()
        # mean_output = netG(hmc_samples.view(bsz, d, 1, 1).to(device))
        ##bsz = g_fake_data.size(0)
        # bsz = real_cpu.size(0)

        # mean_output_summed = torch.zeros_like(g_fake_data)
        # mean_output_summed = torch.zeros_like(real_cpu)
        # for cnt in range(args.num_samples_posterior):
        #    mean_output_summed = mean_output_summed + mean_output[cnt * bsz:(cnt + 1) * bsz]
        # mean_output_summed = mean_output_summed / args.num_samples_posterior

        # c = ((g_fake_data - mean_output_summed) / sigma_x ** 2).detach()

        # c = ((g_fake_data - mean_output_summed) / sigma_x ** 2).detach()
        # c = ((gGgGg_fake_data - mean_output_summed) / sigma_x ** 2).detach()

        # c = ((gGgGg_fake_data - mean_output_summed) / sigma_x ** 2).detach()
        # c = ((varOutOut - mean_output_summed) / sigma_x ** 2).detach()

        # c = ((real_cpu - mean_output_summed) / sigma_x ** 2).detach()
        # c = ((varOutOut - mean_output_summed) / sigma_x ** 2)

        # print(c)
        # print(c.requires_grad)

        # sdafasfsafs

        # c = ((g_fake_data - mean_output_summed) / sigma_x ** 2).detach()
        # g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()

        # g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()

        # g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()
        # g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).mean()

        # p_probP = -g_error_entropy

        myNikMy_entropy = torch.exp(-g_error_entropy)
        # nikNikmyNikMy_entropy = scipy.special.lambertw(myNikMy_entropy.cpu().detach().numpy())

        # print(g_error_entropy)
        # print(myNikMy_entropy)

        # print(nikNikmyNikMy_entropy)
        # print(g_error_entropy)

        # ndNdnikNikmyNikMy_entropy = torch.zeros(1, device=device, requires_grad=False)
        # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * np.real(nikNikmyNikMy_entropy)

        # print(ndNdnikNikmyNikMy_entropy)
        # asdfasdfasdf

        # print(ndNdnikNikmyNikMy_entropy)
        # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5

        # print(ndNdnikNikmyNikMy_entropy)
        # print(ndNdnikNikmyNikMy_entropy.requires_grad)

        # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5

        # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5
        ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5

        for _ in range(200):
            ndNdnikNikmyNikMy_entropy -= (
                    (ndNdnikNikmyNikMy_entropy.clone() * torch.exp(
                        ndNdnikNikmyNikMy_entropy.clone()) - myNikMy_entropy) / (
                            torch.exp(ndNdnikNikmyNikMy_entropy.clone()) + (
                            ndNdnikNikmyNikMy_entropy.clone() * torch.exp(ndNdnikNikmyNikMy_entropy.clone()))))

            # ndNdnikNikmyNikMy_entropy -= (
            #            (ndNdnikNikmyNikMy_entropy * torch.exp(ndNdnikNikmyNikMy_entropy) - myNikMy_entropy) / (
            #                torch.exp(ndNdnikNikmyNikMy_entropy) + (
            #                    ndNdnikNikmyNikMy_entropy * torch.exp(ndNdnikNikmyNikMy_entropy))))

        # print(ndNdnikNikmyNikMy_entropy)
        # asdfasdfas

        # print(ndNdnikNikmyNikMy_entropy)
        # print(ndNdnikNikmyNikMy_entropy.requires_grad)

        # asdfasdfas

        # p_probP = -g_error_entropy
        # p_probP = g_error_entropy

        # g_error_entropy = -g_error_entropy
        # (?)

        # print(g_error_entropy)
        # aasdfasfsaf

        # print(ndNdnikNikmyNikMy_entropy)
        # adfadsfasdf

        # l1_usel1 = torch.log(g_error_entropy)
        # l2_usel2 = torch.log(torch.log(g_error_entropy))

        '''
        # use: t = torch.log(F.relu(t) + 1e-7)
        l1_usel1 = torch.log(F.relu(g_error_entropy) + 1e-7)

        # use: t = torch.log(F.relu(t) + 1e-7)
        l2_usel2 = torch.log(F.relu(torch.log(F.relu(g_error_entropy) + 1e-7)) + 1e-7)

        #print('')
        #print(l1_usel1)

        #print(l1_usel1)
        #print(l2_usel2)

        gErrorEntropy2 = l1_usel1 - l2_usel2 + (l2_usel2 / l1_usel1) + (
                (l2_usel2 * (-2 + l2_usel2)) / (2 * (l1_usel1 ** 2))) + (
                                 ((l2_usel2 * (6 - (9 * l2_usel2) + (2 * (l2_usel2 ** 2))))) / (
                                 6 * (l1_usel1 ** 3))) + ((l2_usel2 * (
                -12 + (36 * l2_usel2) - (22 * (l2_usel2 ** 2)) + (3 * (l2_usel2 ** 3)))) / (
                                                                  12 * (l1_usel1 ** 4))) + ((l2_usel2 * (
                    60 - (300 * l2_usel2) + (350 * (l2_usel2 ** 2)) - (125 * (l2_usel2 ** 3)) + (
                        12 * (l2_usel2 ** 4)))) / (60 * (l1_usel1 ** 5)))
        '''

        # gErrorEntropy2 = g_error_entropy - (g_error_entropy ** 2) + (1.5 * (g_error_entropy ** 3)) - (
        #        (8 / 3) * (g_error_entropy ** 4)) + ((125 / 24) * (g_error_entropy ** 5))

        # gErrorEntropy = torch.exp(gErrorEntropy2)

        # gErrorEntropy = torch.exp(gErrorEntropy2)
        # p_probP = torch.exp(gErrorEntropy2)

        # p_probP = torch.exp(gErrorEntropy2)
        # p_probP = g_error_entropy / gErrorEntropy2

        # print(ndNdnikNikmyNikMy_entropy)

        p_probP = ndNdnikNikmyNikMy_entropy
        # print(p_probP)

        print(p_probP)
        # asdfasfasz

        g_error, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(p_probP,
                                                                                         varOutOut, args, netG2,
                                                                                         varInIn,
                                                                                         real_cpu.to(device))
        # ROC and AUROC on L_1
        # AUROC on secondOnly_lossGen
    """

    '''
    print('')
    print(g_error)

    # print(g_error)
    print(firstOnly_lossGen)

    print(secondOnly_lossGen)
    print(thirdOnly_lossGen)

    # safsafs

    # asdfdsa
    # sadfadsf

    print('')
    # print(X_training.shape)

    # asdfdasfdsdfs
    '''

    #print(g_error)
    #print(g_error)

    #print('')
    #print(g_error)

    #print(g_error)
    #print(firstOnly_lossGen)

    #print(secondOnly_lossGen)
    #print(thirdOnly_lossGen)

    # safsafs

    # asdfdsa
    # sadfadsf

    #print('')
    #print(X_training.shape)

    #asdfdasfdsdfs

    # print(X_training.shape)
    # adfasdfasdfzs

    # print(X_training.shape)
    # sadfasdfasfz

    # print(X_training.shape)
    # asdfasfasdf

    # sadfsa
    # safsafs

    #asdfasfasfas

    # sadfsa
    # sadfa



    # safsafs
    
    #asdfdsa
    #sadfadsf

    #print(X_training.shape)
    #sadfasdfasfz

    #print(X_training.shape)
    #asdfasfasfsz

    #print(X_training.shape)
    #asdfasfasdf

    #print(X_training.shape)
    #print(X_training.shape)

    # sadfsa
    #safsafs

    #sadfsa
    #sadfa

    #print(X_training.shape)
    #asdfasfsfsz

    #sadfasdfs

    #print(X_training.shape)
    #sadfasfzdfsz

    # netG.eval()
    # netG.eval()

    # netG.eval()

    # netG.eval()
    netG.eval()

    for param in netG.parameters():
        param.requires_grad = False

    # print(X_training.shape)

    X_training = dat['X_test'].to(device)
    x = X_training

    y = dat['Y_test'].to(device)

    args.batchSize = 1
    bsz = args.batchSize

    nrand = 100
    args.val_batchsize = args.batchSize

    # print(X_training.shape)
    # asdfasdfasdf

    # print(X_training.shape)

    # print(X_training.shape)
    # print(X_training.shape)

    losses_NIKlosses = []

    loLosses_NIKlosses = []
    loLosses_NIKlosses2 = []

    # loLosses_NIKlosses2 = []
    loLosses_NIKlosses3 = []

    for epoch in range(1, 1 + 1):
        for i in range(0, len(X_training), bsz):
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

            # x = x.to(device)
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

            # with torch.no_grad():
            #    firstOnly_lossGen2 = computeLoss(x, model)

            with torch.no_grad():
                sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)

                netD.zero_grad()
                stop = min(bsz, len(X_training[i:]))
                real_cpu = X_training[i:i + stop].to(device)

                # print(real_cpu.shape)
                # asdfasdf

                # batch_size = real_cpu.size(0)
                batch_size = args.batchSize
                label = torch.full((batch_size,), real_label, device=device)

                noise_eta = torch.randn_like(real_cpu)
                noised_data = real_cpu + sigma_x.detach() * noise_eta
                # out_real = netD(noised_data)
                # errD_real = criterion(out_real, label)
                # errD_real.backward()
                # D_x = out_real.mean().item()

                # train with fake

                # noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
                # mu_fake = netG(noise)
                # fake = mu_fake + sigma_x * noise_eta
                # label.fill_(fake_label)
                # out_fake = netD(fake.detach())
                # errD_fake = criterion(out_fake, label)
                # errD_fake.backward()
                # D_G_z1 = out_fake.mean().item()
                # errD = errD_real + errD_fake
                # optimizerD.step()

                # update G network: maximize log(D(G(z)))

                netG.zero_grad()
                sigma_optimizer.zero_grad()

                label.fill_(real_label)
                gen_input = torch.randn(batch_size, args.nz, 1, 1, device=device)
                out = netG(gen_input)

                # print(out.shape)
                # asdfasdf

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
                    # hmc_samples, acceptRate, stepsize = hmc.get_samples(
                    #    netG, g_fake_data.detach(), gen_input.clone(), sigma_x.detach(), args.burn_in,
                    #        args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                    #            args.hmc_learning_rate, args.hmc_opt_accept)

                    # bsz, d = hmc_samples.size()
                    # mean_output = netG(hmc_samples.view(bsz, d, 1, 1).to(device))
                    # bsz = g_fake_data.size(0)

                    # mean_output_summed = torch.zeros_like(g_fake_data)
                    # for cnt in range(args.num_samples_posterior):
                    #    mean_output_summed = mean_output_summed + mean_output[cnt*bsz:(cnt+1)*bsz]
                    # mean_output_summed = mean_output_summed / args.num_samples_posterior

                    # c = ((g_fake_data - mean_output_summed) / sigma_x**2).detach()
                    # g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()

                    # print(mean_output)
                    # print(mean_output.shape)

                    # print(bsz)
                    # print(d)

                    # print(torch.randn(batch_size, args.nz, 1, 1, device=device).shape)
                    # print(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device).shape)

                    # use: netG( torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device) )
                    # print(netG( torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device) ).shape)

                    # netG( torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device) )
                    # we use: netG( torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device) )

                    # print(g_error_entropy)
                    # asdfasdfds

                    # print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).shape)
                    # asdfsdfs

                    # print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).requires_grad)
                    # print(netG(torch.randn(batch_size, args.nz, 1, 1, device=device)).requires_grad)

                    # netG2.eval()

                    # for param in netG2.parameters():
                    #    param.requires_grad = False

                    # print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).shape)
                    # print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).requires_grad)

                    # print(netG2(torch.randn(batch_size, args.nz, 1, 1, requires_grad=False, device=device)).shape)
                    # print(netG2(torch.randn(batch_size, args.nz, 1, 1, requires_grad=False, device=device)).requires_grad)

                    # print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).requires_grad)
                    # print(netG2(torch.randn(batch_size, args.nz, 1, 1, requires_grad=False, device=device)).requires_grad)

                    # print(netG2(torch.randn(batch_size, args.nz, 1, 1, requires_grad=False, device=device)).requires_grad)
                    # print(netG2(torch.randn(batch_size, args.nz, 1, 1, device=device)).requires_grad)

                    # asdfasdf

                    # _, _, _, p_probP = hmc.get_samples(
                    #    netG2, netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).detach(),
                    #    gen_input.clone(), sigma_x.detach(), args.burn_in,
                    #    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                    #    args.hmc_learning_rate, args.hmc_opt_accept)

                    '''
                    _, _, _, p_probP = hmc.get_samples(
                        netG2, netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).detach(),
                        gen_input.clone(), sigma_x.detach(), args.burn_in,
                        args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                        args.hmc_learning_rate, args.hmc_opt_accept)
                    '''

                    # print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).requires_grad)
                    # sdfasdfs

                    _, _, _, p_probP = hmc2.get_samples(
                        netG, netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)),
                        gen_input.clone(), sigma_x.detach(), args.burn_in,
                        args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                        args.hmc_learning_rate, args.hmc_opt_accept)

                    # _, _, _, p_probP = hmc.get_samples(
                    #    netG2, netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)),
                    #    gen_input.clone(), sigma_x.detach(), args.burn_in,
                    #    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                    #    args.hmc_learning_rate, args.hmc_opt_accept)

                firstOnly_lossGen2 = p_probP.mean()

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

            # if y.item() == 1:
            # if y[i] == 2:

            # if y[i] == 2:
            if y[i] == 0:
                # loLosses_NIKlosses3.append(0)

                # loLosses_NIKlosses3.append(0)
                loLosses_NIKlosses3.append(1)

                # print(y)
                # print(y.item())

            else:
                # loLosses_NIKlosses3.append(1)

                # loLosses_NIKlosses3.append(1)
                loLosses_NIKlosses3.append(0)

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

    import numpy as np

    # print(loLosses_NIKlosses3)
    # print(len(loLosses_NIKlosses3))

    print('')

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
        plt.savefig('mnMnistFor6MyROC.png', bbox_inches='tight')

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
        plt.savefig('mnMnistFor6MyPR.png', bbox_inches='tight')

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
        plt.savefig('mnMnistFor6MyROCPR.png', bbox_inches='tight')

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

    # import numpy as np
    loLosses_NIKlosses3 = np.array(loLosses_NIKlosses3)

    # where_0 = np.where(loLosses_NIKlosses3 == 0)
    # where_1 = np.where(loLosses_NIKlosses3 == 1)

    # loLosses_NIKlosses3[where_0] = 1
    # loLosses_NIKlosses3[where_1] = 0

    indices_one = loLosses_NIKlosses3 == 1
    indices_zero = loLosses_NIKlosses3 == 0

    loLosses_NIKlosses3[indices_one] = 0  # replacing 1s with 0s
    loLosses_NIKlosses3[indices_zero] = 1  # replacing 0s with 1s

    loLosses_NIKlosses3 = loLosses_NIKlosses3.tolist()

    # del where_0
    # del where_1

    # print(loLosses_NIKlosses3)
    # print(len(loLosses_NIKlosses3))

    # adsfasdfzs

    # print(loLosses_NIKlosses2)
    # print(loLosses_NIKlosses3)

    # import numpy as np
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
        plt.savefig('mnMnistFor6MyROC.png', bbox_inches='tight')

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
        plt.savefig('mnMnistFor6MyPR.png', bbox_inches='tight')

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
        plt.savefig('mnMnistFor6MyROCPR.png', bbox_inches='tight')

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

    asdfas
    asdfasfas

    # print(X_training.shape)
    # print(X_training.shape)

    # print(X_training.shape)
    # asdfasdfasf

    #asdfasdfa

    # print(X_training.shape)
    # print(X_training.shape)

    """
    # netG.eval()
    # netG.eval()

    # netG.eval()

    # netG.eval()
    netG.eval()

    for param in netG.parameters():
        param.requires_grad = False

    # print(X_training.shape)

    X_training = dat['X_test'].to(device)
    x = X_training

    y = dat['Y_test'].to(device)

    args.batchSize = 1
    bsz = args.batchSize

    nrand = 100
    args.val_batchsize = args.batchSize

    # print(X_training.shape)
    # asdfasdfasdf

    # print(X_training.shape)

    # print(X_training.shape)
    # print(X_training.shape)

    losses_NIKlosses = []

    loLosses_NIKlosses = []
    loLosses_NIKlosses2 = []

    # loLosses_NIKlosses2 = []
    loLosses_NIKlosses3 = []

    for epoch in range(1, 1 + 1):
        for i in range(0, len(X_training), bsz):
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

            # x = x.to(device)
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

            # with torch.no_grad():
            #    firstOnly_lossGen2 = computeLoss(x, model)

            with torch.no_grad():
                sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)

                netD.zero_grad()
                stop = min(bsz, len(X_training[i:]))
                real_cpu = X_training[i:i + stop].to(device)

                # print(real_cpu.shape)
                # asdfasdf

                # batch_size = real_cpu.size(0)
                batch_size = args.batchSize
                label = torch.full((batch_size,), real_label, device=device)

                noise_eta = torch.randn_like(real_cpu)
                noised_data = real_cpu + sigma_x.detach() * noise_eta
                # out_real = netD(noised_data)
                # errD_real = criterion(out_real, label)
                # errD_real.backward()
                # D_x = out_real.mean().item()

                # train with fake

                # noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
                # mu_fake = netG(noise)
                # fake = mu_fake + sigma_x * noise_eta
                # label.fill_(fake_label)
                # out_fake = netD(fake.detach())
                # errD_fake = criterion(out_fake, label)
                # errD_fake.backward()
                # D_G_z1 = out_fake.mean().item()
                # errD = errD_real + errD_fake
                # optimizerD.step()

                # update G network: maximize log(D(G(z)))

                netG.zero_grad()
                sigma_optimizer.zero_grad()

                label.fill_(real_label)
                gen_input = torch.randn(batch_size, args.nz, 1, 1, device=device)
                out = netG(gen_input)

                # print(out.shape)
                # asdfasdf

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
                    # hmc_samples, acceptRate, stepsize = hmc.get_samples(
                    #    netG, g_fake_data.detach(), gen_input.clone(), sigma_x.detach(), args.burn_in,
                    #        args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                    #            args.hmc_learning_rate, args.hmc_opt_accept)

                    # bsz, d = hmc_samples.size()
                    # mean_output = netG(hmc_samples.view(bsz, d, 1, 1).to(device))
                    # bsz = g_fake_data.size(0)

                    # mean_output_summed = torch.zeros_like(g_fake_data)
                    # for cnt in range(args.num_samples_posterior):
                    #    mean_output_summed = mean_output_summed + mean_output[cnt*bsz:(cnt+1)*bsz]
                    # mean_output_summed = mean_output_summed / args.num_samples_posterior

                    # c = ((g_fake_data - mean_output_summed) / sigma_x**2).detach()
                    # g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()

                    # print(mean_output)
                    # print(mean_output.shape)

                    # print(bsz)
                    # print(d)

                    # print(torch.randn(batch_size, args.nz, 1, 1, device=device).shape)
                    # print(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device).shape)

                    # use: netG( torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device) )
                    # print(netG( torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device) ).shape)

                    # netG( torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device) )
                    # we use: netG( torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device) )

                    # print(g_error_entropy)
                    # asdfasdfds

                    # print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).shape)
                    # asdfsdfs

                    # print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).requires_grad)
                    # print(netG(torch.randn(batch_size, args.nz, 1, 1, device=device)).requires_grad)

                    # netG2.eval()

                    # for param in netG2.parameters():
                    #    param.requires_grad = False

                    # print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).shape)
                    # print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).requires_grad)

                    # print(netG2(torch.randn(batch_size, args.nz, 1, 1, requires_grad=False, device=device)).shape)
                    # print(netG2(torch.randn(batch_size, args.nz, 1, 1, requires_grad=False, device=device)).requires_grad)

                    # print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).requires_grad)
                    # print(netG2(torch.randn(batch_size, args.nz, 1, 1, requires_grad=False, device=device)).requires_grad)

                    # print(netG2(torch.randn(batch_size, args.nz, 1, 1, requires_grad=False, device=device)).requires_grad)
                    # print(netG2(torch.randn(batch_size, args.nz, 1, 1, device=device)).requires_grad)

                    # asdfasdf

                    # _, _, _, p_probP = hmc.get_samples(
                    #    netG2, netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).detach(),
                    #    gen_input.clone(), sigma_x.detach(), args.burn_in,
                    #    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                    #    args.hmc_learning_rate, args.hmc_opt_accept)

                    '''
                    _, _, _, p_probP = hmc.get_samples(
                        netG2, netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).detach(),
                        gen_input.clone(), sigma_x.detach(), args.burn_in,
                        args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                        args.hmc_learning_rate, args.hmc_opt_accept)
                    '''

                    # print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).requires_grad)
                    # sdfasdfs

                    _, _, _, p_probP = hmc2.get_samples(
                        netG, netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)),
                        gen_input.clone(), sigma_x.detach(), args.burn_in,
                        args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                        args.hmc_learning_rate, args.hmc_opt_accept)

                    # _, _, _, p_probP = hmc.get_samples(
                    #    netG2, netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)),
                    #    gen_input.clone(), sigma_x.detach(), args.burn_in,
                    #    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                    #    args.hmc_learning_rate, args.hmc_opt_accept)

                firstOnly_lossGen2 = p_probP.mean()

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

            # if y.item() == 1:
            # if y[i] == 2:

            # if y[i] == 2:
            if y[i] == 0:
                # loLosses_NIKlosses3.append(0)

                # loLosses_NIKlosses3.append(0)
                loLosses_NIKlosses3.append(1)

                # print(y)
                # print(y.item())

            else:
                # loLosses_NIKlosses3.append(1)

                # loLosses_NIKlosses3.append(1)
                loLosses_NIKlosses3.append(0)

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

    import numpy as np

    # print(loLosses_NIKlosses3)
    # print(len(loLosses_NIKlosses3))

    print('')

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
        plt.savefig('mnMnistFor6MyROC.png', bbox_inches='tight')

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
        plt.savefig('mnMnistFor6MyPR.png', bbox_inches='tight')

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
        plt.savefig('mnMnistFor6MyROCPR.png', bbox_inches='tight')

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

    # import numpy as np
    loLosses_NIKlosses3 = np.array(loLosses_NIKlosses3)

    # where_0 = np.where(loLosses_NIKlosses3 == 0)
    # where_1 = np.where(loLosses_NIKlosses3 == 1)

    # loLosses_NIKlosses3[where_0] = 1
    # loLosses_NIKlosses3[where_1] = 0

    indices_one = loLosses_NIKlosses3 == 1
    indices_zero = loLosses_NIKlosses3 == 0

    loLosses_NIKlosses3[indices_one] = 0  # replacing 1s with 0s
    loLosses_NIKlosses3[indices_zero] = 1  # replacing 0s with 1s

    loLosses_NIKlosses3 = loLosses_NIKlosses3.tolist()

    # del where_0
    # del where_1

    # print(loLosses_NIKlosses3)
    # print(len(loLosses_NIKlosses3))

    # adsfasdfzs

    # print(loLosses_NIKlosses2)
    # print(loLosses_NIKlosses3)

    # import numpy as np
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
        plt.savefig('mnMnistFor6MyROC.png', bbox_inches='tight')

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
        plt.savefig('mnMnistFor6MyPR.png', bbox_inches='tight')

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
        plt.savefig('mnMnistFor6MyROCPR.png', bbox_inches='tight')

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

    asdfas
    asdfasfas
    """

    #asdfasf

    #asdfasfas
    #asdfasfadfs



    for epoch in range(1, args.epochs+1):
        for i in range(0, len(X_training), bsz): 
            sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)

            netD.zero_grad()
            stop = min(bsz, len(X_training[i:]))
            real_cpu = X_training[i:i+stop].to(device)

            #print(real_cpu.shape)
            #asdfasdf

            #batch_size = real_cpu.size(0)
            batch_size = args.batchSize
            label = torch.full((batch_size,), real_label, device=device)

            #noise_eta = torch.randn_like(real_cpu)
            #noised_data = real_cpu + sigma_x.detach() * noise_eta

            #out_real = netD(noised_data)
            #errD_real = criterion(out_real, label)
            #errD_real.backward()
            #D_x = out_real.mean().item()

            # train with fake
            
            #noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            #mu_fake = netG(noise)
            #fake = mu_fake + sigma_x * noise_eta
            #label.fill_(fake_label)
            #out_fake = netD(fake.detach())
            #errD_fake = criterion(out_fake, label)
            #errD_fake.backward()
            #D_G_z1 = out_fake.mean().item()
            #errD = errD_real + errD_fake
            #optimizerD.step()

            # update G network: maximize log(D(G(z)))

            netG.zero_grad()
            sigma_optimizer.zero_grad()

            label.fill_(real_label)  

            #gen_input = torch.randn(batch_size, args.nz, 1, 1, device=device)
            #gen_input = torch.randn(batch_size, args.nz, 1, 1, device=device)

            gen_input = torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)
            out = netG(gen_input)

            varInIn = gen_input
            varOutOut = out

            #varInIn = torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)
            #varOutOut = netG(varInIn)

            #print(out.shape)
            #asdfasdf

            noise_eta = torch.randn_like(out)
            g_fake_data = out + noise_eta * sigma_x

            #dg_fake_decision = netD(g_fake_data)
            #g_error_gan = criterion(dg_fake_decision, label)
            #D_G_z2 = dg_fake_decision.mean().item()

            if args.lambda_ == 0:
                #g_error_gan.backward()

                optimizerG.step()
                #sigma_optimizer.step()

            else:
                #hmc_samples, acceptRate, stepsize = hmc.get_samples(
                #    netG, g_fake_data.detach(), gen_input.clone(), sigma_x.detach(), args.burn_in,
                #    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                #    args.hmc_learning_rate, args.hmc_opt_accept)

                #print(g_fake_data.shape)
                #print(real_cpu.shape)

                #asdfadsf

                #print(real_cpu.shape)
                #asdfasdfdassa

                """
                hmc_samples, acceptRate, stepsize = hmc.get_samples(
                    netG2, real_cpu, gen_input.clone(), sigma_x.detach(), args.burn_in,
                    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                    args.hmc_learning_rate, args.hmc_opt_accept)

                bsz, d = hmc_samples.size()
                mean_output = netG(hmc_samples.view(bsz, d, 1, 1).to(device))
                bsz = g_fake_data.size(0)

                mean_output_summed = torch.zeros_like(g_fake_data)
                for cnt in range(args.num_samples_posterior):
                    mean_output_summed = mean_output_summed + mean_output[cnt * bsz:(cnt + 1) * bsz]
                mean_output_summed = mean_output_summed / args.num_samples_posterior

                #c = ((g_fake_data - mean_output_summed) / sigma_x ** 2).detach()

                #c = ((g_fake_data - mean_output_summed) / sigma_x ** 2).detach()
                c = ((real_cpu - mean_output_summed) / sigma_x ** 2).detach()

                #c = ((g_fake_data - mean_output_summed) / sigma_x ** 2).detach()
                g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()

                g_error_entropy = -g_error_entropy
                print(g_error_entropy)
                """

                #adfadfasdf

                #hmc_samples, acceptRate, stepsize = hmc.get_samples(
                #    netG, g_fake_data.detach(), gen_input.clone(), sigma_x.detach(), args.burn_in,
                #        args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                #            args.hmc_learning_rate, args.hmc_opt_accept)
                
                #bsz, d = hmc_samples.size()
                #mean_output = netG(hmc_samples.view(bsz, d, 1, 1).to(device))
                #bsz = g_fake_data.size(0)

                #mean_output_summed = torch.zeros_like(g_fake_data)
                #for cnt in range(args.num_samples_posterior):
                #    mean_output_summed = mean_output_summed + mean_output[cnt*bsz:(cnt+1)*bsz]
                #mean_output_summed = mean_output_summed / args.num_samples_posterior

                #c = ((g_fake_data - mean_output_summed) / sigma_x**2).detach()
                #g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()

                #print(mean_output)
                #print(mean_output.shape)

                #print(bsz)
                #print(d)

                #print(torch.randn(batch_size, args.nz, 1, 1, device=device).shape)
                #print(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device).shape)

                # use: netG( torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device) )
                #print(netG( torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device) ).shape)

                # netG( torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device) )
                # we use: netG( torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device) )

                #print(g_error_entropy)
                #asdfasdfds

                #print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).shape)
                #asdfsdfs

                #print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).requires_grad)
                #print(netG(torch.randn(batch_size, args.nz, 1, 1, device=device)).requires_grad)

                #netG2.eval()

                #for param in netG2.parameters():
                #    param.requires_grad = False

                #print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).shape)
                #print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).requires_grad)

                #print(netG2(torch.randn(batch_size, args.nz, 1, 1, requires_grad=False, device=device)).shape)
                #print(netG2(torch.randn(batch_size, args.nz, 1, 1, requires_grad=False, device=device)).requires_grad)

                #print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).requires_grad)
                #print(netG2(torch.randn(batch_size, args.nz, 1, 1, requires_grad=False, device=device)).requires_grad)

                #print(netG2(torch.randn(batch_size, args.nz, 1, 1, requires_grad=False, device=device)).requires_grad)
                #print(netG2(torch.randn(batch_size, args.nz, 1, 1, device=device)).requires_grad)

                #asdfasdf

                #_, _, _, p_probP = hmc.get_samples(
                #    netG2, netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).detach(),
                #    gen_input.clone(), sigma_x.detach(), args.burn_in,
                #    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                #    args.hmc_learning_rate, args.hmc_opt_accept)

                '''
                _, _, _, p_probP = hmc.get_samples(
                    netG2, netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).detach(),
                    gen_input.clone(), sigma_x.detach(), args.burn_in,
                    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                    args.hmc_learning_rate, args.hmc_opt_accept)
                '''

                #print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).requires_grad)
                #sdfasdfs

                #hmc_samples, acceptRate, stepsize = hmc.get_samples(
                #    netG, g_fake_data.detach(), gen_input.clone(), sigma_x.detach(), args.burn_in,
                #    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                #    args.hmc_learning_rate, args.hmc_opt_accept)

                '''
                gGgGg_fake_data2 = torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)
                gGgGg_fake_data = netG(gGgGg_fake_data2)

                hmc_samples, acceptRate, stepsize, _ = hmc.get_samples(
                    netG2, gGgGg_fake_data, gen_input.clone(), sigma_x.detach(), args.burn_in,
                    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                    args.hmc_learning_rate, args.hmc_opt_accept)

                bsz, d = hmc_samples.size()
                mean_output = netG(hmc_samples.view(bsz, d, 1, 1).to(device))
                bsz = g_fake_data.size(0)

                mean_output_summed = torch.zeros_like(g_fake_data)
                for cnt in range(args.num_samples_posterior):
                    mean_output_summed = mean_output_summed + mean_output[cnt * bsz:(cnt + 1) * bsz]
                mean_output_summed = mean_output_summed / args.num_samples_posterior

                #c = ((g_fake_data - mean_output_summed) / sigma_x ** 2).detach()

                #c = ((g_fake_data - mean_output_summed) / sigma_x ** 2).detach()
                c = ((gGgGg_fake_data - mean_output_summed) / sigma_x ** 2).detach()

                #c = ((g_fake_data - mean_output_summed) / sigma_x ** 2).detach()
                g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()

                p_probP = -g_error_entropy
                #g_error = g_error_gan - args.lambda_ * g_error_entropy
                '''

                #_, _, _, p_probP = hmc.get_samples(
                #    netG2, netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)),
                #    gen_input.clone(), sigma_x.detach(), args.burn_in,
                #    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                #    args.hmc_learning_rate, args.hmc_opt_accept)

                #print(p_probP.shape)
                #print(p_probP.mean())

                # 0.0004
                #asdfasdfas

                #_, _, _, p_probP = hmc.get_samples(
                #    netG2, X_training[0:0+64].to(device),
                #    gen_input.clone(), sigma_x.detach(), args.burn_in,
                #    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                #    args.hmc_learning_rate, args.hmc_opt_accept)

                #print(p_probP.mean())
                #print(p_probP.mean().grad)

                #print(p_probP.mean().requires_grad)
                #sadfasdfks

                #print(p_probP.mean())
                #print(p_probP.mean().requires_grad)

                #asdfasfdfs

                #  -609010.3125
                # -1401163.0000

                #print(p_probP.mean())
                #print(p_probP.mean().requires_grad)

                #asdfasdfs

                #print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).requires_grad)
                #print(netG2(torch.randn(batch_size, args.nz, 1, 1, device=device)).requires_grad)

                #print(p_probP.mean())
                #print(p_probP.requires_grad)
                #print(p_probP.shape)

                #print(p_probP.mean())
                #print(p_probP.mean().requires_grad)

                # g_error = p_probP.mean() + (?)
                # use: g_error = p_probP.mean() + (?)

                #asdfsadf

                # g_error = (?)
                # g_error = p_probP.mean() + (?)

                #print(p_probP.mean())
                #print(p_probP.mean().requires_grad)

                # g_error = p_probP.mean() +
                # we use: g_error = p_probP.mean() +

                # p_probP.mean() +
                # g_error = p_probP.mean() +

                # we now use: p_probP.mean()
                #firstTerm_theFirstTerm = p_probP.mean()

                #varInIn = torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)
                #varOutOut = netG(varInIn)

                #gGgGg_fake_data2 = torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)
                #gGgGg_fake_data = netG(gGgGg_fake_data2)

                #hmc_samples, acceptRate, stepsize, _ = hmc.get_samples(
                #    netG2, gGgGg_fake_data, gen_input.clone(), sigma_x.detach(), args.burn_in,
                #    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                #    args.hmc_learning_rate, args.hmc_opt_accept)

                #hmc_samples, acceptRate, stepsize = hmc.get_samples(
                #    netG, g_fake_data.detach(), gen_input.clone(), sigma_x.detach(), args.burn_in,
                #    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                #    args.hmc_learning_rate, args.hmc_opt_accept)

                #hmc_samples, acceptRate, stepsize, _ = hmc.get_samples(
                #    netG2, varOutOut, gen_input.clone(), sigma_x.detach(), args.burn_in,
                #    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                #    args.hmc_learning_rate, args.hmc_opt_accept)

                hmc_samples, acceptRate, stepsize, _ = hmc.get_samples(
                    netG2, varOutOut.detach(), gen_input.clone(), sigma_x.detach(), args.burn_in,
                    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                    args.hmc_learning_rate, args.hmc_opt_accept)

                bsz, d = hmc_samples.size()
                mean_output = netG(hmc_samples.view(bsz, d, 1, 1).to(device))
                bsz = g_fake_data.size(0)

                mean_output_summed = torch.zeros_like(g_fake_data)
                for cnt in range(args.num_samples_posterior):
                    mean_output_summed = mean_output_summed + mean_output[cnt * bsz:(cnt + 1) * bsz]
                mean_output_summed = mean_output_summed / args.num_samples_posterior

                #c = ((g_fake_data - mean_output_summed) / sigma_x ** 2).detach()

                #c = ((g_fake_data - mean_output_summed) / sigma_x ** 2).detach()
                #c = ((gGgGg_fake_data - mean_output_summed) / sigma_x ** 2).detach()

                #c = ((gGgGg_fake_data - mean_output_summed) / sigma_x ** 2).detach()
                #c = ((varOutOut - mean_output_summed) / sigma_x ** 2).detach()

                c = ((varOutOut - mean_output_summed) / sigma_x ** 2).detach()
                #c = ((varOutOut - mean_output_summed) / sigma_x ** 2)

                #print(c)
                #print(c.requires_grad)

                #sdafasfsafs

                #c = ((g_fake_data - mean_output_summed) / sigma_x ** 2).detach()
                #g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()

                #g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()

                g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()
                #g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).mean()



                #hmc_samples, acceptRate, stepsize = hmc.get_samples(
                #    netG, g_fake_data.detach(), gen_input.clone(), sigma_x.detach(), args.burn_in,
                #    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                #    args.hmc_learning_rate, args.hmc_opt_accept)

                #bsz, d = hmc_samples.size()
                #mean_output = netG(hmc_samples.view(bsz, d, 1, 1).to(device))
                #bsz = g_fake_data.size(0)

                #mean_output_summed = torch.zeros_like(g_fake_data)
                #for cnt in range(args.num_samples_posterior):
                #    mean_output_summed = mean_output_summed + mean_output[cnt * bsz:(cnt + 1) * bsz]
                #mean_output_summed = mean_output_summed / args.num_samples_posterior

                #c = ((g_fake_data - mean_output_summed) / sigma_x ** 2).detach()
                #g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()

                #print(g_error_entropy)
                #asdfadsfas



                """
                myNikMy_entropy = torch.exp(-g_error_entropy)
                # nikNikmyNikMy_entropy = scipy.special.lambertw(myNikMy_entropy.cpu().detach().numpy())

                # print(g_error_entropy)
                # print(myNikMy_entropy)

                # print(nikNikmyNikMy_entropy)
                # print(g_error_entropy)

                # ndNdnikNikmyNikMy_entropy = torch.zeros(1, device=device, requires_grad=False)
                # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * np.real(nikNikmyNikMy_entropy)

                # print(ndNdnikNikmyNikMy_entropy)
                # asdfasdfasdf

                # print(ndNdnikNikmyNikMy_entropy)
                # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5

                # print(ndNdnikNikmyNikMy_entropy)
                # print(ndNdnikNikmyNikMy_entropy.requires_grad)

                # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5

                # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5
                ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5

                # for _ in range(10):

                # for _ in range(10):
                for _ in range(100):
                    ndNdnikNikmyNikMy_entropy -= (
                            (ndNdnikNikmyNikMy_entropy.clone() * torch.exp(
                                ndNdnikNikmyNikMy_entropy.clone()) - myNikMy_entropy) / (
                                    torch.exp(ndNdnikNikmyNikMy_entropy.clone()) + (
                                    ndNdnikNikmyNikMy_entropy.clone() * torch.exp(ndNdnikNikmyNikMy_entropy.clone()))))

                    # ndNdnikNikmyNikMy_entropy -= (
                    #            (ndNdnikNikmyNikMy_entropy * torch.exp(ndNdnikNikmyNikMy_entropy) - myNikMy_entropy) / (
                    #                torch.exp(ndNdnikNikmyNikMy_entropy) + (
                    #                    ndNdnikNikmyNikMy_entropy * torch.exp(ndNdnikNikmyNikMy_entropy))))

                # print(ndNdnikNikmyNikMy_entropy)
                # asdfasdfas

                # print(ndNdnikNikmyNikMy_entropy)
                # print(ndNdnikNikmyNikMy_entropy.requires_grad)

                # asdfasdfas

                # p_probP = -g_error_entropy
                # p_probP = g_error_entropy

                # g_error_entropy = -g_error_entropy
                # (?)

                # print(g_error_entropy)
                # aasdfasfsaf

                # print(ndNdnikNikmyNikMy_entropy)
                # adfadsfasdf

                # l1_usel1 = torch.log(g_error_entropy)
                # l2_usel2 = torch.log(torch.log(g_error_entropy))

                '''
                # use: t = torch.log(F.relu(t) + 1e-7)
                l1_usel1 = torch.log(F.relu(g_error_entropy) + 1e-7)

                # use: t = torch.log(F.relu(t) + 1e-7)
                l2_usel2 = torch.log(F.relu(torch.log(F.relu(g_error_entropy) + 1e-7)) + 1e-7)

                #print('')
                #print(l1_usel1)

                #print(l1_usel1)
                #print(l2_usel2)

                gErrorEntropy2 = l1_usel1 - l2_usel2 + (l2_usel2 / l1_usel1) + (
                        (l2_usel2 * (-2 + l2_usel2)) / (2 * (l1_usel1 ** 2))) + (
                                         ((l2_usel2 * (6 - (9 * l2_usel2) + (2 * (l2_usel2 ** 2))))) / (
                                         6 * (l1_usel1 ** 3))) + ((l2_usel2 * (
                        -12 + (36 * l2_usel2) - (22 * (l2_usel2 ** 2)) + (3 * (l2_usel2 ** 3)))) / (
                                                                          12 * (l1_usel1 ** 4))) + ((l2_usel2 * (
                            60 - (300 * l2_usel2) + (350 * (l2_usel2 ** 2)) - (125 * (l2_usel2 ** 3)) + (
                                12 * (l2_usel2 ** 4)))) / (60 * (l1_usel1 ** 5)))
                '''

                # gErrorEntropy2 = g_error_entropy - (g_error_entropy ** 2) + (1.5 * (g_error_entropy ** 3)) - (
                #        (8 / 3) * (g_error_entropy ** 4)) + ((125 / 24) * (g_error_entropy ** 5))

                # gErrorEntropy = torch.exp(gErrorEntropy2)

                # gErrorEntropy = torch.exp(gErrorEntropy2)
                # p_probP = torch.exp(gErrorEntropy2)

                # p_probP = torch.exp(gErrorEntropy2)
                # p_probP = g_error_entropy / gErrorEntropy2

                # print(ndNdnikNikmyNikMy_entropy)

                p_probP = ndNdnikNikmyNikMy_entropy
                # print(p_probP)

                #print(p_probP)
                #asdfasdfas

                #print(p_probP)
                #print(p_probP.requires_grad)

                #asdfasdfas

                #print(g_error_entropy)
                #print(g_error_entropy.requires_grad)

                #print(g_error_entropy.shape)
                #asdfasdfas
                """



                """
                #print(g_error_entropy)
                #asdfadsfas

                myNikMy_entropy = torch.exp(-g_error_entropy)
                # nikNikmyNikMy_entropy = scipy.special.lambertw(myNikMy_entropy.cpu().detach().numpy())

                # print(g_error_entropy)
                # print(myNikMy_entropy)

                # print(nikNikmyNikMy_entropy)
                # print(g_error_entropy)

                # ndNdnikNikmyNikMy_entropy = torch.zeros(1, device=device, requires_grad=False)
                # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * np.real(nikNikmyNikMy_entropy)

                # print(ndNdnikNikmyNikMy_entropy)
                # asdfasdfasdf

                # print(ndNdnikNikmyNikMy_entropy)
                # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5

                # print(ndNdnikNikmyNikMy_entropy)
                # print(ndNdnikNikmyNikMy_entropy.requires_grad)

                # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5

                # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5
                ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5

                # for _ in range(10):

                # for _ in range(10):
                for _ in range(100):
                    ndNdnikNikmyNikMy_entropy -= (
                            (ndNdnikNikmyNikMy_entropy.clone() * torch.exp(
                                ndNdnikNikmyNikMy_entropy.clone()) - myNikMy_entropy) / (
                                    torch.exp(ndNdnikNikmyNikMy_entropy.clone()) + (
                                    ndNdnikNikmyNikMy_entropy.clone() * torch.exp(ndNdnikNikmyNikMy_entropy.clone()))))

                    # ndNdnikNikmyNikMy_entropy -= (
                    #            (ndNdnikNikmyNikMy_entropy * torch.exp(ndNdnikNikmyNikMy_entropy) - myNikMy_entropy) / (
                    #                torch.exp(ndNdnikNikmyNikMy_entropy) + (
                    #                    ndNdnikNikmyNikMy_entropy * torch.exp(ndNdnikNikmyNikMy_entropy))))

                # print(ndNdnikNikmyNikMy_entropy)
                # asdfasdfas

                # print(ndNdnikNikmyNikMy_entropy)
                # print(ndNdnikNikmyNikMy_entropy.requires_grad)

                # asdfasdfas

                # p_probP = -g_error_entropy
                # p_probP = g_error_entropy

                # g_error_entropy = -g_error_entropy
                # (?)

                # print(g_error_entropy)
                # aasdfasfsaf

                # print(ndNdnikNikmyNikMy_entropy)
                # adfadsfasdf

                # l1_usel1 = torch.log(g_error_entropy)
                # l2_usel2 = torch.log(torch.log(g_error_entropy))

                '''
                # use: t = torch.log(F.relu(t) + 1e-7)
                l1_usel1 = torch.log(F.relu(g_error_entropy) + 1e-7)

                # use: t = torch.log(F.relu(t) + 1e-7)
                l2_usel2 = torch.log(F.relu(torch.log(F.relu(g_error_entropy) + 1e-7)) + 1e-7)

                #print('')
                #print(l1_usel1)

                #print(l1_usel1)
                #print(l2_usel2)

                gErrorEntropy2 = l1_usel1 - l2_usel2 + (l2_usel2 / l1_usel1) + (
                        (l2_usel2 * (-2 + l2_usel2)) / (2 * (l1_usel1 ** 2))) + (
                                         ((l2_usel2 * (6 - (9 * l2_usel2) + (2 * (l2_usel2 ** 2))))) / (
                                         6 * (l1_usel1 ** 3))) + ((l2_usel2 * (
                        -12 + (36 * l2_usel2) - (22 * (l2_usel2 ** 2)) + (3 * (l2_usel2 ** 3)))) / (
                                                                          12 * (l1_usel1 ** 4))) + ((l2_usel2 * (
                            60 - (300 * l2_usel2) + (350 * (l2_usel2 ** 2)) - (125 * (l2_usel2 ** 3)) + (
                                12 * (l2_usel2 ** 4)))) / (60 * (l1_usel1 ** 5)))
                '''

                # gErrorEntropy2 = g_error_entropy - (g_error_entropy ** 2) + (1.5 * (g_error_entropy ** 3)) - (
                #        (8 / 3) * (g_error_entropy ** 4)) + ((125 / 24) * (g_error_entropy ** 5))

                # gErrorEntropy = torch.exp(gErrorEntropy2)

                # gErrorEntropy = torch.exp(gErrorEntropy2)
                # p_probP = torch.exp(gErrorEntropy2)

                # p_probP = torch.exp(gErrorEntropy2)
                # p_probP = g_error_entropy / gErrorEntropy2

                # print(ndNdnikNikmyNikMy_entropy)

                p_probP = ndNdnikNikmyNikMy_entropy
                # print(p_probP)

                #print(p_probP)
                #asdfasdfas
                """



                """
                # print(g_error_entropy)
                # asdfadsfas

                myNikMy_entropy = torch.exp(-g_error_entropy)
                # nikNikmyNikMy_entropy = scipy.special.lambertw(myNikMy_entropy.cpu().detach().numpy())

                # print(g_error_entropy)
                # print(myNikMy_entropy)

                # print(nikNikmyNikMy_entropy)
                # print(g_error_entropy)

                # ndNdnikNikmyNikMy_entropy = torch.zeros(1, device=device, requires_grad=False)
                # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * np.real(nikNikmyNikMy_entropy)

                # print(ndNdnikNikmyNikMy_entropy)
                # asdfasdfasdf

                # print(ndNdnikNikmyNikMy_entropy)
                # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5

                # print(ndNdnikNikmyNikMy_entropy)
                # print(ndNdnikNikmyNikMy_entropy.requires_grad)

                # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5

                # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5
                ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5

                # for _ in range(10):

                # for _ in range(10):
                for _ in range(100):
                    ndNdnikNikmyNikMy_entropy -= (
                            (ndNdnikNikmyNikMy_entropy.clone() * torch.exp(
                                ndNdnikNikmyNikMy_entropy.clone()) - myNikMy_entropy) / (
                                    torch.exp(ndNdnikNikmyNikMy_entropy.clone()) + (
                                    ndNdnikNikmyNikMy_entropy.clone() * torch.exp(ndNdnikNikmyNikMy_entropy.clone()))))

                    # ndNdnikNikmyNikMy_entropy -= (
                    #            (ndNdnikNikmyNikMy_entropy * torch.exp(ndNdnikNikmyNikMy_entropy) - myNikMy_entropy) / (
                    #                torch.exp(ndNdnikNikmyNikMy_entropy) + (
                    #                    ndNdnikNikmyNikMy_entropy * torch.exp(ndNdnikNikmyNikMy_entropy))))

                # print(ndNdnikNikmyNikMy_entropy)
                # asdfasdfas

                # print(ndNdnikNikmyNikMy_entropy)
                # print(ndNdnikNikmyNikMy_entropy.requires_grad)

                # asdfasdfas

                # p_probP = -g_error_entropy
                # p_probP = g_error_entropy

                # g_error_entropy = -g_error_entropy
                # (?)

                # print(g_error_entropy)
                # aasdfasfsaf

                # print(ndNdnikNikmyNikMy_entropy)
                # adfadsfasdf

                # l1_usel1 = torch.log(g_error_entropy)
                # l2_usel2 = torch.log(torch.log(g_error_entropy))

                '''
                # use: t = torch.log(F.relu(t) + 1e-7)
                l1_usel1 = torch.log(F.relu(g_error_entropy) + 1e-7)

                # use: t = torch.log(F.relu(t) + 1e-7)
                l2_usel2 = torch.log(F.relu(torch.log(F.relu(g_error_entropy) + 1e-7)) + 1e-7)

                #print('')
                #print(l1_usel1)

                #print(l1_usel1)
                #print(l2_usel2)

                gErrorEntropy2 = l1_usel1 - l2_usel2 + (l2_usel2 / l1_usel1) + (
                        (l2_usel2 * (-2 + l2_usel2)) / (2 * (l1_usel1 ** 2))) + (
                                         ((l2_usel2 * (6 - (9 * l2_usel2) + (2 * (l2_usel2 ** 2))))) / (
                                         6 * (l1_usel1 ** 3))) + ((l2_usel2 * (
                        -12 + (36 * l2_usel2) - (22 * (l2_usel2 ** 2)) + (3 * (l2_usel2 ** 3)))) / (
                                                                          12 * (l1_usel1 ** 4))) + ((l2_usel2 * (
                            60 - (300 * l2_usel2) + (350 * (l2_usel2 ** 2)) - (125 * (l2_usel2 ** 3)) + (
                                12 * (l2_usel2 ** 4)))) / (60 * (l1_usel1 ** 5)))
                '''

                # gErrorEntropy2 = g_error_entropy - (g_error_entropy ** 2) + (1.5 * (g_error_entropy ** 3)) - (
                #        (8 / 3) * (g_error_entropy ** 4)) + ((125 / 24) * (g_error_entropy ** 5))

                # gErrorEntropy = torch.exp(gErrorEntropy2)

                # gErrorEntropy = torch.exp(gErrorEntropy2)
                # p_probP = torch.exp(gErrorEntropy2)

                # p_probP = torch.exp(gErrorEntropy2)
                # p_probP = g_error_entropy / gErrorEntropy2

                # print(ndNdnikNikmyNikMy_entropy)

                p_probP = ndNdnikNikmyNikMy_entropy
                # print(p_probP)

                # print(p_probP)
                # asdfasdfas
                """



                # (?)
                #p_probP = -g_error_entropy
                # (?)

                # print(g_error_entropy)
                # asdfadsfas

                myNikMy_entropy = torch.exp(-g_error_entropy)
                # nikNikmyNikMy_entropy = scipy.special.lambertw(myNikMy_entropy.cpu().detach().numpy())

                # print(g_error_entropy)
                # print(myNikMy_entropy)

                # print(nikNikmyNikMy_entropy)
                # print(g_error_entropy)

                # ndNdnikNikmyNikMy_entropy = torch.zeros(1, device=device, requires_grad=False)
                # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * np.real(nikNikmyNikMy_entropy)

                # print(ndNdnikNikmyNikMy_entropy)
                # asdfasdfasdf

                # print(ndNdnikNikmyNikMy_entropy)
                # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5

                # print(ndNdnikNikmyNikMy_entropy)
                # print(ndNdnikNikmyNikMy_entropy.requires_grad)

                # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5

                # ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5
                ndNdnikNikmyNikMy_entropy = torch.ones(1, device=device) * 0.5

                for _ in range(200):
                    ndNdnikNikmyNikMy_entropy -= (
                            (ndNdnikNikmyNikMy_entropy.clone() * torch.exp(
                                ndNdnikNikmyNikMy_entropy.clone()) - myNikMy_entropy) / (
                                    torch.exp(ndNdnikNikmyNikMy_entropy.clone()) + (
                                    ndNdnikNikmyNikMy_entropy.clone() * torch.exp(ndNdnikNikmyNikMy_entropy.clone()))))

                    # ndNdnikNikmyNikMy_entropy -= (
                    #            (ndNdnikNikmyNikMy_entropy * torch.exp(ndNdnikNikmyNikMy_entropy) - myNikMy_entropy) / (
                    #                torch.exp(ndNdnikNikmyNikMy_entropy) + (
                    #                    ndNdnikNikmyNikMy_entropy * torch.exp(ndNdnikNikmyNikMy_entropy))))

                # print(ndNdnikNikmyNikMy_entropy)
                # asdfasdfas

                # print(ndNdnikNikmyNikMy_entropy)
                # print(ndNdnikNikmyNikMy_entropy.requires_grad)

                # asdfasdfas

                # p_probP = -g_error_entropy
                # p_probP = g_error_entropy

                # g_error_entropy = -g_error_entropy
                # (?)

                # print(g_error_entropy)
                # aasdfasfsaf

                # print(ndNdnikNikmyNikMy_entropy)
                # adfadsfasdf

                # l1_usel1 = torch.log(g_error_entropy)
                # l2_usel2 = torch.log(torch.log(g_error_entropy))

                '''
                # use: t = torch.log(F.relu(t) + 1e-7)
                l1_usel1 = torch.log(F.relu(g_error_entropy) + 1e-7)

                # use: t = torch.log(F.relu(t) + 1e-7)
                l2_usel2 = torch.log(F.relu(torch.log(F.relu(g_error_entropy) + 1e-7)) + 1e-7)

                #print('')
                #print(l1_usel1)

                #print(l1_usel1)
                #print(l2_usel2)

                gErrorEntropy2 = l1_usel1 - l2_usel2 + (l2_usel2 / l1_usel1) + (
                        (l2_usel2 * (-2 + l2_usel2)) / (2 * (l1_usel1 ** 2))) + (
                                         ((l2_usel2 * (6 - (9 * l2_usel2) + (2 * (l2_usel2 ** 2))))) / (
                                         6 * (l1_usel1 ** 3))) + ((l2_usel2 * (
                        -12 + (36 * l2_usel2) - (22 * (l2_usel2 ** 2)) + (3 * (l2_usel2 ** 3)))) / (
                                                                          12 * (l1_usel1 ** 4))) + ((l2_usel2 * (
                            60 - (300 * l2_usel2) + (350 * (l2_usel2 ** 2)) - (125 * (l2_usel2 ** 3)) + (
                                12 * (l2_usel2 ** 4)))) / (60 * (l1_usel1 ** 5)))
                '''

                # gErrorEntropy2 = g_error_entropy - (g_error_entropy ** 2) + (1.5 * (g_error_entropy ** 3)) - (
                #        (8 / 3) * (g_error_entropy ** 4)) + ((125 / 24) * (g_error_entropy ** 5))

                # gErrorEntropy = torch.exp(gErrorEntropy2)

                # gErrorEntropy = torch.exp(gErrorEntropy2)
                # p_probP = torch.exp(gErrorEntropy2)

                # p_probP = torch.exp(gErrorEntropy2)
                # p_probP = g_error_entropy / gErrorEntropy2

                # print(ndNdnikNikmyNikMy_entropy)

                p_probP = ndNdnikNikmyNikMy_entropy
                # print(p_probP)

                # (?)
                #p_probP = -g_error_entropy
                # (?)

                #print(g_error_entropy)
                #g_error_entropy = -g_error_entropy

                #g_error_entropy = -g_error_entropy
                #g_error_entropy = -g_error_entropy

                #g_error_entropy = -g_error_entropy

                #g_error_entropy = -g_error_entropy
                #g_error_entropy = -g_error_entropy

                # (?)
                #g_error_entropy = -g_error_entropy
                # (?)

                #l1_usel1 = torch.log(g_error_entropy)
                #l2_usel2 = torch.log(torch.log(g_error_entropy))

                '''
                # use: t = torch.log(F.relu(t) + 1e-7)
                l1_usel1 = torch.log(F.relu(g_error_entropy) + 1e-7)

                # use: t = torch.log(F.relu(t) + 1e-7)
                l2_usel2 = torch.log(F.relu(torch.log(F.relu(g_error_entropy) + 1e-7)) + 1e-7)

                #print('')
                #print(l1_usel1)

                #print(l1_usel1)
                #print(l2_usel2)

                gErrorEntropy2 = l1_usel1 - l2_usel2 + (l2_usel2 / l1_usel1) + (
                        (l2_usel2 * (-2 + l2_usel2)) / (2 * (l1_usel1 ** 2))) + (
                                         ((l2_usel2 * (6 - (9 * l2_usel2) + (2 * (l2_usel2 ** 2))))) / (
                                         6 * (l1_usel1 ** 3))) + ((l2_usel2 * (
                        -12 + (36 * l2_usel2) - (22 * (l2_usel2 ** 2)) + (3 * (l2_usel2 ** 3)))) / (
                                                                          12 * (l1_usel1 ** 4))) + ((l2_usel2 * (
                            60 - (300 * l2_usel2) + (350 * (l2_usel2 ** 2)) - (125 * (l2_usel2 ** 3)) + (
                                12 * (l2_usel2 ** 4)))) / (60 * (l1_usel1 ** 5)))
                '''

                #gErrorEntropy2 = g_error_entropy - (g_error_entropy ** 2) + (1.5 * (g_error_entropy ** 3)) - (
                #            (8 / 3) * (g_error_entropy ** 4)) + ((125 / 24) * (g_error_entropy ** 5))

                #gErrorEntropy = torch.exp(gErrorEntropy2)

                #gErrorEntropy = torch.exp(gErrorEntropy2)
                #gErrorEntropy = gErrorEntropy2

                #gErrorEntropy = -gErrorEntropy
                #gErrorEntropy = torch.exp(gErrorEntropy)

                #print(gErrorEntropy)
                #dasfasdfz

                #gErrorEntropy = torch.exp(gErrorEntropy2)
                #gErrorEntropy = g_error_entropy / gErrorEntropy2

                #print('')
                #print(gErrorEntropy)

                #print(gErrorEntropy.shape)
                #print(gErrorEntropy.requires_grad)

                #print(gErrorEntropy)
                #asdfasdfs



                #asdfsdfas



                #g_error_entropy = torch.zeros(1, device=device, requires_grad=True)

                #p_probP = -g_error_entropy
                #g_error = g_error_gan - args.lambda_ * g_error_entropy

                #print(p_probP)
                #sdafdsafaf

                g_error, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(p_probP,
                                                                                                 varOutOut, args, netG2,
                                                                                                 varInIn,
                                                                                                 real_cpu.to(device))

                #g_error, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(gErrorEntropy,
                #                                                                                 varOutOut, args, netG2,
                #                                                                                 varInIn,
                #                                                                                 real_cpu.to(device))

                #g_error, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(g_error_entropy,
                #                                                                                 varOutOut, args, netG2,
                #                                                                                 varInIn,
                #                                                                                 real_cpu.to(device))

                #g_error, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(-g_error_entropy,
                #                                                                                 varOutOut, args, netG2,
                #                                                                                 varInIn,
                #                                                                                 real_cpu.to(device))

                #g_error, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(p_probP.mean(),
                #                                                                                 varOutOut, args, netG2,
                #                                                                                 varInIn,
                #                                                                                 real_cpu.to(device))

                #print('')
                #print(g_error)

                #print(g_error)
                #print(firstOnly_lossGen)

                #print(secondOnly_lossGen)
                #print(thirdOnly_lossGen)

                #print(g_error)
                #print(g_error.requires_grad)

                #print(firstOnly_lossGen)
                #print(firstOnly_lossGen.requires_grad)

                #print(secondOnly_lossGen)
                #print(secondOnly_lossGen.requires_grad)

                #print(thirdOnly_lossGen)
                #print(thirdOnly_lossGen.requires_grad)

                #print('')
                #asdfasdfsaf

                #print(firstOnly_lossGen)
                #print(secondOnly_lossGen)

                #print(thirdOnly_lossGen)
                #print(g_error)

                #asdfadsfas

                #firstTerm_theFirstTerm = p_probP.mean()
                #g_error = firstTerm_theFirstTerm +

                #g_error => Use netG( 64, 100, 1, 1 )
                # gen_input = torch.randn(batch_size, args.nz, 1, 1, device=device)

                # (?)
                #g_error = (?)

                #print(torch.mean(netG.main[0].weight.grad))

                #g_error = g_error_gan - args.lambda_ * g_error_entropy
                g_error.backward()

                #print(netG.parameters().grad)
                #print(netG.grad)

                #print(netG[0].weight.grad)
                #print(netG.main[0].weight.grad)

                #print(netG.main[0].weight.grad)
                #print(netG.main[0].weight.grad)

                #print(torch.mean(netG.main[0].weight.grad))
                #gradGrad_lossGen = torch.mean(netG.main[0].weight.grad).item()

                #gradGrad_lossGen = 1.0 / torch.mean(netG.main[0].weight.grad).item()

                # gradGrad_lossGen = 1.0 / torch.mean(netG.main[0].weight.grad).item()

                # use: torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.)
                # grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.)

                # graGradNorm = torch.nn.utils.clip_grad.clip_grad_norm_(netG.parameters(), 1.)
                # graGradNorm = 1.0 / torch.nn.utils.clip_grad.clip_grad_norm_(netG.parameters(), 1.)

                # graGradNorm = 1.0 / torch.nn.utils.clip_grad.clip_grad_norm_(netG.parameters(), 1.)

                # graGradNorm = 1.0 / torch.nn.utils.clip_grad.clip_grad_norm_(netG.parameters(), 1.)
                # gradGrad_lossGen = 1.0 / torch.nn.utils.clip_grad.clip_grad_norm_(netG.parameters(), 1.)

                # gradGrad_lossGen = 1.0 / torch.mean(netG.main[0].weight.grad).item()
                gradGrad_lossGen = 1.0 / torch.nn.utils.clip_grad.clip_grad_norm_(netG.parameters(), 1.)

                #print(gradGrad_lossGen)
                #asdfasdf

                #gradGrad_lossGen = 0.0
                #couCounter31 = 0
                #for param in netG2.parameters():
                #    couCounter31 += 1
                #    gradGrad_lossGen += param.grad()
                #gradGrad_lossGen /= couCounter31
                #del couCounter31

                #asdfasdf

                #gradGrad_lossGen = (torch.mean(netG.lin1.weight.grad) + torch.mean(netG.lin2.weight.grad) + torch.mean(
                #        netG.dc1.weight.grad) + torch.mean(netG.dc2.weight.grad) + torch.mean(
                #        netG.dc3.weight.grad) + torch.mean(netG.lin1.bias.grad) + torch.mean(
                #        netG.lin2.bias.grad) + torch.mean(
                #        netG.dc1.bias.grad) + torch.mean(netG.dc2.bias.grad) + torch.mean(
                #        netG.dc3.bias.grad)).item()

                '''
                gr1loss2_meter.update(
                    (torch.mean(genGen.lin1.weight.grad) + torch.mean(genGen.lin2.weight.grad) + torch.mean(
                        genGen.dc1.weight.grad) + torch.mean(genGen.dc2.weight.grad) + torch.mean(
                        genGen.dc3.weight.grad) + torch.mean(genGen.lin1.bias.grad) + torch.mean(
                        genGen.lin2.bias.grad) + torch.mean(
                        genGen.dc1.bias.grad) + torch.mean(genGen.dc2.bias.grad) + torch.mean(
                        genGen.dc3.bias.grad)).item())

                gr2loss2_meter.update(torch.mean(ggenFGen2.grad).item())
                # gr3loss2_meter.update(torch.mean(genFGen2.grad).item())

                # gr3loss2_meter.update(torch.mean(genFGen2.grad).item())

                # gr3loss2_meter.update(torch.mean(genFGen2.grad).item())
                # gr3loss2_meter.update(torch.mean(genFGen2.grad).item())

                if genFGen2.grad is not None:
                    gr3loss2_meter.update(torch.mean(genFGen2.grad).item())
                else:
                    gr3loss2_meter.update(0.0)
                '''

                optimizerG.step()
                sigma_optimizer.step()

            if args.restrict_sigma:
                log_sigma.data.clamp_(min=logsigma_min, max=logsigma_max)

            ## log performance
            if i % args.log == 0:
                print(
                    'Epoch [%d/%d] .. Batch [%d/%d] .. Loss: %.4f .. L0: %.4f .. L1: %.4f .. L2: %.4f .. G: %.4f'
                    % (epoch, args.epochs, i, len(X_training), g_error.item(), firstOnly_lossGen.item(),
                       secondOnly_lossGen.item(), thirdOnly_lossGen.item(), gradGrad_lossGen))

                loss_theLoss[epoch-1] = g_error.item()
                loss_theLoss0[epoch-1] = firstOnly_lossGen.item()

                loss_theLoss1[epoch-1] = secondOnly_lossGen.item()
                loss_theLoss2[epoch-1] = thirdOnly_lossGen.item()

                loss_theLoss3[epoch-1] = gradGrad_lossGen

                #print(
                #    'Epoch [%d/%d] .. Batch [%d/%d] .. Loss: %.4f .. L0: %.4f .. L1: %.4f .. D(G(z)): %.4f / %.4f'
                #    % (epoch, args.epochs, i, len(X_training), g_error.item(), firstOnly_lossGen.item(),
                #       secondOnly_lossGen.item(), thirdOnly_lossGen.item(), thirdOnly_lossGen.item()))

                #print(
                #    'Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
                #    % (epoch, args.epochs, i, len(X_training), errD.data, g_error_gan.data, D_x, D_G_z1, D_G_z2))

                #print('Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
                #        % (epoch, args.epochs, i, len(X_training), errD.data, g_error_gan.data, D_x, D_G_z1, D_G_z2))

        #print('*'*100)
        #print('End of epoch {}'.format(epoch))
        #print('sigma min: {} .. sigma max: {}'.format(torch.min(sigma_x), torch.max(sigma_x)))
        #print('*'*100)
        #if args.lambda_ > 0:
        #    print('| MCMC diagnostics ====> | stepsize: {} | min ar: {} | mean ar: {} | max ar: {} |'.format(
        #                stepsize, acceptRate.min().item(), acceptRate.mean().item(), acceptRate.max().item()))

        if epoch % args.save_imgs_every == 0:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(loss_theLoss.cpu())
            plt.xlim(0, epoch-1)
            plt.savefig('theFinFinalNiNikNdUoeNdNikUoeMyNdNdNdMyneNewloLossLoss_plot')

            plt.figure()
            plt.plot(loss_theLoss0.cpu())
            plt.xlim(0, epoch-1)
            plt.savefig('theFinFinalNiNikNdUoeNdNikUoeMyNdNdNdMyneNewloLossLoss0_plot')

            plt.figure()
            plt.plot(loss_theLoss1.cpu())
            plt.xlim(0, epoch-1)
            plt.savefig('theFinFinalNiNikNdUoeNdNikUoeMyNdNdNdMyneNewloLossLoss1_plot')

            plt.figure()
            plt.plot(loss_theLoss2.cpu())
            plt.xlim(0, epoch-1)
            plt.savefig('theFinFinalNiNikNdUoeNdNikUoeMyNdNdNdMyneNewloLossLoss2_plot')

            plt.figure()
            plt.plot(loss_theLoss3.cpu())
            plt.xlim(0, epoch - 1)
            plt.savefig('theFinFinalNiNikNdUoeNdNikUoeMyNdNdNdMyneNewloLossLoLossssLoLossLoss2_plot')

            plt.figure()
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].plot(range(1, 1+epoch), loss_theLoss[:epoch].cpu())
            axs[0, 0].set_title('Loss')
            axs[0, 1].plot(range(1, 1+epoch), loss_theLoss0[:epoch].cpu(), 'tab:orange')
            axs[0, 1].set_title('L0')
            axs[1, 0].plot(range(1, 1+epoch), loss_theLoss1[:epoch].cpu(), 'tab:green')
            axs[1, 0].set_title('L1')
            axs[1, 1].plot(range(1, 1+epoch), loss_theLoss2[:epoch].cpu(), 'tab:red')
            axs[1, 1].set_title('L2')
            plt.savefig('theFinFinalNiNikNdUoeNdNikUoeMyNdNdNdMyneNewloLossLossTotal_plot')

            plt.figure()
            fig, axs = plt.subplots(3, 2)
            axs[0, 0].plot(range(1, 1 + epoch), loss_theLoss[:epoch].cpu())
            axs[0, 0].set_title('Loss')
            axs[0, 1].plot(range(1, 1 + epoch), loss_theLoss0[:epoch].cpu(), 'tab:orange')
            axs[0, 1].set_title('L0')
            axs[1, 0].plot(range(1, 1 + epoch), loss_theLoss1[:epoch].cpu(), 'tab:green')
            axs[1, 0].set_title('L1')
            axs[1, 1].plot(range(1, 1 + epoch), loss_theLoss2[:epoch].cpu(), 'tab:red')
            axs[1, 1].set_title('L2')
            axs[2, 1].plot(range(1, 1 + epoch), loss_theLoss3[:epoch].cpu(), 'tab:orange')
            axs[2, 1].set_title('Grad')
            #axs[2, 0].plot(range(1, 1 + epoch), loss_theLoss3[:epoch].cpu(), 'tab:green')
            #axs[2, 0].set_title('L2')
            plt.savefig('theFinFinalNiNikNdUoeNikNdNikUoeMyNdNdNdMyneNewloLossLossTotal_plot')

            #for ax in axs.flat:
            #    ax.set(xlabel='x-label', ylabel='y-label')

            # Hide x labels and tick labels for top plots and y ticks for right plots
            for ax in axs.flat:
                ax.label_outer()

            fake = netG2(fixed_noise).detach()
            vutils.save_image(fake, '%s/presgan_%s_fake_epoch_%03d.png' % (args.results_folder, args.dataset, epoch),
                              normalize=True, nrow=20)

            fake = netG(fixed_noise).detach()
            vutils.save_image(fake, '%s/presgan_%s_faFake_epoch_%03d.png' % (args.results_folder, args.dataset, epoch),
                              normalize=True, nrow=20)

        if epoch % args.save_ckpt_every == 0:
            #torch.save(netG.state_dict(), os.path.join(args.results_folder, 'netG_presgan_%s_epoch_%s.pth'%(args.dataset, epoch)))

            #torch.save(netG.state_dict(),
            #           os.path.join(args.results_folder, 'netG_presgan_%s_epoch_%s.pth' % (args.dataset, epoch)))

            #asdfasfsdfsafsdfs

            torch.save(netG.state_dict(),
                       os.path.join(args.results_folder, 'neNetG_presgan_%s_epoch_%s.pth' % (args.dataset, epoch)))

            #torch.save(netG.state_dict(),
            #           os.path.join(args.results_folder, 'netG_presgan_%s_epoch_%s.pth' % (args.dataset, epoch)))

            #torch.save(netG.state_dict(),
            #           os.path.join(args.results_folder, 'netG_presgan_%s_epoch_%s.pth' % (args.dataset, epoch)))

            torch.save(log_sigma, os.path.join(args.results_folder, 'log_sigma_%s_%s.pth'%(args.dataset, epoch)))
            
