import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils

# import os
# import seaborn as sns

import pickle
import math

import utils
# import hmc

# import hmc
import hmc2

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
    for epoch in range(1, args.epochs + 1):
        for i in range(0, len(X_training), args.batchSize):
            netD.zero_grad()
            stop = min(args.batchSize, len(X_training[i:]))
            real_cpu = X_training[i:i + stop].to(device)

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
                print(
                    'Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
                    % (epoch, args.epochs, i, len(X_training), errD.data, errG.data, D_x, D_G_z1, D_G_z2))

        print('*' * 100)
        print('End of epoch {}'.format(epoch))
        print('*' * 100)

        if epoch % args.save_imgs_every == 0:
            fake = netG(fixed_noise).detach()
            vutils.save_image(fake, '%s/dcgan_%s_fake_epoch_%03d.png' % (args.results_folder, args.dataset, epoch),
                              normalize=True, nrow=20)

        if epoch % args.save_ckpt_every == 0:
            torch.save(netG.state_dict(),
                       os.path.join(args.results_folder, 'netG_dcgan_%s_epoch_%s.pth' % (args.dataset, epoch)))


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

    bsz = args.batchSize

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
            #if y[i] == 2:

            #if y[i] == 2:
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

