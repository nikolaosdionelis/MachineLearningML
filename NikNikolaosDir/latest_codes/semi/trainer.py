import os
import numpy as np
from time import time

from math import ceil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import grad

from collections import deque

from utils import log_sum_exp, tensor2Var, np2Var

torch.backends.cudnn.benchmark = True

def create_noise(*size):
    return torch.FloatTensor(*size).uniform_(0, 1)

def pullaway_loss(x1):
    norm_x1 = F.normalize(x1)

    N = x1.size(0)
    cosine_similarity_matrix = torch.matmul(norm_x1, norm_x1.t())

    mask = torch.ones(cosine_similarity_matrix.size()) - torch.diag(torch.ones(N))
    mask_v = tensor2Var(mask)

    cosine_similarity_matrix = (cosine_similarity_matrix * mask_v) ** 2

    return cosine_similarity_matrix.sum() / (N * (N - 1))

def pullaway_loss_lp(x1, p=2):
    dist = torch.norm(x1[:, None] - x1, dim=2, p=p)

    dist = dist / dist.max()

    N = x1.size(0)

    mask = torch.ones(dist.size()) - torch.diag(torch.ones(N))
    mask_v = tensor2Var(mask)

    dist = dist * mask_v

    return dist.sum() / (N * (N - 1))

def calc_gradient_penalty(net, real_data, fake_data):
    alpha = torch.FloatTensor(real_data.size(0), 1).uniform_(0, 1)

    alpha = tensor2Var(alpha)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates.requires_grad_(True)

    disc_interpolates = net(interpolates, 'critic')

    ones = tensor2Var(torch.ones(disc_interpolates.size()))

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    while len(gradients.size()) > 1:
        gradients = gradients.norm(2, dim=(len(gradients.size()) - 1))

    gradient_penalty = ((gradients - 1.0) ** 2).mean()

    return gradient_penalty

def set_require_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad


class percent_scheduler(object):
    def __init__(self, max_value, max_epoch, cycle_epoch, now_epoch):
        self.max_value = max_value
        self.max_epoch = max_epoch
        self.cycle_epoch = cycle_epoch
        self.now_epoch = now_epoch

        self.alpha = max_value

        # self.threhold = max_epoch / 5
        self.threhold = max_epoch / 10

        self.min_value = 0.1

    def update(self):

        # self.alpha = self.max_value + (self.max_value - self.min_value) * \
        #     (np.exp(-(self.now_epoch / self.threhold))) * \
        #     np.cos(self.now_epoch * np.pi / self.cycle_epoch)
        self.alpha = self.max_value + (self.max_value - self.min_value) * \
            np.cos(self.now_epoch * np.pi / self.cycle_epoch) / (np.ceil((
                self.now_epoch + 1e-8) / self.threhold))

        self.alpha = min(self.max_value, self.alpha)
        self.alpha = max(self.min_value, self.alpha)
        self.now_epoch += 1

    def update_dir(self, _dir):
        self.alpha += 0.01 * _dir
        self.alpha = min(self.max_value, self.alpha)
        self.alpha = max(self.min_value, self.alpha)
        self.now_epoch += 1

class Trainer():

    def __init__(self, trainer_dict):
        self.args = trainer_dict['args']
        self.logger = trainer_dict['logger']

        args = self.args

        if args.dataset == 'mnist':
            import mnist_model
            classifier_model = mnist_model.WN_Classifier
            gen_model = mnist_model.WN_Generator

            # if args.use_consistency:
                # dis_model = mnist_model.feature_Discriminator_drop
            # else:
            dis_model = mnist_model.feature_Discriminator

            feature_size = 250
        elif args.dataset == 'svhn' or args.dataset == 'cifar':
            import cnn_model
            classifier_model = cnn_model.WN_Classifier
            gen_model = cnn_model.Generator

            # if args.use_consistency:
                # dis_model = cnn_model.feature_Discriminator_drop
            # else:
            dis_model = cnn_model.feature_Discriminator

            feature_size = 128 if args.dataset == 'svhn' else 192
        else:
            raise NotImplementedError

        self.classifier = classifier_model(args, feature_size)
        self.gen = gen_model(args)

        self.dis = dis_model(args, feature_size)

        self.logger.info(self.gen)
        self.logger.info(self.dis)
        self.logger.info(self.classifier)


        self.gan_checkpoint = os.path.join(args.model_folder, 'gan_checkpoint.pt')
        self.classifier_checkpoint = os.path.join(args.model_folder, 'classifier_checkpoint.pt')

        # load the models
        if args.gan_checkpoint != '':
            checkpoint = torch.load(args.gan_checkpoint, map_location=lambda storage, loc: storage)

            self.gen.load_state_dict(checkpoint['gen_state_dict'])
            self.dis.load_state_dict(checkpoint['dis_state_dict'])
     
        if args.classifier_checkpoint != '':
            checkpoint = torch.load(args.classifier_checkpoint, map_location=lambda storage, loc: storage)

            # model_dict = self.classifier.state_dict()

            # for key in list(checkpoint['classifier_state_dict'].keys()):
            #     if key.startswith('module.out_class'):
            #         checkpoint['classifier_state_dict'].pop(key)

            # params_dict = {k: v for k, v in checkpoint['classifier_state_dict'].items() if k in model_dict}
            
            # model_dict.update(params_dict)

            # self.classifier.load_state_dict(model_dict)

            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])

        if torch.cuda.is_available():
            self.dis.cuda()
            self.gen.cuda()
            self.classifier.cuda()

        # set up the optimizers
        self.classifier_opt = optim.Adam(self.classifier.parameters(), lr=args.classifier_lr, 
            betas=(0.5, 0.9999))
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=args.gen_lr, betas=(0.5, 0.9))

        self.dis_opt = optim.Adam(self.dis.parameters(), lr=args.dis_lr, betas=(0.5, 0.9))

        self.schedual_c = optim.lr_scheduler.CosineAnnealingLR(self.classifier_opt, 10, args.classifier_lr/100)
        self.schedual_g = optim.lr_scheduler.CosineAnnealingLR(self.gen_opt, 10, args.gen_lr/100)
        self.schedual_d = optim.lr_scheduler.CosineAnnealingLR(self.dis_opt, 10, args.dis_lr/100)

        self.ps = percent_scheduler(args.alpha, args.max_epochs, 20, 0)

        self.alpha = args.alpha

        self.total_iter = 0

        self.record_file = '%s_record%s.txt' % (args.dataset, args.record_file_affix)

        self.mode = 1     # if mode == 0 => use true dist. seeking, else => diff dist. seeking  

        if not os.path.exists(self.record_file):
            with open(self.record_file, 'w') as f:
                f.write('########################################################################\n')
                f.write('#####                                                              #####\n')
                f.write('#####This file is to record the results of all previous experiments#####\n')
                f.write('#####                                                              #####\n')
                f.write('########################################################################\n')

    def load_gan(self, gen_checkpoint, dis_checkpoint):
        self.dis.load(dis_checkpoint)
        self.gen.load(gen_checkpoint)

    def load_classifier(self, classifier_checkpoint):
        self.classifier.load(classifier_checkpoint)

    # train the generator
    def train_g(self):
        args = self.args
        set_require_grad(self.dis, False)

        noise = create_noise(args.train_batch_size, args.noise_size)
        noise = tensor2Var(noise)

        gen_data = self.gen(noise)

        # get the feature of generated data
        gen_data = self.classifier(gen_data, 'feat')

        pullaway = pullaway_loss(gen_data)

        gen_out = self.dis(gen_data, 'critic')

        gen_loss = -gen_out.mean()

        g_loss = gen_loss + args.lambda_p * pullaway
        
        self.gen_opt.zero_grad()
        g_loss.backward()
        self.gen_opt.step()

        return gen_loss.cpu().item()

    # train the discriminator / critic
    def train_d(self, p_d, p_d_bar):
        args = self.args

        set_require_grad(self.dis, requires_grad=True)
        set_require_grad(self.classifier, requires_grad=False)

        j = 0

        # train discriminator multiples times per generator iteration
        while j < args.iter_c:
            j += 1

            if self.mode == 0:
                true_data, _ = p_d.next()
                true_data = tensor2Var(true_data)
                true_data = self.classifier(true_data, 'feat')

                noise = create_noise(true_data.size(0), args.noise_size)
                noise = tensor2Var(noise)

                gen_data = self.gen(noise).detach()

                gen_data = self.classifier(gen_data, 'feat')

                true_data_out = self.dis(true_data, 'critic')
                gen_data_out = self.dis(gen_data, 'critic')
     
                dis_loss = -true_data_out.mean() + gen_data_out.mean()

                d_loss = dis_loss + \
                    args.lambda_g * calc_gradient_penalty(self.dis, true_data, gen_data)

            else:
                true_data_bar = p_d_bar.sample_feat(self.classifier)
                true_data, _ = p_d.next()
                true_data = tensor2Var(true_data)
                true_data = self.classifier(true_data, 'feat')

                noise = create_noise(true_data.size(0), args.noise_size)
                noise = tensor2Var(noise)

                gen_data = self.gen(noise).detach()

                gen_data = self.classifier(gen_data, 'feat')

                true_data_size = int(true_data.size(0) * self.ps.alpha)

                gen_size = true_data.size(0) - true_data_size

                # concatenate true and gen data
                true_gen_data = torch.cat(
                    [true_data[:true_data_size], gen_data[:gen_size]], 
                    dim=0)

                true_data_bar_out = self.dis(true_data_bar, 'critic')
                true_gen_data_out = self.dis(true_gen_data, 'critic')
     
                dis_loss = -true_data_bar_out.mean() + true_gen_data_out.mean()

                d_loss = dis_loss + \
                    args.lambda_g * calc_gradient_penalty(self.dis, true_data_bar, true_gen_data)

            self.dis_opt.zero_grad()
            d_loss.backward()
            self.dis_opt.step()  

        return -dis_loss.cpu().item()

    # train the classifier
    def train_c(self, labeled_loader, unlabeled_loader):
        args = self.args
        set_require_grad(self.classifier, requires_grad=True)
        # standard classification loss
        lab_data, lab_labels = labeled_loader.next()
        lab_data, lab_labels = tensor2Var(lab_data), tensor2Var(lab_labels)

        lab_labels = lab_labels.view(-1)
        unl_data, _ = unlabeled_loader.next()
        unl_data = tensor2Var(unl_data)

        noise = create_noise(unl_data.size(0), args.noise_size)
        noise = tensor2Var(noise)

        gen_data = self.gen(noise).detach()

        lab_logits = self.classifier(lab_data, 'class')
        unl_logits = self.classifier(unl_data, 'class')
        gen_logits = self.classifier(gen_data, 'class')

        lab_loss = F.cross_entropy(lab_logits, lab_labels)

        unl_logsumexp = log_sum_exp(unl_logits)
        gen_logsumexp = log_sum_exp(gen_logits)

        unl_acc = torch.mean(torch.sigmoid(unl_logsumexp.detach()).gt(0.5).float())
        gen_acc = torch.mean(torch.sigmoid(gen_logsumexp.detach()).lt(0.5).float())

        # This is the typical GAN cost, where sumexp(logits) is seen as the input to the sigmoid
        true_loss = - 0.5 * torch.mean(unl_logsumexp) + 0.5 * torch.mean(F.softplus(unl_logsumexp))
        fake_loss = 0.5 * torch.mean(F.softplus(gen_logsumexp))

        # max_unl_acc = torch.mean(unl_logits.max(1)[0].detach().gt(0.0).float())
        # max_gen_acc = torch.mean(gen_logits.max(1)[0].detach().gt(0.0).float())

        unl_prob = F.softmax(unl_logits, dim=1)

        entropy = -(unl_prob * torch.log(unl_prob + 1e-8)).sum(1).mean()

        unl_loss = true_loss + fake_loss
         
        c_loss = lab_loss + args.lambda_gan * unl_loss + args.lambda_e * entropy

        if args.lambda_consistency > 0:
            unl_logits_2 = self.classifier(unl_data, 'class')
            unl_prob_2 = F.softmax(unl_logits_2, dim=1)
            consistency_loss = ((unl_prob - unl_prob_2) ** 2).mean()

            c_loss += args.lambda_consistency * consistency_loss

            if self.total_iter % 1000 == 0:
                print(consistency_loss)

        self.classifier_opt.zero_grad()
        c_loss.backward()
        self.classifier_opt.step()

        return lab_loss.cpu().item(), unl_loss.cpu().item(), entropy.cpu().item()

    def eval(self, data_loader):
        self.gen.eval()
        self.dis.eval()
        self.classifier.eval()

        loss, incorrect, cnt = 0, 0, 0
        total_num = 0
        max_unl_acc, max_gen_acc = 0, 0
        with torch.no_grad():
            for i, (data, labels) in enumerate(data_loader.get_iter()):
                data, labels = tensor2Var(data), tensor2Var(labels)

                noise = create_noise(data.size(0), self.args.noise_size)
                noise = tensor2Var(noise)

                gen_data = self.gen(noise).detach()
                gen_logits = self.classifier(gen_data, 'class')

                pred_logits = self.classifier(data, 'class')
                labels = labels.view(-1)
                loss += F.cross_entropy(pred_logits, labels).item() * data.size(0)
                cnt += 1
                total_num += data.size(0)
                incorrect += torch.ne(torch.max(pred_logits, 1)[1], labels).float().sum().item()

                max_unl_acc += torch.sum(pred_logits.max(1)[0].detach().gt(0.0).float()).item()
                max_gen_acc += torch.sum(gen_logits.max(1)[0].detach().lt(0.0).float()).item()

        return loss / total_num, incorrect, total_num, max_unl_acc / total_num, max_gen_acc / total_num

    def param_init_dnn(self, unlabeled_loader):
        def func_gen(flag):
            def func(m):
                if hasattr(m, 'init_mode'):
                    setattr(m, 'init_mode', flag)
            return func

        images = []
        for i in range(ceil(500 / self.args.train_batch_size)):
            unl_images, _ = unlabeled_loader.next()
            images.append(unl_images)
        images = torch.cat(images, 0)

        self.classifier.apply(func_gen(True))
        logits = self.classifier(tensor2Var(images))
        self.classifier.apply(func_gen(False))

    def param_init_cnn(self, labeled_loader):
        def func_gen(flag):
            def func(m):
                if hasattr(m, 'init_mode'):
                    setattr(m, 'init_mode', flag)
            return func

        images = []
        for i in range(ceil(500 / self.args.train_batch_size)):
            lab_images, _ = labeled_loader.next()
            images.append(lab_images)
        images = torch.cat(images, 0)

        self.gen.apply(func_gen(True))
        noise = tensor2Var(torch.Tensor(images.size(0), self.args.noise_size).uniform_())
        gen_images = self.gen(noise)
        self.gen.apply(func_gen(False))

        self.classifier.apply(func_gen(True))
        logits = self.classifier(tensor2Var(images))
        self.classifier.apply(func_gen(False))

    def visualize(self):
        self.gen.eval()
        self.dis.eval()

        vis_size = 100
        noise = create_noise(vis_size, self.args.noise_size)
        with torch.no_grad():
            noise = tensor2Var(noise)
            gen_images = self.gen(noise)

        save_path = os.path.join(self.args.log_folder, 'gen_images_%d.png' % self.total_iter)

        if self.args.dataset == 'mnist':
            gen_images = gen_images.view(-1, self.args.n_channels, self.args.image_size, self.args.image_size)
        elif self.args.dataset == 'svhn' or self.args.dataset == 'cifar':
            gen_images = gen_images * 0.5 + 0.5
        else:
            raise NotImplementedError

        vutils.save_image(gen_images.data.cpu(), save_path, nrow=10)

    def compute_update_dir(self, p_d):
        self.gen.eval()
        self.dis.eval()
        self.classifier.eval()
        i = 0

        _dir = 0
        with torch.no_grad():
            while i < 10:
                true_data, _ = p_d.next()
                true_data = tensor2Var(true_data)

                noise = create_noise(true_data.size(0), self.args.noise_size)
                noise = tensor2Var(noise)

                gen_data = self.gen(noise).detach()

                true_out = self.dis(self.classifier(true_data, 'feat'), 'critic')
                gen_out = self.dis(self.classifier(gen_data, 'feat'), 'critic')

                _dir += (-true_out.mean() + gen_out.mean() - 1 * 2 * self.ps.alpha).item()

                # self.ps.update_dir(-1 * ((-real_output_c.mean() + fake_output_c.mean()).item() - \
                #     0.5 * 2 * self.ps.alpha))
                i += 1

            print(true_out.mean().item(), gen_out.mean().item())

        self.gen.train()
        self.dis.train()
        self.classifier.train()

        return -1 * np.sign(_dir)
        # return -1 * _dir / i

    def train(self, tr_data_dict):
        args = self.args

        labeled_loader = tr_data_dict['labeled_loader']
        unlabeled_loader = tr_data_dict['unlabeled_loader']
        p_d = tr_data_dict['p_d']
        dev_loader = tr_data_dict['dev_loader']
        p_d_bar = tr_data_dict['p_d_bar']
        p_d_2 = tr_data_dict['p_d_2']


        # init the weight of classifier
        if args.classifier_checkpoint == '':
            if args.dataset == 'mnist':
                self.param_init_dnn(unlabeled_loader)
            elif args.dataset == 'svhn' or args.dataset == 'cifar':
                self.param_init_cnn(labeled_loader)

        
        ######################################################################
        ### start training

        batch_per_epoch = int((len(unlabeled_loader) + args.train_batch_size - 1) / args.train_batch_size)
        total_iter = 0

        best_loss = 1e8
        best_err = 1e8
        best_err_per = 1e8
        begin_time = time()

        while True:

            # if total_iter % 10000 == 0:
              #   self.mode += 1
               #  self.mode %= 2

            if total_iter % batch_per_epoch == 0:
                
                epoch = total_iter // batch_per_epoch
                if epoch >= args.max_epochs:
                    break

                epoch_ratio = float(epoch) / float(args.max_epochs)

                self.dis_opt.param_groups[0]['lr'] = \
                    max(4*args.min_lr, args.dis_lr * max(0., min(3. * (1. - epoch_ratio), 1.)))
                    
                self.gen_opt.param_groups[0]['lr'] = \
                    max(args.min_lr, args.gen_lr * max(0., min(3. * (1. - epoch_ratio), 1.)))

                self.classifier_opt.param_groups[0]['lr'] = \
                    max(args.min_lr, args.classifier_lr * max(0., min(3. * (1. - epoch_ratio), 1.)))
                # self.schedual_c.step()
                # self.schedual_d.step()
                # self.schedual_g.step()

            self.dis.train()
            self.gen.train()
            self.classifier.train()

            ##################
            ## train the model
            
            # train all the networks
            dis_dist, gen_critic = 0, 0
            if total_iter % args.update_freq == 0:
                dis_dist = self.train_d(p_d, p_d_bar)
           
                gen_critic = self.train_g()
            lab_loss, unl_loss, entropy = self.train_c(labeled_loader, unlabeled_loader)

            if total_iter % args.eval_period == 0:
                tr_loss, tr_incorrect, tr_total_num, _, _ = self.eval(labeled_loader)
                tr_incorrect_per = 100 * tr_incorrect / tr_total_num

                loss, incorrect, total_num, max_unl_acc, max_gen_acc = self.eval(dev_loader)
                incorrect_per = 100 * incorrect / total_num
                print(time() - begin_time)
                best_err_per = min(incorrect_per, best_err_per)

                if incorrect < best_err:
                    best_err = incorrect
                    with open(self.record_file, 'r') as f:
                        record = f.readlines()

                        if total_iter != 0:
                            record.pop() 

                        record.append('%s, %s, %d, %d, %.2f%%\n' % (args.save_folder, args.name, total_iter, 
                            best_err, best_err_per))

                    with open(self.record_file, 'w') as f:
                        f.writelines(record)

                    best_loss = loss

                    stop = 0

                    if args.save_gan:
                        save_dict = {'total_iter': total_iter, 
                            'gen_state_dict': self.gen.state_dict(), 
                            'dis_state_dict': self.dis.state_dict(),
                            'gen_opt': self.gen_opt.state_dict(),
                            'dis_opt': self.dis_opt.state_dict()}

                        torch.save(save_dict, self.gan_checkpoint)

                    if args.save_classifier:
                        save_dict = {'total_iter': total_iter, 
                            'classifier_state_dict': self.classifier.state_dict(), 
                            'classifier_opt': self.classifier_opt.state_dict()}

                        torch.save(save_dict, self.classifier_checkpoint)

                self.logger.info('epoch: %d, iter: %d, alpha: %.3f, mode: %d, spent: %.3f s' % (epoch, 
                    total_iter, self.ps.alpha, self.mode, time() - begin_time))
                self.logger.info('[train] loss: %.4f, incorrect: %d / %d (%.4f %%)' % (tr_loss, tr_incorrect, 
                    tr_total_num, tr_incorrect_per))
                self.logger.info('[dev] loss: %.4f, incorrect: %d (%d) / %d (%.4f%%, %.4f%%)' % (loss, 
                    incorrect, best_err, total_num, incorrect_per, best_err_per))
                self.logger.info('%s: %.4f, %s: %.4f, %s: %.4f' % ('dis_dist', dis_dist, 
                    'max_unl_acc', max_unl_acc, 'max_gen_acc', max_gen_acc))
                self.logger.info('--------')

                begin_time = time()

            if total_iter % args.visual_period == 0:
                self.visualize()

            total_iter += 1
            self.total_iter += 1

        return best_err, best_err_per
