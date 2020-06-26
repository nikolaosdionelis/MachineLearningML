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

import torchvision as tv

import matplotlib.pyplot as plt

import sklearn
import sklearn.manifold

def create_noise(*size):
    return torch.FloatTensor(*size).normal_(0, 1)

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
    alpha = torch.FloatTensor(real_data.size(0), 1, 1, 1).uniform_(0, 1)

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
            classifier_model = mnist_model.Classifier
            gen_model = mnist_model.feature_Generator

            dis_model = mnist_model.feature_Discriminator

            # if args.todo == 'ae':
            #     vae_model = mnist_model.AE
            # else:
            #     vae_model = mnist_model.VAE

            if args.feature_extractor == 'ae':
                vae_model = mnist_model.AE
            else:
                vae_model = mnist_model.VAE
            feature_size = args.feature_size

        elif args.dataset == 'svhn' or args.dataset == 'cifar':
            import cnn_model
            classifier_model = cnn_model.Classifier
            gen_model = cnn_model.feature_Generator
            # gen_model = cnn_model.Generator
            dis_model = cnn_model.feature_Discriminator

            # if args.todo == 'ae':
            #     vae_model = cnn_model.AE
            # else:                
            #     vae_model = cnn_model.VAE
            if args.feature_extractor == 'ae':
                vae_model = cnn_model.AE
            else:
                vae_model = cnn_model.VAE
            feature_size = args.feature_size
        else:
            raise NotImplementedError

        self.feature_size = feature_size

        self.classifier = classifier_model(args, feature_size)
        self.gen = gen_model(args, feature_size)
        # self.gen = gen_model(args)

        self.dis = dis_model(args, feature_size)

        self.vae = vae_model(args, feature_size)

        self.logger.info(self.gen)
        self.logger.info(self.dis)
        self.logger.info(self.classifier)
        self.logger.info(self.vae)

        self.gan_checkpoint = os.path.join(args.model_folder, 'gan_checkpoint.pt')
        self.classifier_checkpoint = os.path.join(args.model_folder, 'classifier_checkpoint.pt')
        self.vae_checkpoint = os.path.join(args.model_folder, 'vae_checkpoint.pt')

        # load the models
        if args.gan_checkpoint != '':
            self.logger.info('load gan...')
            checkpoint = torch.load(args.gan_checkpoint, map_location=lambda storage, loc: storage)

            self.gen.load_state_dict(checkpoint['gen_state_dict'])
            self.dis.load_state_dict(checkpoint['dis_state_dict'])
     
        if args.classifier_checkpoint != '':
            self.logger.info('load classifier...')
            checkpoint = torch.load(args.classifier_checkpoint, map_location=lambda storage, loc: storage)

            # model_dict = self.classifier.state_dict()

            # for key in list(checkpoint['classifier_state_dict'].keys()):
            #     if key.startswith('module.out_class'):
            #         checkpoint['classifier_state_dict'].pop(key)

            # params_dict = {k: v for k, v in checkpoint['classifier_state_dict'].items() if k in model_dict}
            
            # model_dict.update(params_dict)

            # self.classifier.load_state_dict(model_dict)

            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])

        if args.vae_checkpoint != '':
            self.logger.info('load vae...')
            checkpoint = torch.load(args.vae_checkpoint, map_location=lambda storage, loc: storage)

            self.vae.load_state_dict(checkpoint['vae_state_dict'])


        if torch.cuda.is_available():
            self.dis.cuda()
            self.gen.cuda()
            self.classifier.cuda()
            self.vae.cuda()

        # set up the optimizers
        self.classifier_opt = optim.Adam(self.classifier.parameters(), lr=args.classifier_lr, 
            betas=(0.5, 0.9999))
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=args.gen_lr, betas=(0.5, 0.9))

        self.dis_opt = optim.Adam(self.dis.parameters(), lr=args.dis_lr, betas=(0.5, 0.9))

        self.vae_opt = optim.Adam(self.vae.parameters(), lr=args.dis_lr, betas=(0.5, 0.9))

        self.schedual_c = optim.lr_scheduler.CosineAnnealingLR(self.classifier_opt, 10, args.classifier_lr/100)
        self.schedual_g = optim.lr_scheduler.CosineAnnealingLR(self.gen_opt, 10, args.gen_lr/100)
        self.schedual_d = optim.lr_scheduler.CosineAnnealingLR(self.dis_opt, 10, args.dis_lr/100)

        self.ps = percent_scheduler(args.alpha, args.max_epochs, 20, 0)

        self.alpha = args.alpha

        self.total_iter = 0

        self.sample_size = 64

    def load_gan(self, gen_checkpoint, dis_checkpoint):
        self.dis.load(dis_checkpoint)
        self.gen.load(gen_checkpoint)

    def load_classifier(self, classifier_checkpoint):
        self.classifier.load(classifier_checkpoint)

    def load_vae(self, vae_checkpoint):
        self.vae.load(vae_checkpoint)

    # train the generator
    def train_g(self, p_d_2):
        args = self.args
        set_require_grad(self.dis, False)

        true_data, _ = p_d_2.next()
        true_data = tensor2Var(true_data)

        # noise = create_noise(args.train_batch_size, args.noise_size)
        # noise = tensor2Var(noise)

        true_data = self.vae.get_features(true_data)

        gen_data = self.gen(true_data)

        # self.vae.eval()

        # gen_data = self.vae.get_features(gen_data)

        # self.vae.train()

        gen_out = self.dis(gen_data, 'critic')

        feat_loss = ((gen_data - true_data).view(gen_data.shape[0], -1) ** 2).mean()

        gen_loss = -gen_out.mean() + args.lambda_feat * feat_loss

        g_loss = gen_loss
        
        self.gen_opt.zero_grad()
        g_loss.backward()
        self.gen_opt.step()

        return gen_loss.cpu().item(), feat_loss.cpu().item()

    # train the discriminator / critic
    def train_d(self, p_d, p_d_bar):
        args = self.args

        set_require_grad(self.dis, requires_grad=True)

        j = 0

        # train discriminator multiples times per generator iteration
        while j < args.iter_c:
            j += 1
            
            true_data_bar = p_d_bar.sample_feat(self.vae.get_features)
            true_data, _ = p_d.next()
            true_data = tensor2Var(true_data)

            true_data = self.vae.get_features(true_data)

            # noise = create_noise(true_data.size(0), args.noise_size)
            # noise = tensor2Var(noise)

            gen_data = self.gen(true_data).detach()

            # self.vae.eval()

            # gen_data = self.vae.get_features(gen_data)

            # self.vae.train()

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
    def train_c(self, train_loader, semi_weight):
        args = self.args
        # set_require_grad(self.classifier, requires_grad=True)
        # standard classification loss
        lab_data, lab_labels = train_loader.next()
        lab_data, lab_labels = tensor2Var(lab_data), tensor2Var(lab_labels)

        noise = create_noise(lab_data.size(0), args.noise_size)
        noise = tensor2Var(noise)

        gen_data = self.gen(noise).detach()

        lab_logits = self.classifier(lab_data, 'class')
        gen_logits = self.classifier(gen_data, 'class')

        lab_loss = F.cross_entropy(lab_logits, lab_labels)

        gen_prob = F.softmax(gen_logits, dim=1)

        entropy = -(gen_prob * torch.log(gen_prob + 1e-8)).sum(1).mean()

        c_loss = lab_loss + semi_weight * entropy

        self.classifier_opt.zero_grad()
        c_loss.backward()
        self.classifier_opt.step()

        return lab_loss.cpu().item(), entropy.cpu().item(), c_loss.cpu().item()

    def eval(self, data_loader):
        self.gen.eval()
        self.dis.eval()
        self.classifier.eval()

        loss, incorrect, cnt = 0, 0, 0
        total_num = 0
        with torch.no_grad():
            for i, (data, labels) in enumerate(data_loader.get_iter()):
                data, labels = tensor2Var(data), tensor2Var(labels)

                pred_logits = self.classifier(data, 'class')
                loss += F.cross_entropy(pred_logits, labels).item() * data.size(0)
                cnt += 1
                total_num += data.size(0)
                incorrect += torch.ne(torch.max(pred_logits, 1)[1], labels).float().sum().item()

        return loss / total_num, incorrect, total_num

    def param_init_dnn(self, loader):
        def func_gen(flag):
            def func(m):
                if hasattr(m, 'init_mode'):
                    setattr(m, 'init_mode', flag)
            return func

        images = []
        for i in range(ceil(500 / self.args.train_batch_size)):
            unl_images, _ = loader.next()
            images.append(unl_images)
        images = torch.cat(images, 0)

        self.classifier.apply(func_gen(True))
        logits = self.classifier(tensor2Var(images))
        self.classifier.apply(func_gen(False))

    def param_init_cnn(self, loader):
        def func_gen(flag):
            def func(m):
                if hasattr(m, 'init_mode'):
                    setattr(m, 'init_mode', flag)
            return func

        images = []
        for i in range(ceil(500 / self.args.train_batch_size)):
            lab_images, _ = loader.next()
            images.append(lab_images)
        images = torch.cat(images, 0)

        self.gen.apply(func_gen(True))
        noise = tensor2Var(torch.Tensor(images.size(0), self.args.noise_size).uniform_())
        gen_images = self.gen(noise)
        self.gen.apply(func_gen(False))

        # self.classifier.apply(func_gen(True))
        # logits = self.classifier(tensor2Var(images))
        # self.classifier.apply(func_gen(False))

    def visualize(self, train_loader):
        self.gen.eval()
        self.dis.eval()
        self.vae.eval()

        vis_size = 100
        for i, (data, _) in enumerate(train_loader.get_iter()):
            data = tensor2Var(data)
            with torch.no_grad():
                feat = self.vae.get_features(data)
                gen_images = self.gen(feat)
                gen_images = self.vae.decode(gen_images)

            break

        save_path = os.path.join(self.args.log_folder, '%d_gen_images.png' % self.total_iter)

        if self.args.dataset == 'mnist':
            # gen_images = gen_images.view(-1, self.args.n_channels, self.args.image_size, self.args.image_size)
            gen_images = gen_images * 0.5 + 0.5
            # print(gen_images.shape)
        elif self.args.dataset == 'svhn' or self.args.dataset == 'cifar':
            gen_images = gen_images * 0.5 + 0.5
        else:
            raise NotImplementedError

        vutils.save_image(gen_images.data.cpu(), save_path, nrow=10)

        save_path = os.path.join(self.args.log_folder, '%d_ori_images.png' % self.total_iter)

        vutils.save_image(data.data.cpu() * 0.5 + 0.5, save_path, nrow=10)
        self.vae.train()

    def visualize_embedding(self, p_d_2):
        self.gen.eval()
        self.dis.eval()
        self.vae.eval()

        vis_size = 200

        true_emb = []

        gen_emb = []

        cum_size = 0

        with torch.no_grad():
            for i, (data, _) in enumerate(p_d_2.get_iter()):
                data = tensor2Var(data)

                feat = self.vae.get_features(data)

                true_emb.append(feat.squeeze().cpu().numpy())

                gen_feat = self.gen(feat)

                gen_emb.append(gen_feat.squeeze().cpu().numpy())

                # gen_emb.append(gen_data.squeeze().cpu().numpy())

                cum_size += data.shape[0]

                if cum_size >= vis_size:
                    break


        true_emb = np.vstack(true_emb)
        gen_emb = np.vstack(gen_emb)

        # print(true_emb.shape, gen_emb.shape)

        tsne = sklearn.manifold.TSNE(2)

        all_emb = np.vstack([true_emb, gen_emb])

        # print(all_emb.shape)

        all_emb = tsne.fit_transform(all_emb)

        size = all_emb.shape[0] // 2
        
        true_emb = all_emb[: size]
        gen_emb = all_emb[size:]

        plt.clf()

        t = plt.scatter(true_emb[:, 0], true_emb[:, 1], label='true data')
        g = plt.scatter(gen_emb[:, 0], gen_emb[:, 1], label='gen data')

        plt.legend([t, g], ['true data', 'gen data'])

        save_path = os.path.join(self.args.log_folder, 'embedding_%d.png' % self.total_iter)

        plt.savefig(save_path)

        self.vae.train()


    def eval_gen(self, gen_num):
        self.gen.eval()
        self.dis.eval()
        self.classifier.eval()

        loss = 0
        total_num = 0
        batch_size = self.args.train_batch_size
        with torch.no_grad():
            while total_num < gen_num:
                if total_num + batch_size > gen_num:
                    batch_size = gen_num - total_num

                noise = create_noise(batch_size, self.args.noise_size)
                noise = tensor2Var(noise)
                gen_images = self.gen(noise)
                gen_logits = self.classifier(gen_images, 'class')
                gen_prob = F.softmax(gen_logits, dim=1)

                loss += -(gen_prob * torch.log(gen_prob + 1e-8)).sum().item()

                total_num += batch_size
        return loss / total_num

    def train_classifier(self, tr_data_dict):
        args = self.args
        set_require_grad(self.dis, requires_grad=False)
        set_require_grad(self.gen, requires_grad=False)
        set_require_grad(self.vae, requires_grad=False)

        self.gen.eval()
        self.dis.eval()
        self.vae.eval()

        train_loader = tr_data_dict['train_loader']
        p_d_2 = tr_data_dict['p_d_2']
        # valid_loader = tr_data_dict['valid_loader']

        total_iter = 0

        best_loss = 1e8
        best_err = 1e8
        best_err_per = 1e8
        begin_time = time()

        stop = 0

        self.visualize_embedding(p_d_2)

        for epoch in range(args.max_epochs):
            epoch_ratio = float(epoch) / float(args.max_epochs)
            # self.classifier_opt.param_groups[0]['lr'] = \
            #     max(args.min_lr, args.classifier_lr * max(0., min(3. * (1. - epoch_ratio), 1.)))

            self.classifier.train()

            for i, (lab_data, lab_labels) in enumerate(train_loader.get_iter()):
                lab_data, lab_labels = tensor2Var(lab_data), tensor2Var(lab_labels)

                noise = create_noise(lab_data.size(0), args.noise_size)
                noise = tensor2Var(noise)

                gen_data = self.gen(noise).detach()

                lab_data = self.vae.get_features(lab_data)
                gen_data = self.vae.get_features(gen_data)

                lab_logits = self.classifier(lab_data)
                gen_logits = self.classifier(gen_data)

                label_true = tensor2Var(torch.ones(lab_data.shape[0]))
                label_gen = tensor2Var(torch.zeros(gen_data.shape[0]))

                pred = torch.cat([lab_logits, gen_logits], dim=0)
                label = torch.cat([label_true, label_gen], dim=0)

                lab_loss = F.binary_cross_entropy(pred, label)

                self.classifier_opt.zero_grad()
                lab_loss.backward()
                self.classifier_opt.step()

            # if epoch % 10:
            #     print(pred.shape)


            if args.save_classifier:
                save_dict = {'total_iter': total_iter, 
                        'classifier_state_dict': self.classifier.state_dict(), 
                        'classifier_opt': self.classifier_opt.state_dict()}
                torch.save(save_dict, self.classifier_checkpoint)
                
            self.logger.info('epoch: %d, iter: %d, spent: %.3f s' % (epoch, total_iter, time() - begin_time))
            self.logger.info('[train] loss: %.4f' % (lab_loss.cpu().item()))

            self.logger.info('--------')

            begin_time = time()

            total_iter += 1
            self.total_iter += 1

    def train_gan(self, tr_data_dict):
        args = self.args
        train_loader = tr_data_dict['train_loader']
        p_d = tr_data_dict['p_d']
        p_d_bar = tr_data_dict['p_d_bar']
        p_d_2 = tr_data_dict['p_d_2']

        batch_per_epoch = int((len(p_d) + args.train_batch_size - 1) / args.train_batch_size)
        total_iter = 0

        best_loss = 1e8
        best_err = 1e8
        best_err_per = 1e8
        begin_time = time()

        while True:
            if total_iter % batch_per_epoch == 0:
                
                epoch = total_iter // batch_per_epoch
                if epoch >= args.max_epochs:
                    break

                epoch_ratio = float(epoch) / float(args.max_epochs)

                # self.dis_opt.param_groups[0]['lr'] = \
                #     max(args.min_lr, args.dis_lr * max(0., min(3. * (1. - epoch_ratio), 1.)))
                    
                # self.gen_opt.param_groups[0]['lr'] = \
                #     max(args.min_lr, args.gen_lr * max(0., min(3. * (1. - epoch_ratio), 1.)))

            self.dis.train()
            self.gen.train()

            self.vae.train()

            ##################
            ## train the model
            
            # train all the networks
            dis_dist = self.train_d(p_d, p_d_bar)

            gen_critic, feat_loss = self.train_g(p_d_2)

            if total_iter % args.eval_period == 0:

                if args.save_gan:
                    save_dict = {'total_iter': total_iter, 
                        'gen_state_dict': self.gen.state_dict(), 
                        'dis_state_dict': self.dis.state_dict(),
                        'gen_opt': self.gen_opt.state_dict(),
                        'dis_opt': self.dis_opt.state_dict()}

                    torch.save(save_dict, self.gan_checkpoint)

                self.logger.info('epoch: %d, iter: %d, alpha: %.3f, spent: %.3f s' % (epoch, 
                    total_iter, self.ps.alpha, time() - begin_time))
             
                self.logger.info('%s: %.4f, %s: %.4f' % ('dis_dist', dis_dist, 'feat_loss', feat_loss))
                self.logger.info('--------')

                begin_time = time()

            if total_iter % args.visual_period == 0:
                self.visualize(train_loader)
                self.visualize_embedding(train_loader)

            if total_iter == 0:
                self.visualize_reconstruction(train_loader, epoch)
                # self.visualize_generation(epoch)

            total_iter += 1
            self.total_iter += 1


    def visualize_reconstruction(self, train_loader, _iter):
        self.vae.eval()

        with torch.no_grad():
            for i, (data, _) in enumerate(train_loader.get_iter()):
                data_v = tensor2Var(data)
                reconstruct, mean, _ = self.vae(data_v)

                # noise_in = tensor2Var(torch.FloatTensor(mean.size()).uniform_(-self.args.beta_1, self.args.beta_1))
                # noise_out = tensor2Var(torch.FloatTensor(mean.size()).uniform_(-self.args.beta_2, self.args.beta_2))

                # reconstruct = self.vae.decode(torch.clamp(mean + noise_in, -1.0, 1.0))

                # reconstruct_out = self.vae.decode(torch.clamp(mean + noise_out, -1.0, 1.0))

                break


        tv.utils.save_image(data[:self.sample_size] * 0.5 + 0.5, 
            os.path.join(self.args.log_folder, '%d_origin.png' % _iter))

        tv.utils.save_image(reconstruct.data[:self.sample_size] * 0.5 + 0.5, 
            os.path.join(self.args.log_folder, '%d_reconstruct.png' % _iter))

        # tv.utils.save_image(reconstruct_out.data[:self.sample_size] * 0.5 + 0.5, 
        #     os.path.join(self.args.log_folder, '%d_reconstruct_out.png' % _iter))

    def visualize_generation(self, _iter):
        self.vae.eval()
        noise = torch.randn(self.sample_size, self.feature_size, 1, 1)

        with torch.no_grad():
            noise_v = tensor2Var(noise)
            output = self.vae.decode(noise_v)

        tv.utils.save_image(output.data * 0.5 + 0.5, 
            os.path.join(self.args.log_folder, 'generation_%d.png' % _iter))

    def train_vae(self, tr_data_dict):
        args = self.args

        train_loader = tr_data_dict['train_loader']

        ######################################################################
        ### start training

        total_iter = 0

        best_loss = 1e8
        best_err = 1e8
        best_err_per = 1e8
        begin_time = time()

        stop = 0

        for epoch in range(args.max_epochs):
            epoch_ratio = float(epoch) / float(args.max_epochs)

            self.vae.train()
            for i, (lab_data, _) in enumerate(train_loader.get_iter()):
                lab_data = tensor2Var(lab_data)

                reconstruct, mean, logvar = self.vae(lab_data)

                # reconstruction_loss = F.binary_cross_entropy(reconstruct, data_v, size_average=True)
                reconstruction_loss = ((reconstruct - lab_data) ** 2).mean()
                kl_div = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)).mean()

                loss = reconstruction_loss + self.args.LAMBDA * kl_div

                self.vae_opt.zero_grad()
                loss.backward()
                self.vae_opt.step()

            if args.save_vae:
                save_dict = {'total_iter': total_iter, 
                    'vae_state_dict': self.vae.state_dict(), 
                    'vae_opt': self.vae_opt.state_dict()}

                torch.save(save_dict, self.vae_checkpoint)


            self.logger.info('epoch: %d, iter: %d, spent: %.3f s' % (epoch, total_iter, time() - begin_time))
            self.logger.info('[train] loss: %.4f, reconst loss: %.4f, kl_div: %.4f' % (loss.cpu().item(), 
                reconstruction_loss.cpu().item(), kl_div.cpu().item()))

            self.logger.info('--------')

            if epoch % 10 == 0:
                self.visualize_reconstruction(train_loader, epoch)
                self.visualize_generation(epoch)

            begin_time = time()

            total_iter += 1
            self.total_iter += 1


    def train_ae(self, tr_data_dict):
        args = self.args

        train_loader = tr_data_dict['train_loader']

        ######################################################################
        ### start training

        total_iter = 0

        best_loss = 1e8
        best_err = 1e8
        best_err_per = 1e8
        begin_time = time()

        stop = 0

        for epoch in range(args.max_epochs):
            epoch_ratio = float(epoch) / float(args.max_epochs)

            self.vae.train()
            for i, (lab_data, _) in enumerate(train_loader.get_iter()):
                lab_data = tensor2Var(lab_data)

                reconstruct, mean, _ = self.vae(lab_data)

                # reconstruction_loss = F.binary_cross_entropy(reconstruct, data_v, size_average=True)
                reconstruction_loss = ((reconstruct - lab_data) ** 2).mean()
                feature_loss = ((mean - 0) ** 2).mean()

                loss = reconstruction_loss + self.args.LAMBDA * feature_loss

                self.vae_opt.zero_grad()
                loss.backward()
                self.vae_opt.step()

            if args.save_vae:
                save_dict = {'total_iter': total_iter, 
                    'vae_state_dict': self.vae.state_dict(), 
                    'vae_opt': self.vae_opt.state_dict()}

                torch.save(save_dict, self.vae_checkpoint)


            self.logger.info('epoch: %d, iter: %d, spent: %.3f s' % (epoch, total_iter, time() - begin_time))
            self.logger.info('[train] loss: %.4f, reconst loss: %.4f, feature_loss: %.4f' % (loss.cpu().item(), 
                reconstruction_loss.cpu().item(), feature_loss.cpu().item()))

            self.logger.info('--------')

            if epoch % 10 == 0:
                self.visualize_reconstruction(train_loader, epoch)
                # self.visualize_generation(epoch)

            begin_time = time()

            total_iter += 1
            self.total_iter += 1

    def train(self, tr_data_dict):
        args = self.args

        train_loader = tr_data_dict['train_loader']

        p_d = tr_data_dict['p_d']
        p_d_bar = tr_data_dict['p_d_bar']
        p_d_2 = tr_data_dict['p_d_2']

        ######################################################################
        ### start training

        # if args.gan_checkpoint == "":
        #     self.param_init_cnn(p_d_2)

        total_iter = 0

        best_loss = 1e8
        best_err = 1e8
        best_err_per = 1e8
        begin_time = time()

        stop = 0

        for epoch in range(args.max_epochs):
            epoch_ratio = float(epoch) / float(args.max_epochs)

            self.dis.train()
            self.gen.train()

            self.vae.train()
            self.classifier.train()

            for i, (lab_data, _) in enumerate(train_loader.get_iter()):
                lab_data = tensor2Var(lab_data)

                # noise = create_noise(args.train_batch_size, args.noise_size)
                # noise = tensor2Var(noise)

                # gen_data = self.gen(noise).detach()


                # lab_feat = self.vae.get_features(lab_data)
                # # gen_feat = self.vae.get_features(gen_data)
                # gen_feat = p_d_bar.sample_feat(self.vae.get_features)

                # lab_logits = self.classifier(lab_feat)
                # gen_logits = self.classifier(gen_feat)

                # label_true = tensor2Var(torch.ones(lab_feat.shape[0]))
                # label_gen = tensor2Var(torch.zeros(gen_feat.shape[0]))

                # pred = torch.cat([lab_logits, gen_logits], dim=0)
                # label = torch.cat([label_true, label_gen], dim=0)

                # lab_loss = F.binary_cross_entropy(pred, label)

                # self.classifier_opt.zero_grad()
                # lab_loss.backward()
                # self.classifier_opt.step()


                reconstruct, mean, logvar = self.vae(lab_data)

                noise_in = tensor2Var(torch.FloatTensor(mean.size()).uniform_(-self.args.beta_1, self.args.beta_1))
                noise_out = tensor2Var(torch.FloatTensor(mean.size()).uniform_(-self.args.beta_2, self.args.beta_2))

                reconstruct = self.vae.decode(torch.clamp(mean + noise_in, -1.0, 1.0))

                reconstruct_out = self.vae.decode(torch.clamp(mean + noise_out, -1.0, 1.0))

                # reconstruct_gen, *_ = self.vae(gen_data)

                # reconstruction_loss = F.binary_cross_entropy(reconstruct, data_v, size_average=True)
                reconstruction_loss = ((reconstruct - lab_data) ** 2).mean()

                # gen_reconstruction_loss = torch.max(
                #     tensor2Var(torch.zeros(gen_data.shape[0])),
                #     1.0 - ((reconstruct_gen - gen_data) ** 2).view(gen_data.shape[0], -1).mean(1)).mean()

                gen_reconstruction_loss = torch.max(
                    tensor2Var(torch.zeros(lab_data.shape[0])),
                    0.1 - ((reconstruct_out - lab_data) ** 2).view(lab_data.shape[0], -1).mean(1)).mean()

                # gen_reconstruction_loss = ((reconstruct_gen - gen_data) ** 2).mean()
                # gen_reconstruction_loss = 0*((reconstruct - lab_data) ** 2).mean()

                feature_loss = ((mean - 0) ** 2).mean()

                loss = reconstruction_loss + self.args.LAMBDA * feature_loss + gen_reconstruction_loss

                # reconstruction_loss = ((reconstruct - lab_data) ** 2).mean()
                # kl_div = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)).mean()

                # loss = reconstruction_loss + self.args.LAMBDA * kl_div + gen_reconstruction_loss

                self.vae_opt.zero_grad()
                loss.backward()
                self.vae_opt.step()

                ##################
                ## train the model
                
                # train all the networks
                # dis_dist = self.train_d(p_d, p_d_bar)

                # gen_critic = self.train_g()

            if args.save_vae:
                save_dict = {'total_iter': total_iter, 
                    'vae_state_dict': self.vae.state_dict(), 
                    'vae_opt': self.vae_opt.state_dict()}

                torch.save(save_dict, self.vae_checkpoint)

                # save_dict = {'total_iter': total_iter, 
                #         'classifier_state_dict': self.classifier.state_dict(), 
                #         'classifier_opt': self.classifier_opt.state_dict()}
                # torch.save(save_dict, self.classifier_checkpoint)


            self.logger.info('epoch: %d, iter: %d, spent: %.3f s' % (epoch, total_iter, time() - begin_time))
            self.logger.info('[train] loss: %.4f, reconst loss: %.4f, feature_loss: %.4f, \
                gen_reconstruction_loss: %.4f' % (loss.cpu().item(), 
                reconstruction_loss.cpu().item(), feature_loss.cpu().item(), gen_reconstruction_loss.cpu().item()))

            # self.logger.info('[train] loss: %.4f, reconst loss: %.4f, kl_div: %.4f, \
            #     gen_reconstruction_loss: %.4f' % (loss.cpu().item(), 
            #     reconstruction_loss.cpu().item(), kl_div.cpu().item(), gen_reconstruction_loss.cpu().item()))

            # self.logger.info('[train] loss: %.4f, reconst loss: %.4f, feature_loss: %.4f' % (loss.cpu().item(), 
            #     reconstruction_loss.cpu().item(), feature_loss.cpu().item()))

            # self.logger.info('%s: %.4f' % ('dis_dist', dis_dist))

            # self.logger.info('classifier loss: %.4f' % lab_loss.cpu().item())

            self.logger.info('--------')

            if epoch % 10 == 0:
                self.visualize_reconstruction(train_loader, epoch)
                # self.visualize()
                # self.visualize_embedding(p_d_2)
                # self.visualize_generation(epoch)

            begin_time = time()

            total_iter += 1
            self.total_iter += 1

    def finetune(self, tr_data_dict):
        args = self.args

        train_loader = tr_data_dict['train_loader']
        p_d = tr_data_dict['p_d']
        ######################################################################
        ### start training

        total_iter = 0

        best_loss = 1e8
        best_err = 1e8
        best_err_per = 1e8
        begin_time = time()

        stop = 0

        # for p in self.vae.conv_feature.parameters():
        #     p.requires_grad = False
        # for p in self.vae.mean_layer.parameters():
        #     p.requires_grad = False
        # for p in self.vae.std_layer.parameters():
        #     p.requires_grad = False

        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.vae_opt, [300, 400, 500], 0.1)

        for epoch in range(args.max_epochs):
            epoch_ratio = float(epoch) / float(args.max_epochs)

            self.vae.train()
            for i, (lab_data, _) in enumerate(train_loader.get_iter()):
                lab_data = tensor2Var(lab_data)

                feat = self.vae.get_features(lab_data).detach()

                reconstruct = self.vae.decode(feat)

                gen_feat = self.gen(feat).detach()

                reconstruct_gen = self.vae.decode(gen_feat)

                gen_reconstruction_loss = torch.max(
                    tensor2Var(torch.zeros(gen_feat.shape[0])),
                    args.threshold - ((reconstruct_gen - lab_data) ** 2).view(gen_feat.shape[0], -1).mean(1)).mean()

                # gen_reconstruction_loss = ((reconstruct_gen - lab_data) ** 2).mean()
                # reconstruction_loss = F.binary_cross_entropy(reconstruct, data_v, size_average=True)
                reconstruction_loss = ((reconstruct - lab_data) ** 2).mean()

                loss = reconstruction_loss + args.lambda_out * gen_reconstruction_loss

                self.vae_opt.zero_grad()
                loss.backward()
                self.vae_opt.step()

            scheduler.step()

            if args.save_vae:
                save_dict = {'total_iter': total_iter, 
                    'vae_state_dict': self.vae.state_dict(), 
                    'vae_opt': self.vae_opt.state_dict()}

                torch.save(save_dict, self.vae_checkpoint)


            self.logger.info('epoch: %d, iter: %d, spent: %.3f s' % (epoch, total_iter, time() - begin_time))
            self.logger.info('[train] loss: %.4f, reconst loss: %.4f, gen_reconst_loss: %.4f' % (loss.cpu().item(), 
                reconstruction_loss.cpu().item(), gen_reconstruction_loss.cpu().item()))

            self.logger.info('--------')

            if epoch % 10 == 0:
                self.visualize_reconstruction(train_loader, epoch)
                self.visualize_generation(epoch)

            begin_time = time()

            total_iter += 1
            self.total_iter += 1
