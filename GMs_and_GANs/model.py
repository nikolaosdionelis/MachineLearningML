from __future__ import division
import os
import sys
import time
import math
import tensorflow as tf
import numpy as np
from six.moves import xrange
import dataset_loaders.cifar_loader as cifar_data
import dataset_loaders.mnist_loader as mnist_data

import dataset_loaders.fashionmnist_loader as fashionmnist_data

#from moModel import GANDCGAN2

#from moModel import GANDCGAN2
#from moModel import GANDCGAN2

import scipy
from ops import *
from utils import *

import real_nvp.model as nvp
import real_nvp.nn as nvp_op

#import inception_score

#import inception_score
#import inception_score

import torch
import torch.nn as nn

import torch.nn.init as init
import torch.nn.functional as F


class GeGenerator(nn.Module):
  def __init__(self, gpus, nz, ngf, nc):
    super(GeGenerator, self).__init__()
    self.ngpu = gpus
    self.main = nn.Sequential(
      # inputs is Z, going into a convolution
      nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(True),
      # state size. (ngf*8) x 4 x 4
      nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(True),
      # state size. (ngf*4) x 8 x 8
      nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),
      # state size. (ngf*2) x 14 x 14
      nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
      nn.Tanh(),
      # state size. (ngf) x 28 x 28
    )

  def forward(self, inputs):
    if inputs.is_cuda and self.ngpu > 1:
      outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
    else:
      outputs = self.main(inputs)
    return outputs


class generator(nn.Module):
  # initializers
  def __init__(self, input_size=32, n_class=10):
    super(generator, self).__init__()
    self.fc1 = nn.Linear(input_size, 256)
    self.fc2 = nn.Linear(self.fc1.out_features, 512)
    self.fc3 = nn.Linear(self.fc2.out_features, 1024)
    self.fc4 = nn.Linear(self.fc3.out_features, n_class)

  # forward method
  def forward(self, input):
    x = F.leaky_relu(self.fc1(input), 0.2)
    x = F.leaky_relu(self.fc2(x), 0.2)
    x = F.leaky_relu(self.fc3(x), 0.2)
    x = F.tanh(self.fc4(x))

    return x


class DCDCGANDCGenerator2(nn.Module):
  def __init__(self, imgSize, nz, ngf, nc):
    super(DCDCGANDCGenerator2, self).__init__()

    self.main = nn.Sequential(
      # input is Z, going into a convolution
      nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
      nn.BatchNorm2d(ngf * 8),
      nn.ReLU(True),
      # state size. (ngf*8) x 4 x 4
      nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(True),
      # state size. (ngf*4) x 8 x 8
      nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(True),
      # state size. (ngf*2) x 16 x 16
      nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),
      # state size. (ngf) x 32 x 32
      nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
      nn.Tanh()
      # state size. (nc) x 64 x 64
    )

  def forward(self, input):
    output = self.main(input)

    # print(output.shape)
    # adsfasdf
    
    print(input.shape)
    print(output.shape)
    
    asdfasfda

    return output


class DCDCGANDCGenerator(nn.Module):
  def __init__(self, nrand):
    super(DCDCGANDCGenerator, self).__init__()
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

  def forward(self, z):
    h = F.relu(self.lin1bn(self.lin1(z)))
    h = torch.reshape(h, (-1, 512, 4, 4))

    # deconv stack
    h = F.relu(self.dc1bn(self.dc1(h)))
    h = F.relu(self.dc2bn(self.dc2(h)))
    h = F.relu(self.dc3abn(self.dc3a(h)))
    x = self.dc3b(h)

    return x


class DCGAN(object):
  def __init__(self, sess, input_height=32, input_width=32,
         batch_size=64, sample_num = 64, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default', checkpoint_dir=None,
         f_div='cross-ent', prior="logistic", min_lr=0.0, lr_decay=1.0,
         model_type="nice", alpha=1e-7, loLog_dir=None,
         init_type="uniform",reg=0.5, n_critic=1.0, hidden_layers=1000,
         no_of_layers= 8, like_reg=0.1, just_sample=False, batch_norm_adaptive=1):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.is_grayscale = (c_dim == 1)

    self.batch_size = batch_size
    self.sample_num = batch_size
    
    self.input_height = input_height
    self.input_width = input_width
    self.prior = prior

    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.c_dim = c_dim

    self.lr_decay = lr_decay
    self.min_lr = min_lr
    self.model_type = model_type
    self.loLog_dir = loLog_dir
    self.alpha = alpha
    self.init_type = init_type
    self.reg = reg
    self.n_critic = n_critic
    self.hidden_layers = hidden_layers
    self.no_of_layers = no_of_layers
    
    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.dataset_name = dataset_name
    self.like_reg = like_reg
    if self.dataset_name != 'mnist':
      self.d_bn3 = batch_norm(name='d_bn3')

    self.checkpoint_dir = checkpoint_dir
    self.f_div = f_div
    
    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    self.build_model()

  def build_model(self):
    seed =0
    np.random.seed(seed)
    tf.set_random_seed(seed)

    image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs')
    self.image_size = np.prod(image_dims)
    self.image_dims = image_dims
    if self.dataset_name == "cifar":
      inputs = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), self.inputs)
    else:
      inputs = self.inputs

    sample_inputs = self.sample_inputs

    self.z = tf.placeholder(
      tf.float32, [self.batch_size, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    #### f: Image Space to Latent space #########
    self.flow_model = tf.make_template('model', 
      lambda x: nvp.model_spec(x, reuse=False, model_type=self.model_type, train=False, 
        alpha=self.alpha, init_type=self.init_type, hidden_layers=self.hidden_layers,
        no_of_layers=self.no_of_layers, batch_norm_adaptive=1), unique_name_='model')

    #### f: Image Space to Latent space for training #########
    self.trainable_flow_model = tf.make_template('model', 
      lambda x: nvp.model_spec(x, reuse=True, model_type=self.model_type, train=True, 
        alpha=self.alpha, init_type=self.init_type, hidden_layers=self.hidden_layers,
        no_of_layers=self.no_of_layers, batch_norm_adaptive=1), unique_name_='model')

    # ##### f^-1: Latent to image (trainable)#######
    self.flow_inv_model = tf.make_template('model', 
      lambda x: nvp.inv_model_spec(x, reuse=True, model_type=self.model_type,
       train=True,alpha=self.alpha), unique_name_='model')
    # ##### f^-1: Latent to image (not-trainable just for sampling)#######
    self.sampler_function = tf.make_template('model', 
      lambda x: nvp.inv_model_spec(x, reuse=True, model_type=self.model_type, 
        alpha=self.alpha,train=False), unique_name_='model')

    
    self.generator_train_batch = self.flow_inv_model
    
    ############### SET SIZE FOR TEST BATCH DEPENDING ON WHETHER WE USE Linear or Conv arch##########
    if self.model_type == "nice":
      self.log_like_batch = tf.placeholder(\
        tf.float32, [self.batch_size, self.image_size], name='log_like_batch')
    elif self.model_type == "real_nvp":
      self.log_like_batch = tf.placeholder(\
        tf.float32, [self.batch_size] + self.image_dims, name='log_like_batch')
    ###############################################

    gen_para, jac = self.flow_model(self.log_like_batch)
    if self.dataset_name == "mnist":
      self.log_likelihood = nvp_op.log_likelihood(gen_para, jac, self.prior)/(self.batch_size)
    else:
      # to calculate values in bits per dim we need to
      # multiply the density by the width of the 
      # discrete probability area, which is 1/256.0, per dimension.
      # The calculation is performed in the log space.
      self.log_likelihood = nvp_op.log_likelihood(gen_para, jac, self.prior)/(self.batch_size)
      self.log_likelihood = 8. + self.log_likelihood / (np.log(2)*self.image_size)

    self.G_before_postprocessing = self.generator_train_batch(self.z)
    self.sampler_before_postprocessing = self.sampler_function(self.z)

    if self.model_type == "real_nvp":
      ##For data dependent init (not completely implemented)
      self.x_init = tf.placeholder(tf.float32, shape=[self.batch_size] + image_dims)
      # run once for data dependent initialization of parameters
      self.trainable_flow_model(self.x_init)
    
    inputs_tr_flow = inputs
    if self.model_type == "nice":
      split_val = int(self.image_size /2)
      self.permutation = np.arange(self.image_size)
      tmp = self.permutation.copy()
      self.permutation[:split_val] = tmp[::2]
      self.permutation[split_val:] = tmp[1::2]
      self.for_perm = np.identity(self.image_size)
      self.for_perm = tf.constant(self.for_perm[:,self.permutation], tf.float32)
      self.rev_perm = np.identity(self.image_size)
      self.rev_perm = tf.constant(self.rev_perm[:,np.argsort(self.permutation)], tf.float32)
      self.G_before_postprocessing \
      = tf.matmul(self.G_before_postprocessing,self.rev_perm)
      self.sampler_before_postprocessing \
      = tf.clip_by_value(tf.matmul(self.sampler_before_postprocessing, self.rev_perm) , 0., 1.)
      inputs_tr_flow = tf.matmul(tf.reshape(inputs, [self.batch_size, self.image_size]), self.for_perm)

    train_gen_para, train_jac = self.trainable_flow_model(inputs_tr_flow)
    self.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac, self.prior) / self.batch_size
    
    self.sampler = tf.reshape(self.sampler_before_postprocessing, [self.batch_size] + image_dims)
    self.G = tf.reshape(self.G_before_postprocessing, [self.batch_size] + image_dims)

    inputs = inputs*255.0
    corruption_level = 1.0
    inputs = inputs + corruption_level * tf.random_uniform([self.batch_size] + image_dims)
    inputs = inputs/(255.0 + corruption_level)

    self.D, self.D_logits = self.discriminator(inputs, reuse=False)

    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    ### Vanilla gan loss
    if self.f_div == 'ce':
      self.d_loss_real = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
      self.d_loss_fake = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
      self.g_loss = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
    else:
    ### other gan losses
      if self.f_div == 'hellinger':
        self.d_loss_real = tf.reduce_mean(tf.exp(-self.D_logits))
        self.d_loss_fake = tf.reduce_mean(tf.exp(self.D_logits_) - 2.)
        self.g_loss = tf.reduce_mean(tf.exp(-self.D_logits_))
      elif self.f_div == 'rkl':
        self.d_loss_real = tf.reduce_mean(tf.exp(self.D_logits))
        self.d_loss_fake = tf.reduce_mean(-self.D_logits_ - 1.)
        self.g_loss = -tf.reduce_mean(-self.D_logits_ - 1.)
      elif self.f_div == 'kl':
        self.d_loss_real = tf.reduce_mean(-self.D_logits)
        self.d_loss_fake = tf.reduce_mean(tf.exp(self.D_logits_ - 1.))
        self.g_loss = tf.reduce_mean(-self.D_logits_)
      elif self.f_div == 'tv':
        self.d_loss_real = tf.reduce_mean(-0.5 * tf.tanh(self.D_logits))
        self.d_loss_fake = tf.reduce_mean(0.5 * tf.tanh(self.D_logits_))
        self.g_loss = tf.reduce_mean(-0.5 * tf.tanh(self.D_logits_))
      elif self.f_div == 'lsgan':
        self.d_loss_real = 0.5 * tf.reduce_mean((self.D_logits-1)**2)
        self.d_loss_fake = 0.5 * tf.reduce_mean(self.D_logits_**2)
        self.g_loss = 0.5 * tf.reduce_mean((self.D_logits_-1)**2)
      elif self.f_div == "wgan":
        self.g_loss = -tf.reduce_mean(self.D_logits_)
        self.d_loss_real = -tf.reduce_mean(self.D_logits)
        self.d_loss_fake = tf.reduce_mean(self.D_logits_)
        alpha = tf.random_uniform(
            shape=[1, self.batch_size], 
            minval=0.,
            maxval=1.
        )
        fake_data = self.G
        real_data = inputs
        differences = fake_data - real_data
        interpolates = real_data + \
        tf.transpose(alpha*tf.transpose(differences, perm=[1,2,3,0]), [3,0,1,2])
        _, d_inter = self.discriminator(interpolates, reuse=True) 
        gradients = tf.gradients(d_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        self.gradient_penalty = tf.reduce_mean((slopes-1.)**2)
      else:
        print("ERROR: Unrecognized f-divergence...exiting")
        exit(-1)

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    if self.f_div == "wgan":
      self.d_loss = self.d_loss_real + self.d_loss_fake + self.reg * self.gradient_penalty
    else:
      self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if '/d_' in var.name]
    self.g_vars = [var for var in t_vars if '/g_' in var.name]
    print("gen_vars:")
    for var in self.g_vars:
      print(var.name)

    print("disc_vars:")
    for var in self.d_vars:
      print(var.name)
    
    self.saver = tf.train.Saver(max_to_keep=0)

  def evaluate_neg_loglikelihood(self, data, config):
    log_like_batch_idxs = len(data) // config.batch_size
    lli_list = []
    inter_list = []
    for idx in xrange(0, log_like_batch_idxs):
      batch_images = data[idx*config.batch_size:(idx+1)*config.batch_size]
      batch_images = np.cast[np.float32](batch_images)
      
      if self.model_type == "nice":
        batch_images = batch_images[:,self.permutation]

      lli = self.sess.run([self.log_likelihood],
        feed_dict={self.log_like_batch: batch_images})
      
      lli_list.append(lli)

    return np.mean(lli_list)

  def evaluate_neg_loglikelihood2(self, data, config):
      log_like_batch_idxs = len(data) // config.batch_size
      lli_list = []
      inter_list = []
      for idx in xrange(0, log_like_batch_idxs):
          batch_images = data[idx * config.batch_size:(idx + 1) * config.batch_size]
          batch_images = np.cast[np.float32](batch_images)

          if self.model_type == "nice":
              batch_images = batch_images[:, self.permutation]

          lli = self.sess.run([self.log_likelihood],
                              feed_dict={self.log_like_batch: batch_images})

          lli_list.append(lli)

      #return np.mean(lli_list) / (28 * 28 * 1) / np.log(2)
      #return np.mean(lli_list) / (28 * 28 * 1) / np.log(2)
      #return np.mean(lli_list) / (28 * 28 * 1) / np.log(2)
      #return np.mean(lli_list) / (28 * 28 * 1) / np.log(2)
      #return np.exp(-np.mean(lli_list) / 100000)
      #return np.exp(-np.mean(lli_list) / 100000)
      return np.exp(-np.mean(lli_list) / 1000000)
      #dfsgdsg43 = np.mean(lli_list) / (28 * 28 * 1) / np.log(2)
      #return np.exp(-dfsgdsg43)

  def evaluate_neg_loglikelihood22(self, data, config):
      log_like_batch_idxs = data.shape[0] // config.batch_size
      #print(log_like_batch_idxs)
      #print(log_like_batch_idxs)
      #print(log_like_batch_idxs)
      lli_list = []
      inter_list = []
      for idx in xrange(0, log_like_batch_idxs):
          batch_images = data[idx * config.batch_size:(idx + 1) * config.batch_size]
          #print(batch_images)
          #print(batch_images.shape)
          #fdsafsfa
          #batch_images = batch_images.eval()
          batch_images = batch_images.cpu().detach().numpy()
          batch_images = np.cast[np.float32](batch_images)

          if self.model_type == "nice":
              batch_images = batch_images[:, self.permutation]

          lli = self.sess.run([self.log_likelihood],
                              feed_dict={self.log_like_batch: batch_images})

          lli_list.append(lli)

      #return np.mean(lli_list) / (28 * 28 * 1) / np.log(2)
      #return np.mean(lli_list) / (28 * 28 * 1) / np.log(2)
      #return np.mean(lli_list) / (28 * 28 * 1) / np.log(2)
      #return np.mean(lli_list) / (28 * 28 * 1) / np.log(2)
      #return np.exp(-np.mean(lli_list) / 100000)
      #return np.exp(-np.mean(lli_list) / 100000)
      #return np.exp(-np.mean(lli_list) / 1000000)
      #return np.exp(-np.mean(lli_list) / 1000000)
      return np.mean(lli_list)
      #dfsgdsg43 = np.mean(lli_list) / (28 * 28 * 1) / np.log(2)
      #return np.exp(-dfsgdsg43)

  def evaluate_neg_loglikelihood3(self, data, config):
      log_like_batch_idxs = len(data) // config.batch_size
      lli_list = []
      inter_list = []
      for idx in xrange(0, log_like_batch_idxs):
          batch_images = data[idx * config.batch_size:(idx + 1) * config.batch_size]
          batch_images = np.cast[np.float32](batch_images)

          if self.model_type == "nice":
              batch_images = batch_images[:, self.permutation]

          lli = self.sess.run([self.log_likelihood],
                              feed_dict={self.log_like_batch: batch_images})

          #print(lli)
          #asfasfas

          #print(lli)

          #lli_list.append(lli[0])
          lli_list.append(-lli[0])

          #lli_list.append(-lli[0])
          #lli_list.append(lli[0])

      #print(lli_list)
      #asdfasdf

      #print(lli_list)
      #asdfasdf

      #import bigfloat
      #bigfloat.exp(5000, bigfloat.precision(100))

      #return np.mean(np.exp(lli_list))
      #return np.mean(bigfloat.exp(lli_list, bigfloat.precision(100)))

      #bits_per_dim = np.mean(lli_list) / (config.image_size * config.image_size * 1) / np.log(2)
      bits_per_dim = np.mean(lli_list) / (28 * 28 * 1) / np.log(2)
      return np.exp(bits_per_dim)

      '''
      x = lli_list
      b = np.max(x)
      #print(b)
      #asdfas
      #y = np.exp(x - b)
      #return y
      #return np.mean(y)
      return np.exp(np.mean(x-b))
      '''

      #return np.mean(np.exp(lli_list))
      #return np.exp(np.mean(lli_list))

      '''
      #adfas
      x = lli_list
      #b = x.max()
      b = np.max(x)
      y = np.exp(x - b)
      #return y / y.sum()
      #return (y / y.sum())
      return np.mean((y / y.sum()))
      #asdfasd
      '''

      #return np.mean(np.exp(lli_list))

      #return np.mean(np.exp(lli_list))
      #return np.mean(np.exp(lli_list))

  def train(self, config):
    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)
    """Train DCGAN"""
    if config.dataset == "mnist":
      data_X, val_data, test_data, train_dist = mnist_data.load_mnist()
    elif config.dataset == "cifar":
      data_X, val_data, test_data = cifar_data.load_cifar()

    if self.model_type == "nice":
      val_data = np.reshape(val_data, (-1,self.image_size))
      test_data = np.reshape(test_data, (-1, self.image_size))

    lr = config.learning_rate
    self.learning_rate = tf.placeholder(tf.float32, [], name='lr')

    d_optim_ = tf.train.AdamOptimizer(self.learning_rate, beta1=config.beta1, beta2=0.9)
    d_grad = d_optim_.compute_gradients(self.d_loss, var_list=self.d_vars)
    d_grad_mag = tf.global_norm(d_grad)
    d_optim = d_optim_.apply_gradients(d_grad)          

    g_optim_ = tf.train.AdamOptimizer(self.learning_rate, beta1=config.beta1, beta2=0.9)
    if self.n_critic <= 0:
      g_grad = g_optim_.compute_gradients(self.train_log_likelihood\
          , var_list=self.g_vars)
    else:
      if self.like_reg > 0:
        if self.model_type == "real_nvp":
          g_grad_1 = g_optim_.compute_gradients(self.g_loss / self.like_reg, var_list=self.g_vars)
          g_grad_2 = g_optim_.compute_gradients(self.train_log_likelihood, var_list=self.g_vars)
          grads_1, _ = zip(*g_grad_1)
          grads_2, _ = zip(*g_grad_2)
          sum_grad = [g1+g2 for g1, g2 in zip(grads_1, grads_2)]
          g_grad = [pair for pair in zip(sum_grad, [var for grad, var in g_grad_1])]
        else:
          g_grad = g_optim_.compute_gradients(self.g_loss/self.like_reg + self.train_log_likelihood ,var_list=self.g_vars)  
      else:
        g_grad = g_optim_.compute_gradients(self.g_loss, var_list=self.g_vars)

    
    g_grad_mag = tf.global_norm(g_grad)
    g_optim = g_optim_.apply_gradients(g_grad)         

    try: ##for data-dependent init (not implemented)
      if self.model_type == "real_nvp":
        self.sess.run(tf.global_variables_initializer(),
          {self.x_init: data_X[0:config.batch_size]})
      else:
        self.sess.run(tf.global_variables_initializer())
    except:
      if self.model_type == "real_nvp":
        self.sess.run(tf.global_variables_initializer(),
          {self.x_init: data_X[0:config.batch_size]})
      else:
        self.sess.run(tf.global_variables_initializer())

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./"+self.loLog_dir, self.sess.graph)

    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    #if could_load:
    #  counter = checkpoint_counter
    #  print(" [*] Load SUCCESS")
    #else:
    #  print(" [!] Load failed...")

    #if could_load:
    #  counter = checkpoint_counter
    #  print(" [*] Load SUCCESS")
    #else:
    #  print(" [!] Load failed...")

    print(could_load)
    print(checkpoint_counter)

    if could_load:
     counter = checkpoint_counter
     print(" [*] Load SUCCESS")
    else:
     print(" [!] Load failed...")

    #dsafa
    #dfdasdf

    print(checkpoint_counter)
    print(checkpoint_counter)

    #counter = 1

    print(counter)
    print(counter)

    ############## A FIXED BATCH OF Zs FOR GENERATING SAMPLES ######################
    if self.prior == "uniform":
      sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    elif self.prior == "logistic":
      sample_z = np.random.logistic(loc=0., scale=1., size=(self.sample_num , self.z_dim))
    elif self.prior == "gaussian":
      sample_z = np.random.normal(0.0, 1.0, size=(self.sample_num , self.z_dim))
    else:
        print("ERROR: Unrecognized prior...exiting")
        exit(-1)

    ################################ Evaluate initial model lli ########################

    val_nlli = self.evaluate_neg_loglikelihood(val_data, config)
    # train_nlli = self.evaluate_neg_loglikelihood(train_data, config)

    curr_inception_score = self.calculate_inception_and_mode_score()
    print("INITIAL TEST: val neg logli: %.8f,incep score: %.8f" % (val_nlli,\
     curr_inception_score[0]))
    if counter > 1:
      #old_data = np.load("./"+config.sample_dir+'/graph_data.npy')

      #old_data = np.load("./" + config.sample_dir + '/graph_data.npy')

      #old_data = np.load("./" + config.sample_dir + '/graph_data.npy')
      old_data = np.load("./" + config.sample_dir + '/graph_data.npy', allow_pickle=True)

      self.best_val_nlli = old_data[2]
      self.best_model_counter = old_data[3]
      self.best_model_path = old_data[4]
      self.val_nlli_list = old_data[1]
      self.counter_list = old_data[5]
      self.batch_train_nlli_list = old_data[-4]
      self.inception_list = old_data[-2]
      self.samples_list = old_data[0]
      self.loss_list = old_data[-1]
      manifold_h, manifold_w = old_data[6]
    else:
      self.writer.add_summary(tf.Summary(\
              value=[tf.Summary.Value(tag="Val Neg Log-likelihood", simple_value=val_nlli)]), counter)
      # self.writer.add_summary(tf.Summary(\
      #         value=[tf.Summary.Value(tag="Train Neg Log-likelihood", simple_value=train_nlli)]), counter)

      self.best_val_nlli = val_nlli
      # self.best_model_train_nlli = train_nlli
      self.best_model_counter = counter
      self.best_model_path = self.save(config.checkpoint_dir, counter)
      # self.train_nlli_list = [train_nlli]
      self.val_nlli_list = [val_nlli]
      self.counter_list = [1]
      self.batch_train_nlli_list = []
      self.inception_list = [curr_inception_score]
      self.samples_list = self.sess.run([self.sampler],
              feed_dict={
                  self.z: sample_z,
              }
            )
      sample_inputs = data_X[0:config.batch_size]
      samples = self.samples_list[0]
      manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
      manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
      self.loss_list = self.sess.run(
              [self.d_loss_real, self.d_loss_fake],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
              })

      #print(sample_inputs)
      #adsfasdf
    ##################################################################################

    for epoch in xrange(config.epoch):
      np.random.shuffle(data_X)
      batch_idxs = len(data_X) // config.batch_size
      
      for idx in xrange(0, batch_idxs):
        sys.stdout.flush()
        batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]

        print(len(batch_images))

        print(len(batch_images))
        print(len(batch_images))
        
        if self.prior == "uniform":
          batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)
        elif self.prior == "logistic":
          batch_z = np.random.logistic(loc=0.,scale=1.0,size=[config.batch_size, self.z_dim]) \
              .astype(np.float32)
        elif self.prior == "gaussian":
          batch_z = np.random.normal(0.0, 1.0, size=(config.batch_size , self.z_dim))
        else:
          print("ERROR: Unrecognized prior...exiting")
          exit(-1)

        for r in range(self.n_critic):
          _, d_g_mag, errD_fake, errD_real ,summary_str = self.sess.run([d_optim, d_grad_mag, 
            self.d_loss_fake, self.d_loss_real, self.d_sum],
            feed_dict={ 
              self.inputs: batch_images,
              self.z: batch_z,
              self.learning_rate:lr,
            })
        if self.n_critic > 0:
          self.writer.add_summary(summary_str, counter)

        # Update G network
        if self.like_reg > 0 or self.n_critic <= 0:
          _, g_g_mag, errG, summary_str = self.sess.run([g_optim, g_grad_mag, self.g_loss, self.g_sum],
            feed_dict={
              self.z: batch_z, 
              self.learning_rate:lr,
              self.inputs: batch_images,
            })
        else:
          _, g_g_mag ,errG, summary_str = self.sess.run([g_optim, g_grad_mag, self.g_loss, self.g_sum],
            feed_dict={
              self.z: batch_z, 
              self.learning_rate:lr,
            })
        self.writer.add_summary(summary_str, counter)

        batch_images_nl = batch_images
        if self.model_type == "nice":
          batch_images_nl = np.reshape(batch_images_nl,(self.batch_size, -1))[:,self.permutation]
        b_train_nlli = self.sess.run([self.log_likelihood], feed_dict={
          self.log_like_batch: batch_images_nl,
          })
        b_train_nlli = b_train_nlli[0]

        self.batch_train_nlli_list.append(b_train_nlli)
        if self.n_critic > 0:
          self.loss_list.append([errD_real, errD_fake])
          self.writer.add_summary(tf.Summary(\
          value=[tf.Summary.Value(tag="training loss", simple_value=-(errD_fake+errD_real))]) ,counter)
        self.writer.add_summary(tf.Summary(\
          value=[tf.Summary.Value(tag="Batch train Neg Log-likelihood", simple_value=b_train_nlli)]) ,counter)
        counter += 1


        lr = max(lr * self.lr_decay, self.min_lr)

        if np.mod(counter, 703) == 1: #340
          if self.n_critic > 0:
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, d_grad_mag: %.8f, g_grad_mag: %.8f, lr: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG, d_g_mag, g_g_mag, lr))
          else:
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f, g_grad_mag: %.8f, lr: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errG, g_g_mag, lr))

          # print(test_data)
          # print(test_data.shape)

          #test_nlli = self.evaluate_neg_loglikelihood(test_data, config)

          #test_nlli = self.evaluate_neg_loglikelihood(test_data, config)
          test_nlli = self.evaluate_neg_loglikelihood(test_data, config)

          val_nlli = self.evaluate_neg_loglikelihood(val_data, config)
          #train_nlli = self.evaluate_neg_loglikelihood(data_X, config)
          
          #print(train_nlli)
          #print(val_nlli)

          #print(test_nlli)

          #print(test_nlli)
          #print(test_nlli)

          #print(self.model_type)

          #print(self.model_type)
          #print(self.model_type)

          data_X, _, _, _ = mnist_data.load_mnist()
          data_X = np.reshape(data_X, (-1, self.image_size))

          train_nlli = self.evaluate_neg_loglikelihood(data_X, config)
          #print(train_nlli)

          print(train_nlli)
          print(val_nlli)

          print(test_nlli)

          test_nlli = self.evaluate_neg_loglikelihood2(test_data, config)
          val_nlli = self.evaluate_neg_loglikelihood2(val_data, config)

          train_nlli = self.evaluate_neg_loglikelihood2(data_X, config)

          print(train_nlli)
          print(val_nlli)

          print(test_nlli)



          # use: fashionmnist_data
          #data_X, val_data, test_data, train_dist = mnist_data.load_mnist()

          #print(data_X.shape)
          #print(val_data.shape)

          #print(val_data.shape)
          #print(test_data.shape)

          #data_X, val_data, test_data, train_dist = mnist_data.load_mnist()

          #data_X, val_data, test_data, train_dist = mnist_data.load_mnist()
          data_X, val_data, test_data, _ = fashionmnist_data.load_mnist()

          data_X = np.reshape(data_X, (-1, self.image_size))
          val_data = np.reshape(val_data, (-1, self.image_size))

          #data_X = np.reshape(data_X, (-1, self.image_size))
          test_data = np.reshape(test_data, (-1, self.image_size))

          #print(data_X.shape)
          #print(val_data.shape)

          #print(val_data.shape)
          #print(test_data.shape)

          test_nlli = self.evaluate_neg_loglikelihood(test_data, config)
          val_nlli = self.evaluate_neg_loglikelihood(val_data, config)

          train_nlli = self.evaluate_neg_loglikelihood(data_X, config)

          print(train_nlli)
          print(val_nlli)

          print(test_nlli)

          test_nlli = self.evaluate_neg_loglikelihood2(test_data, config)
          val_nlli = self.evaluate_neg_loglikelihood2(val_data, config)

          train_nlli = self.evaluate_neg_loglikelihood2(data_X, config)

          print(train_nlli)
          print(val_nlli)

          print(test_nlli)



          #train_gen_para, train_jac = self.trainable_flow_model(inputs_tr_flow)
          #self.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac, self.prior) / self.batch_size

          #train_gen_para, train_jac = self.trainable_flow_model(inputs_tr_flow)
          #self.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac, self.prior) / self.batch_size

          #train_gen_para, train_jac = self.trainable_flow_model(inputs_tr_flow)
          #self.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac, self.prior) / self.batch_size

          #self.sampler_function(self.z)
          #self.flow_inv_model

          #train_gen_para, train_jac = self.trainable_flow_model(inputs_tr_flow)
          #self.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac, self.prior) / self.batch_size

          #self.flow_inv_model(self.z)
          # use: self.flow_inv_model(self.z)

          #train_nlli = self.evaluate_neg_loglikelihood2(data_X, config)
          # use: train_nlli = self.evaluate_neg_loglikelihood2(data_X, config)

          # we use: self.flow_inv_model(self.z)
          # now use: self.flow_inv_model(self.z)
          
          # first term: self.flow_inv_model(self.z)
          #train_nlli = self.evaluate_neg_loglikelihood2(self.flow_inv_model(self.z), config)

          #train_nlli = self.evaluate_neg_loglikelihood2(self.flow_inv_model(self.z), config)

          #train_nlli = self.evaluate_neg_loglikelihood2(self.flow_inv_model(self.z), config)
          train_nlli = self.evaluate_neg_loglikelihood22(self.flow_inv_model(self.z), config)

          print(train_nlli)



          asdfszd



          #_, _, _, _, _, test_labels = mnist_data.load_mnist(send_labels=True)

          #_, _, _, _, _, test_labels = mnist_data.load_mnist(send_labels=True)
          #_, _, _, trTrTrTrainLabels, _, test_labels = mnist_data.load_mnist(send_labels=True)

          #data_X, val_data, test_data, train_dist = mnist_data.load_mnist()

          #data_X, val_data, test_data, train_dist = mnist_data.load_mnist()
          #data_X, val_data, test_data, train_dist = mnist_data.load_mnist()

          #data_X, val_data, test_data, train_dist = mnist_data.load_mnist()
          #_, _, _, trTrTrTrainLabels, _, test_labels = mnist_data.load_mnist(send_labels=True)

          data_X, _, _, trTrTrTrainLabels, _, test_labels = mnist_data.load_mniMnist(send_labels=True)

          data = test_data
          config.batch_size = 1

          losses_NIKlosses = []

          loLosses_NIKlosses = []
          loLosses_NIKlosses2 = []

          # loLosses_NIKlosses2 = []
          loLosses_NIKlosses3 = []

          #print(data_X)
          #print(data_X.shape)

          #print(data)
          #print(data.shape)

          #dsfsazdf

          #print(data_X.shape)
          #print(data.shape)

          #data = tf.reshape(data_X, [-1, 784])

          #data = tf.reshape(data_X, [-1, 784])
          #data = tf.reshape(data_X, [-1, 784])

          # (?)
          #data = tf.reshape(data_X, [-1, 784])
          # (?)

          #print(data_X.shape)
          #print(data.shape)



          x = tf.reshape(data_X, [-1, 784])
          #x = x.to(device)

          y = trTrTrTrainLabels
          #y = y.to(device)

          #print(x.shape)
          #print(y.shape)

          #dasfsadfazs

          # config.batch_size = 1024
          # config.batch_size = 16384

          # config.batch_size = 1024
          # config.batch_size = 2048

          # config.batch_size = 2048
          # config.batch_size = 150

          # config.batch_size = 2048
          #config.batch_size = 2 * 2048

          #config.batch_size = 2 * 2048
          #config.batch_size = 2048

          config.batch_size = 100

          # config.batch_size = 1024
          # ggenFGen2 = torch.randn([config.batch_size, nrand], device=device, requires_grad=True)

          # genFGen2 = genGen.forward(ggenFGen2)
          # genFGen2 = genGen.forward(ggenFGen2)

          # ggenFGen2 = torch.randn([config.batch_size, nrand], device=device, requires_grad=True)
          # genFGen2 = genGen.forward(ggenFGen2)

          # ggenFGen2 = torch.randn([config.batch_size, nrand], device=device)
          # genFGen2 = genGen.forward(ggenFGen2)

          #lli = self.sess.run([self.log_likelihood],
          #                    feed_dict={self.log_like_batch: batch_images})

          #sample_z = np.random.logistic(loc=0., scale=1., size=(self.sample_num, self.z_dim))
          #sample_z = np.random.logistic(loc=0., scale=1., size=(config.batch_size, self.z_dim))

          batch_z = np.random.logistic(loc=0., scale=1.0, size=[config.batch_size, self.z_dim]) \
            .astype(np.float32)

          ggenFGen2 = batch_z
          #ggenFGen2 = torch.randn([config.batch_size, 100, 1, 1], device=device)

          #genFGen2 = self.sess.run([self.G],
          #                         feed_dict={self.z: sample_z, self.batch_size: config.batch_size})

          genFGen2 = self.sess.run(
            [self.sampler],
            feed_dict={
              self.z: batch_z,
              }
          )

          #np.concatenate((a, b), axis=0)

          #np.concatenate((a, b), axis=0)
          #np.concatenate((a, b), axis=0)

          #print(np.shape(ggenFGen2))
          #print(np.shape(genFGen2))

          ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
            .astype(np.float32)

          ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
          genFGen2 = np.concatenate((genFGen2, self.sess.run(
            [self.sampler],
            feed_dict={
              self.z: ggenGen2FGen2,
              }
          )), axis=1)

          config.batch_size += 100

          #print(np.shape(ggenFGen2))
          #print(np.shape(genFGen2))

          #dasfasdfsdf

          ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
            .astype(np.float32)

          ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
          genFGen2 = np.concatenate((genFGen2, self.sess.run(
            [self.sampler],
            feed_dict={
              self.z: ggenGen2FGen2,
            }
          )), axis=1)

          config.batch_size += 100

          ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
            .astype(np.float32)

          ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
          genFGen2 = np.concatenate((genFGen2, self.sess.run(
            [self.sampler],
            feed_dict={
              self.z: ggenGen2FGen2,
            }
          )), axis=1)

          config.batch_size += 100

          ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
            .astype(np.float32)

          ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
          genFGen2 = np.concatenate((genFGen2, self.sess.run(
            [self.sampler],
            feed_dict={
              self.z: ggenGen2FGen2,
            }
          )), axis=1)

          config.batch_size += 100

          ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
            .astype(np.float32)

          ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
          genFGen2 = np.concatenate((genFGen2, self.sess.run(
            [self.sampler],
            feed_dict={
              self.z: ggenGen2FGen2,
            }
          )), axis=1)

          config.batch_size += 100

          ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
            .astype(np.float32)

          ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
          genFGen2 = np.concatenate((genFGen2, self.sess.run(
            [self.sampler],
            feed_dict={
              self.z: ggenGen2FGen2,
            }
          )), axis=1)

          config.batch_size += 100

          ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
            .astype(np.float32)

          ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
          genFGen2 = np.concatenate((genFGen2, self.sess.run(
            [self.sampler],
            feed_dict={
              self.z: ggenGen2FGen2,
            }
          )), axis=1)

          config.batch_size += 100

          ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
            .astype(np.float32)

          ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
          genFGen2 = np.concatenate((genFGen2, self.sess.run(
            [self.sampler],
            feed_dict={
              self.z: ggenGen2FGen2,
            }
          )), axis=1)

          config.batch_size += 100

          ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
            .astype(np.float32)

          ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
          genFGen2 = np.concatenate((genFGen2, self.sess.run(
            [self.sampler],
            feed_dict={
              self.z: ggenGen2FGen2,
            }
          )), axis=1)

          config.batch_size += 100

          #print(np.shape(ggenFGen2))
          #print(np.shape(genFGen2))

          #dasfasdfsdf

          print(np.shape(ggenFGen2))

          print(np.shape(ggenFGen2))
          print(np.shape(ggenFGen2))

          #samples_curr = self.sess.run(
          #  [self.sampler],
          #  feed_dict={
          #    self.z: batch_z, }
          #)

          #genFGen2 = self.sess.run([self.G],
          #                         feed_dict={self.z: sample_z})

          #genFGen2 = tf.reshape(genFGen2, [config.batch_size, 784])
          genFGen2 = np.reshape(genFGen2, (config.batch_size, 784))

          #print(genFGen2)
          print(np.shape(genFGen2))

          print(np.shape(genFGen2))
          print(np.shape(genFGen2))

          #print(genFGen2.shape)
          #asdfasdf

          print(x.shape)
          print(y.shape)

          #asdfasdfas

          #genFGen2 = self.sess.run([self.G],
          #                    feed_dict={self.log_like_batch: batch_images})

          #with torch.no_grad():
          #    ggenFGen2 = torch.randn([config.batch_size, 100, 1, 1], device=device)
          #    # genFGen2 = netG.forward(ggenFGen2)

          #    # genFGen2 = netG.forward(ggenFGen2)

          #    # genFGen2 = netG.forward(ggenFGen2)
          #    # genFGen2 = netG.forward(ggenFGen2)

          #    genFGen2 = netG.forward(ggenFGen2)
          #    # genFGen2 = netG2.forward(ggenFGen2)

          # print(x.shape)
          # print(y.shape)

          # print(genFGen2.shape)
          # print(config.batch_size)

          # for i21 in range(len(y)):
          #    if y[i21] == 0 and i21 == 0:
          #        y[i21] = y[i21+1]
          #        x[i21, :, :, :] = x[i21+1, :, :, :]
          #    elif y[i21] == 0:
          #        y[i21] = y[i21 - 1]
          #        x[i21, :, :, :] = x[i21 - 1, :, :, :]

          #import torch

          # y2 = []
          x2 = []
          for i21 in range(len(y)):
              if y[i21] == 1:
                  # y2.append(y[i21])
                  #x2.append(x[i21, :, :, :])

                  #x2.append(x[i21, :, :, :])
                  x2.append(x[i21, :])

          #x2 = tf.stack(x2)
          x2 = tf.stack(x2)
          # y2 = torch.stack(y2)

          # y3 = []
          x3 = []
          for i21 in range(len(y)):
              if y[i21] == 2:
                  # y3.append(y[i21])
                  #x3.append(x[i21, :, :, :])

                  #x3.append(x[i21, :, :, :])
                  x3.append(x[i21, :])

          #x3 = tf.stack(x3)
          x3 = tf.stack(x3)
          # y3 = torch.stack(y3)

          # y4 = []
          x4 = []
          for i21 in range(len(y)):
              if y[i21] == 3:
                  # y4.append(y[i21])
                  #x4.append(x[i21, :, :, :])

                  #x4.append(x[i21, :, :, :])
                  x4.append(x[i21, :])

          x4 = tf.stack(x4)
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
                  #x5.append(x[i21, :, :, :])

                  #x5.append(x[i21, :, :, :])
                  x5.append(x[i21, :])

          x5 = tf.stack(x5)
          # y5 = torch.stack(y5)

          # y6 = []
          x6 = []
          for i21 in range(len(y)):
              if y[i21] == 5:
                  # y6.append(y[i21])
                  #x6.append(x[i21, :, :, :])

                  #x6.append(x[i21, :, :, :])
                  x6.append(x[i21, :])

          x6 = tf.stack(x6)
          # y6 = torch.stack(y6)

          # y7 = []
          x7 = []
          for i21 in range(len(y)):
              if y[i21] == 6:
                  # y7.append(y[i21])
                  #x7.append(x[i21, :, :, :])

                  #x7.append(x[i21, :, :, :])
                  x7.append(x[i21, :])

          x7 = tf.stack(x7)
          # y7 = torch.stack(y7)

          # y8 = []
          x8 = []
          for i21 in range(len(y)):
              if y[i21] == 7:
                  # y8.append(y[i21])
                  #x8.append(x[i21, :, :, :])

                  #x8.append(x[i21, :, :, :])
                  x8.append(x[i21, :])

          x8 = tf.stack(x8)
          # y8 = torch.stack(y8)

          # y9 = []
          x9 = []
          for i21 in range(len(y)):
              if y[i21] == 8:
                  # y9.append(y[i21])
                  #x9.append(x[i21, :, :, :])

                  #x9.append(x[i21, :, :, :])
                  x9.append(x[i21, :])

          x9 = tf.stack(x9)
          # y9 = torch.stack(y9)

          # y99 = []
          x99 = []
          for i21 in range(len(y)):
              if y[i21] == 9:
                  # y99.append(y[i21])
                  #x99.append(x[i21, :, :, :])

                  #x99.append(x[i21, :, :, :])
                  x99.append(x[i21, :])

          x99 = tf.stack(x99)
          # y99 = torch.stack(y99)

          x999 = []
          for i21 in range(len(y)):
              if y[i21] == 0:
                  #x999.append(x[i21, :, :, :])

                  #x999.append(x[i21, :, :, :])
                  x999.append(x[i21, :])
          x999 = tf.stack(x999)

          # print(x9.shape)
          # print(x99.shape)
          # print(genFGen2.shape)

          # print(x999.shape)
          # asdfasdfs

          '''
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
          # x3 = x3.view(-1, 64 * 64)

          # x3 = x3.view(-1, 64 * 64)

          # x3 = x3.view(-1, 64 * 64)
          # x3 = x3.view(-1, 64 * 64)

          x3 = x3.view(-1, 64 * 64)

          x2 = x2.view(-1, 64 * 64)
          # x8 = x8.view(-1, 64 * 64)
          '''

          # print(config.batch_size)
          # print(genFGen2.shape)

          x999 = x999.eval()

          x99 = x99.eval()
          x9 = x9.eval()

          x8 = x8.eval()
          x7 = x7.eval()

          x6 = x6.eval()
          x5 = x5.eval()

          x4 = x4.eval()
          x3 = x3.eval()

          x2 = x2.eval()

          #with torch.no_grad():

          #print(config.batch_size)
          #asdfasdfs

          with tf.device('/gpu:0'):
            # second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
            # second_term_loss32 = torch.empty(config.batch_size, device=device, requires_grad=False)
            # second_term_loss32 = torch.empty(config.batch_size, device=device)
            # second_term_loss32 = tf.zeros(config.batch_size)
            second_term_loss32 = []
            # for i in range(args.batch_size):
            for i in range(config.batch_size):
              print(i)

              """
              print(torch.mean(torch.sqrt((genFGen2[i, :] - xData).view(config.batch_size, -1).pow(2).sum(1))))
              print(torch.mean(torch.sqrt((genFGen2[i, :] - genFGen2).view(config.batch_size, -1).pow(2).sum(1))))
              print(torch.mean(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1))))
              print('')

              print(torch.mean(torch.norm((genFGen2[i, :] - xData).view(config.batch_size, -1), p=None, dim=1)))
              print(torch.mean(torch.norm((genFGen2[i, :] - genFGen2).view(config.batch_size, -1), p=None, dim=1)))
              print(torch.mean(torch.norm((genFGen3[i, :] - genFGen3), p=None, dim=1)))
              print('')
              """

              # print(torch.mean(torch.sqrt((genFGen2[i, :] - xData).view(config.batch_size, -1).pow(2).sum(1))))
              # print(torch.mean(torch.sqrt((genFGen2[i, :] - genFGen2).view(config.batch_size, -1).pow(2).sum(1))))
              # print(torch.mean(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1))))
              # print('')

              # print(torch.sqrt((genFGen2[i, :] - xData).view(config.batch_size, -1).pow(2).sum(1)))
              # print(torch.sqrt((genFGen2[i, :] - genFGen2).view(config.batch_size, -1).pow(2).sum(1)))
              # print(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1)))
              # print('')

              # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p='fro', dim=1).requires_grad_()
              # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()
              # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()**2
              # second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1))**2
              # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_()**2

              # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2

              # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2
              # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).view(config.batch_size, -1).pow(2).sum(1)).requires_grad_() ** 2

              # second_term_loss22 = torch.sqrt(
              #    1e-17 + (genFGen2[i, :] - xData).view(config.batch_size, -1).pow(2).sum(1)).requires_grad_() ** 2

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

              # secondSecSec_term_loss32 = tf.zeros(10)
              secondSecSec_term_loss32 = []

              # secondSecSec_term_loss32 = tf.zeros(10)
              # secondSecSec_term_loss32[8] = torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2

              # secondSecSec_term_loss32[8] = torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2
              # secondSecSecSec_term_loss32 = torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2

              # secondSecSec_term_loss32[8] = torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2

              # secondSecSec_term_loss32[8] = torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2
              # secondSecSec_term_loss32[8] = torch.min(torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2)

              # secondSecSec_term_loss32[7] = torch.min(torch.sqrt((genFGen2[i, :] - x9).pow(2).sum(1)) ** 2)
              # secondSecSec_term_loss32[6] = torch.min(torch.sqrt((genFGen2[i, :] - x8).pow(2).sum(1)) ** 2)

              # secondSecSec_term_loss32[5] = torch.min(torch.sqrt((genFGen2[i, :] - x7).pow(2).sum(1)) ** 2)
              # secondSecSec_term_loss32[4] = torch.min(torch.sqrt((genFGen2[i, :] - x6).pow(2).sum(1)) ** 2)

              # secondSecSec_term_loss32[3] = torch.min(torch.sqrt((genFGen2[i, :] - x5).pow(2).sum(1)) ** 2)
              # secondSecSec_term_loss32[2] = torch.min(torch.sqrt((genFGen2[i, :] - x4).pow(2).sum(1)) ** 2)

              # secondSecSec_term_loss32[1] = torch.min(torch.sqrt((genFGen2[i, :] - x3).pow(2).sum(1)) ** 2)
              # secondSecSec_term_loss32[0] = torch.min(torch.sqrt((genFGen2[i, :] - x2).pow(2).sum(1)) ** 2)

              # print(secondSecSec_term_loss32)
              # print(torch.min(torch.sqrt((genFGen2[i, :] - x999).pow(2).sum(1)) ** 2))

              # use: x999
              # secondSecSec_term_loss32.append((tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2)).eval())

              # print((
              #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2)).eval()))

              #x999 = x999.eval()

              #print((
              #   (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x999), 2), 1)) ** 2))))

              # print((
              #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2))))

              # print(genFGen2[i, :])
              # print(x999)

              # print(genFGen2[i, :])

              # with tf.device('/CPU:0'):
              #    print((
              #        (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2)).eval()))

              #asdfasdfs

              #secondSecSec_term_loss32.append(
              #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2)).eval())

              secondSecSec_term_loss32.append((
                (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x999), 2), 1)) ** 2))))

              # print(((tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2)).eval()))

              # print(((tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2)).eval()))
              # print(((tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2)).eval()))

              secondSecSec_term_loss32.append((
                (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x2), 2), 1)) ** 2))))

              #secondSecSec_term_loss32.append(
              #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x2), 2), 1)) ** 2)).eval())

              secondSecSec_term_loss32.append((
                (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x3), 2), 1)) ** 2))))

              #secondSecSec_term_loss32.append(
              #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x3), 2), 1)) ** 2)).eval())

              # print(genFGen2[i, :].shape)
              # print(x4.shape)

              # print(x3.shape)
              # print(x2.shape)

              # asadfasfasf

              secondSecSec_term_loss32.append((
                (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x4), 2), 1)) ** 2))))

              #secondSecSec_term_loss32.append(
              #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x4), 2), 1)) ** 2)).eval())

              secondSecSec_term_loss32.append((
                (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x5), 2), 1)) ** 2))))

              #secondSecSec_term_loss32.append(
              #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x5), 2), 1)) ** 2)).eval())

              secondSecSec_term_loss32.append((
                (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x6), 2), 1)) ** 2))))

              #secondSecSec_term_loss32.append(
              #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x6), 2), 1)) ** 2)).eval())

              secondSecSec_term_loss32.append((
                (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x7), 2), 1)) ** 2))))

              #secondSecSec_term_loss32.append(
              #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x7), 2), 1)) ** 2)).eval())

              secondSecSec_term_loss32.append((
                (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x8), 2), 1)) ** 2))))

              #secondSecSec_term_loss32.append(
              #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x8), 2), 1)) ** 2)).eval())

              secondSecSec_term_loss32.append((
                (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x9), 2), 1)) ** 2))))

              #secondSecSec_term_loss32.append(
              #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x9), 2), 1)) ** 2)).eval())

              secondSecSec_term_loss32.append((
                (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x99), 2), 1)) ** 2))))

              #secondSecSec_term_loss32.append(
              #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x99), 2), 1)) ** 2)).eval())

              # secondSecSec_term_loss32.append(
              #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2)).eval())

              # secondSecSec_term_loss32.append((tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x2), 2), 1)) ** 2)).eval())
              # secondSecSec_term_loss32.append(tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x3), 2), 1)) ** 2).eval())

              # secondSecSec_term_loss32.append(tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x4), 2), 1)) ** 2).eval())
              # secondSecSec_term_loss32.append(tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x5), 2), 1)) ** 2).eval())

              # secondSecSec_term_loss32[5] = tf.reduce_min(tf.sqrt((genFGen2[i, :] - x6).pow(2).sum(1)) ** 2)
              # secondSecSec_term_loss32[6] = tf.reduce_min(tf.sqrt((genFGen2[i, :] - x7).pow(2).sum(1)) ** 2)

              # secondSecSec_term_loss32[7] = tf.reduce_min(tf.sqrt((genFGen2[i, :] - x8).pow(2).sum(1)) ** 2)
              # secondSecSec_term_loss32[8] = tf.reduce_min(tf.sqrt((genFGen2[i, :] - x9).pow(2).sum(1)) ** 2)

              # secondSecSec_term_loss32[8] = torch.min(torch.sqrt((genFGen2[i, :] - x9).pow(2).sum(1)) ** 2)
              # secondSecSec_term_loss32[9] = tf.reduce_min(tf.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2)

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
              # second_term_loss32[i] = torch.argmin(secondSecSec_term_loss32)

              # second_term_loss32[i] = torch.argmin(secondSecSec_term_loss32)

              # secondSecSec_term_loss32 = tf.stack(secondSecSec_term_loss32)

              # second_term_loss32[i] = torch.argmin(secondSecSec_term_loss32)
              # second_term_loss32[i] = tf.argmin(tf.stack(secondSecSec_term_loss32))

              # second_term_loss32[i] = tf.argmin(tf.stack(secondSecSec_term_loss32))
              second_term_loss32.append(tf.argmin(tf.stack(secondSecSec_term_loss32)).eval())

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

            second_term_loss32 = tf.stack(second_term_loss32)

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

          second_term_loss32 = second_term_loss32.eval()

          plt.plot(second_term_loss32)
          plt.savefig('saveMySaSaSaSaveStore_second_term_loss32.png', bbox_inches='tight')

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
              if second_term_loss32[ii_loop21] == 0:
                  counterFor0 += 1
              elif second_term_loss32[ii_loop21] == 1:
                  counterFor1 += 1
              elif second_term_loss32[ii_loop21] == 2:
                  counterFor2 += 1
              elif second_term_loss32[ii_loop21] == 3:
                  counterFor3 += 1
              elif second_term_loss32[ii_loop21] == 4:
                  counterFor4 += 1
              elif second_term_loss32[ii_loop21] == 5:
                  counterFor5 += 1
              elif second_term_loss32[ii_loop21] == 6:
                  counterFor6 += 1
              elif second_term_loss32[ii_loop21] == 7:
                  counterFor7 += 1
              elif second_term_loss32[ii_loop21] == 8:
                  counterFor8 += 1
              elif second_term_loss32[ii_loop21] == 9:
                  counterFor9 += 1

          plt.figure()
          plt.plot(
              [counterFor0, counterFor1, counterFor2, counterFor3, counterFor4, counterFor5, counterFor6, counterFor7,
               counterFor8, counterFor9])
          plt.savefig('saveMySaSaveSaSaveSaSaveSaSaSaveStore_second_term_loss32.png', bbox_inches='tight')
          plt.savefig('NumberMyOfOccOccurences_vs_ClassesClusters.png', bbox_inches='tight')

          plt.figure()
          plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   [counterFor0, counterFor1, counterFor2, counterFor3, counterFor4, counterFor5, counterFor6,
                    counterFor7,
                    counterFor8, counterFor9], '--bo', linewidth=2, markersize=12)
          plt.ylabel('Number of modes')
          plt.xlabel('Modes')
          plt.savefig('NuMyNumberOfOccurences_vs_ClassesClusters.png', bbox_inches='tight')

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
          # plt.ylabel('Number of modes')
          plt.ylabel('Probability')
          plt.xlabel('Modes')
          plt.savefig('NumMyNumNumNumberOfOccurences_vs_ClassesClusters.png', bbox_inches='tight')
          plt.savefig('NumNikMyNumNumNumberOfOccurences_vs_ClassesClusters.png', bbox_inches='tight')

          asdfkfs

          sadf



          dasfasdfd



          #sadfsdfs

          #sfsadfs
          #asdfasd

          log_like_batch_idxs = len(data) // config.batch_size
          lli_list = []
          # inter_list = []
          for idx in xrange(0, log_like_batch_idxs):
            # batch_images = data[idx * config.batch_size:(idx + 1) * config.batch_size]

            # batch_images = data[idx * config.batch_size:(idx + 1) * config.batch_size]
            batch_images = data[idx * config.batch_size:(idx + 1) * config.batch_size]

            # batch_images = [batch_images, batch_images]
            # print(batch_images)

            batch_images = np.repeat(batch_images, 100, axis=0)
            # print(batch_images)

            # print(batch_images.shape)
            # asdfsafs

            # batch_images = data[idx * config.batch_size:(idx + 1) * config.batch_size]
            batch_images = np.cast[np.float32](batch_images)

            # batch_images = tf.reshape(tf.tile(batch_images, tf.constant([100], tf.int32)), [100, 784])
            # batch_images = tf.tile(batch_images, tf.constant([100, 1], tf.int32))

            # batch_images = tf.tile(batch_images, tf.constant([100, 1], tf.int32))
            # batch_images = np.cast[np.float32](batch_images)

            # print(batch_images.shape)
            # adfsadfs

            if self.model_type == "nice":
              batch_images = batch_images[:, self.permutation]

            # batch_images = tf.tile(batch_images, tf.constant([100, 1], tf.int32))
            # batch_images = np.cast[np.float32](batch_images)

            lli = self.sess.run([self.log_likelihood],
                                feed_dict={self.log_like_batch: batch_images})

            lli_list.append(lli)

          # lli_list = np.exp(lli_list)
          print(np.mean(lli_list))

          # print(lli_list)
          print(len(lli_list))

          # -329.59705
          # [[-3393.8728], [-3069.1543], [-3634.7078], [-3107.4153],

          # -329.59866
          # [[-3393.8723], [-3069.1543], [-3634.7078], [-3107.4153], [-3223.6433], [-3631.3137], [-3197.009],

          # llillllii_list = lli_list.max()
          # lllliilllilllii_list = np.exp(lli_list - llillllii_list)
          # firstOnly_lossGen2 =  lllliilllilllii_list / lllliilllilllii_list.sum()

          # firstOnly_lossGen2 = lli_list
          # firstOnly_lossGen2 = np.exp(lli_list)

          llillllii_list = np.max(lli_list)
          lllliilllilllii_list = np.exp(lli_list - llillllii_list)
          firstOnly_lossGen2 = lllliilllilllii_list / lllliilllilllii_list.sum()

          # 0.5000553955240417
          # 0.9026107910480833
          # 0.49994460447595834
          # 0.0974

          # firstOnly_lossGen2 = lli_list
          # 0.19189498646620234
          # 0.8326816764694991
          # 0.8081050135337977
          # 0.2199494240272631

          # firstOnly_lossGen2 = np.exp(lli_list)
          loLosses_NIKlosses2 = firstOnly_lossGen2

          # print(test_labels)
          # print(test_labels.shape)

          # print(test_labels)
          # print(test_labels.shape)

          # import numpy as np
          tesTest_labels = np.array(test_labels)

          # indices_one = tesTest_labels != 8
          # indices_zero = tesTest_labels == 8

          indices_one = tesTest_labels != 3
          indices_zero = tesTest_labels == 3

          tesTest_labels[indices_one] = 0  # replacing 1s with 0s
          tesTest_labels[indices_zero] = 1  # replacing 0s with 1s

          tesTest_labels = tesTest_labels.tolist()

          # print(tesTest_labels)
          # print(tesTest_labels.shape)

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

          loLosses_NIKlosses3 = tesTest_labels

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

          # print(loLoss2)
          # print(loLossNoChange)

          # print(len(loLoss2))
          # print(len(loLossNoChange))

          # adfasdfasdfsdfs

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

          # 0.5000553955240417
          # 0.9026107910480833
          # 0.49994460447595834
          # 0.0974

          # 0.5000553955240417
          # 0.9026107910480833
          # 0.49994460447595834
          # 0.0974

          asdfas

          sdfasfsfs



          #curr_model_path = self.save(config.checkpoint_dir, counter)
          curr_model_path = self.save(config.checkpoint_dir, 703)

          val_nlli=self.evaluate_neg_loglikelihood(val_data, config)

          # train_nlli = self.evaluate_neg_loglikelihood(train_data, config)
          curr_inception_score = self.calculate_inception_and_mode_score()

          print("[LogLi (%d,%d)]: val neg logli: %.8f, ince: %.8f, train lli: %.8f" % (epoch, idx,val_nlli,\
           curr_inception_score[0], np.mean(self.batch_train_nlli_list[-700:])))

          self.writer.add_summary(tf.Summary(\
                  value=[tf.Summary.Value(tag="Val Neg Log-likelihood", simple_value=val_nlli)]), counter)
          # self.writer.add_summary(tf.Summary(\
          #         value=[tf.Summary.Value(tag="Train Neg Log-likelihood", simple_value=train_nlli)]), counter)
          if val_nlli < self.best_val_nlli:
            self.best_val_nlli = val_nlli
            self.best_model_counter = counter
            self.best_model_path = curr_model_path
            # self.best_model_train_nlli = train_nlli
          # self.train_nlli_list.append(train_nlli)
          self.val_nlli_list.append(val_nlli)
          self.counter_list.append(counter)

          samples, d_loss, g_loss = self.sess.run(
            [self.sampler, self.d_loss, self.g_loss],
            feed_dict={
                self.z: sample_z,
                self.inputs: sample_inputs,
            }
          )
          self.samples_list.append(samples)
          self.samples_list[-1].shape[1]
          manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
          manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
          self.inception_list.append(curr_inception_score)
          save_images(samples, [manifold_h, manifold_w],
                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
          print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

          np.save("./"+config.sample_dir+'/graph_data', 
            [self.samples_list, self.val_nlli_list, self.best_val_nlli, self.best_model_counter,\
             self.best_model_path, self.counter_list, [manifold_h, manifold_w], \
             self.batch_train_nlli_list, self.inception_list, self.loss_list])

    
    np.save("./"+config.sample_dir+'/graph_data', 
            [self.samples_list, self.val_nlli_list, self.best_val_nlli, self.best_model_counter,\
             self.best_model_path, self.counter_list, [manifold_h, manifold_w], \
             self.batch_train_nlli_list, self.inception_list, self.loss_list])
    self.test_model(test_data, config)

  #def train2(self, config, dcgan2):
  def train2(self, config):
      seed = 0
      np.random.seed(seed)
      tf.set_random_seed(seed)
      """Train DCGAN"""
      if config.dataset == "mnist":
          data_X, val_data, test_data, train_dist = mnist_data.load_mnist()
      elif config.dataset == "cifar":
          data_X, val_data, test_data = cifar_data.load_cifar()

      if self.model_type == "nice":
          val_data = np.reshape(val_data, (-1, self.image_size))
          test_data = np.reshape(test_data, (-1, self.image_size))

      lr = config.learning_rate
      self.learning_rate = tf.placeholder(tf.float32, [], name='lr')

      d_optim_ = tf.train.AdamOptimizer(self.learning_rate, beta1=config.beta1, beta2=0.9)
      d_grad = d_optim_.compute_gradients(self.d_loss, var_list=self.d_vars)
      d_grad_mag = tf.global_norm(d_grad)
      d_optim = d_optim_.apply_gradients(d_grad)

      g_optim_ = tf.train.AdamOptimizer(self.learning_rate, beta1=config.beta1, beta2=0.9)
      if self.n_critic <= 0:
          g_grad = g_optim_.compute_gradients(self.train_log_likelihood \
                                              , var_list=self.g_vars)
      else:
          if self.like_reg > 0:
              if self.model_type == "real_nvp":
                  g_grad_1 = g_optim_.compute_gradients(self.g_loss / self.like_reg, var_list=self.g_vars)
                  g_grad_2 = g_optim_.compute_gradients(self.train_log_likelihood, var_list=self.g_vars)
                  grads_1, _ = zip(*g_grad_1)
                  grads_2, _ = zip(*g_grad_2)
                  sum_grad = [g1 + g2 for g1, g2 in zip(grads_1, grads_2)]
                  g_grad = [pair for pair in zip(sum_grad, [var for grad, var in g_grad_1])]
              else:
                  g_grad = g_optim_.compute_gradients(self.g_loss / self.like_reg + self.train_log_likelihood,
                                                      var_list=self.g_vars)
          else:
              g_grad = g_optim_.compute_gradients(self.g_loss, var_list=self.g_vars)

      g_grad_mag = tf.global_norm(g_grad)
      g_optim = g_optim_.apply_gradients(g_grad)

      try:  ##for data-dependent init (not implemented)
          if self.model_type == "real_nvp":
              self.sess.run(tf.global_variables_initializer(),
                            {self.x_init: data_X[0:config.batch_size]})
          else:
              self.sess.run(tf.global_variables_initializer())
      except:
          if self.model_type == "real_nvp":
              self.sess.run(tf.global_variables_initializer(),
                            {self.x_init: data_X[0:config.batch_size]})
          else:
              self.sess.run(tf.global_variables_initializer())

      self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                  self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
      self.d_sum = merge_summary(
          [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
      self.writer = SummaryWriter("./" + self.loLog_dir, self.sess.graph)

      counter = 1
      start_time = time.time()
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      # if could_load:
      #  counter = checkpoint_counter
      #  print(" [*] Load SUCCESS")
      # else:
      #  print(" [!] Load failed...")

      # if could_load:
      #  counter = checkpoint_counter
      #  print(" [*] Load SUCCESS")
      # else:
      #  print(" [!] Load failed...")

      print(could_load)
      print(checkpoint_counter)

      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")

      # dsafa
      # dfdasdf

      print(checkpoint_counter)
      print(checkpoint_counter)

      # counter = 1

      print(counter)
      print(counter)

      ############## A FIXED BATCH OF Zs FOR GENERATING SAMPLES ######################
      if self.prior == "uniform":
          sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
      elif self.prior == "logistic":
          sample_z = np.random.logistic(loc=0., scale=1., size=(self.sample_num, self.z_dim))
      elif self.prior == "gaussian":
          sample_z = np.random.normal(0.0, 1.0, size=(self.sample_num, self.z_dim))
      else:
          print("ERROR: Unrecognized prior...exiting")
          exit(-1)

      ################################ Evaluate initial model lli ########################

      val_nlli = self.evaluate_neg_loglikelihood(val_data, config)
      # train_nlli = self.evaluate_neg_loglikelihood(train_data, config)

      curr_inception_score = self.calculate_inception_and_mode_score()
      print("INITIAL TEST: val neg logli: %.8f,incep score: %.8f" % (val_nlli, \
                                                                     curr_inception_score[0]))
      if counter > 1:
          # old_data = np.load("./"+config.sample_dir+'/graph_data.npy')

          # old_data = np.load("./" + config.sample_dir + '/graph_data.npy')

          # old_data = np.load("./" + config.sample_dir + '/graph_data.npy')
          old_data = np.load("./" + config.sample_dir + '/graph_data.npy', allow_pickle=True)

          self.best_val_nlli = old_data[2]
          self.best_model_counter = old_data[3]
          self.best_model_path = old_data[4]
          self.val_nlli_list = old_data[1]
          self.counter_list = old_data[5]
          self.batch_train_nlli_list = old_data[-4]
          self.inception_list = old_data[-2]
          self.samples_list = old_data[0]
          self.loss_list = old_data[-1]
          manifold_h, manifold_w = old_data[6]
      else:
          self.writer.add_summary(tf.Summary( \
              value=[tf.Summary.Value(tag="Val Neg Log-likelihood", simple_value=val_nlli)]), counter)
          # self.writer.add_summary(tf.Summary(\
          #         value=[tf.Summary.Value(tag="Train Neg Log-likelihood", simple_value=train_nlli)]), counter)

          self.best_val_nlli = val_nlli
          # self.best_model_train_nlli = train_nlli
          self.best_model_counter = counter
          self.best_model_path = self.save(config.checkpoint_dir, counter)
          # self.train_nlli_list = [train_nlli]
          self.val_nlli_list = [val_nlli]
          self.counter_list = [1]
          self.batch_train_nlli_list = []
          self.inception_list = [curr_inception_score]
          self.samples_list = self.sess.run([self.sampler],
                                            feed_dict={
                                                self.z: sample_z,
                                            }
                                            )
          sample_inputs = data_X[0:config.batch_size]
          samples = self.samples_list[0]
          manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
          manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
          self.loss_list = self.sess.run(
              [self.d_loss_real, self.d_loss_fake],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
              })

          # print(sample_inputs)
          # adsfasdf
      ##################################################################################

      for epoch in xrange(config.epoch):
          np.random.shuffle(data_X)
          batch_idxs = len(data_X) // config.batch_size

          for idx in xrange(0, batch_idxs):
              sys.stdout.flush()
              batch_images = data_X[idx * config.batch_size:(idx + 1) * config.batch_size]

              print(len(batch_images))

              print(len(batch_images))
              print(len(batch_images))

              if self.prior == "uniform":
                  batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                      .astype(np.float32)
              elif self.prior == "logistic":
                  batch_z = np.random.logistic(loc=0., scale=1.0, size=[config.batch_size, self.z_dim]) \
                      .astype(np.float32)
              elif self.prior == "gaussian":
                  batch_z = np.random.normal(0.0, 1.0, size=(config.batch_size, self.z_dim))
              else:
                  print("ERROR: Unrecognized prior...exiting")
                  exit(-1)

              for r in range(self.n_critic):
                  _, d_g_mag, errD_fake, errD_real, summary_str = self.sess.run([d_optim, d_grad_mag,
                                                                                 self.d_loss_fake, self.d_loss_real,
                                                                                 self.d_sum],
                                                                                feed_dict={
                                                                                    self.inputs: batch_images,
                                                                                    self.z: batch_z,
                                                                                    self.learning_rate: lr,
                                                                                })
              if self.n_critic > 0:
                  self.writer.add_summary(summary_str, counter)

              # Update G network
              if self.like_reg > 0 or self.n_critic <= 0:
                  _, g_g_mag, errG, summary_str = self.sess.run([g_optim, g_grad_mag, self.g_loss, self.g_sum],
                                                                feed_dict={
                                                                    self.z: batch_z,
                                                                    self.learning_rate: lr,
                                                                    self.inputs: batch_images,
                                                                })
              else:
                  _, g_g_mag, errG, summary_str = self.sess.run([g_optim, g_grad_mag, self.g_loss, self.g_sum],
                                                                feed_dict={
                                                                    self.z: batch_z,
                                                                    self.learning_rate: lr,
                                                                })
              self.writer.add_summary(summary_str, counter)

              batch_images_nl = batch_images
              if self.model_type == "nice":
                  batch_images_nl = np.reshape(batch_images_nl, (self.batch_size, -1))[:, self.permutation]
              b_train_nlli = self.sess.run([self.log_likelihood], feed_dict={
                  self.log_like_batch: batch_images_nl,
              })
              b_train_nlli = b_train_nlli[0]

              self.batch_train_nlli_list.append(b_train_nlli)
              if self.n_critic > 0:
                  self.loss_list.append([errD_real, errD_fake])
                  self.writer.add_summary(tf.Summary( \
                      value=[tf.Summary.Value(tag="training loss", simple_value=-(errD_fake + errD_real))]), counter)
              self.writer.add_summary(tf.Summary( \
                  value=[tf.Summary.Value(tag="Batch train Neg Log-likelihood", simple_value=b_train_nlli)]), counter)
              counter += 1

              lr = max(lr * self.lr_decay, self.min_lr)

              if np.mod(counter, 703) == 1:  # 340
                  if self.n_critic > 0:
                      print(
                          "Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, d_grad_mag: %.8f, g_grad_mag: %.8f, lr: %.8f" \
                          % (epoch, idx, batch_idxs,
                             time.time() - start_time, errD_fake + errD_real, errG, d_g_mag, g_g_mag, lr))
                  else:
                      print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f, g_grad_mag: %.8f, lr: %.8f" \
                            % (epoch, idx, batch_idxs,
                               time.time() - start_time, errG, g_g_mag, lr))

                  # print(test_data)
                  # print(test_data.shape)

                  # test_nlli = self.evaluate_neg_loglikelihood(test_data, config)

                  # test_nlli = self.evaluate_neg_loglikelihood(test_data, config)
                  test_nlli = self.evaluate_neg_loglikelihood(test_data, config)

                  val_nlli = self.evaluate_neg_loglikelihood(val_data, config)
                  # train_nlli = self.evaluate_neg_loglikelihood(data_X, config)

                  # print(train_nlli)
                  # print(val_nlli)

                  # print(test_nlli)

                  # print(test_nlli)
                  # print(test_nlli)

                  # print(self.model_type)

                  # print(self.model_type)
                  # print(self.model_type)

                  data_X, _, _, _ = mnist_data.load_mnist()
                  data_X = np.reshape(data_X, (-1, self.image_size))

                  train_nlli = self.evaluate_neg_loglikelihood(data_X, config)
                  # print(train_nlli)

                  print(train_nlli)
                  print(val_nlli)

                  print(test_nlli)

                  test_nlli = self.evaluate_neg_loglikelihood2(test_data, config)
                  val_nlli = self.evaluate_neg_loglikelihood2(val_data, config)

                  train_nlli = self.evaluate_neg_loglikelihood2(data_X, config)

                  print(train_nlli)
                  print(val_nlli)

                  print(test_nlli)

                  # use: fashionmnist_data
                  # data_X, val_data, test_data, train_dist = mnist_data.load_mnist()

                  # print(data_X.shape)
                  # print(val_data.shape)

                  # print(val_data.shape)
                  # print(test_data.shape)

                  # data_X, val_data, test_data, train_dist = mnist_data.load_mnist()

                  # data_X, val_data, test_data, train_dist = mnist_data.load_mnist()
                  data_X, val_data, test_data, _ = fashionmnist_data.load_mnist()

                  data_X = np.reshape(data_X, (-1, self.image_size))
                  val_data = np.reshape(val_data, (-1, self.image_size))

                  # data_X = np.reshape(data_X, (-1, self.image_size))
                  test_data = np.reshape(test_data, (-1, self.image_size))

                  # print(data_X.shape)
                  # print(val_data.shape)

                  # print(val_data.shape)
                  # print(test_data.shape)

                  test_nlli = self.evaluate_neg_loglikelihood(test_data, config)
                  val_nlli = self.evaluate_neg_loglikelihood(val_data, config)

                  train_nlli = self.evaluate_neg_loglikelihood(data_X, config)

                  print(train_nlli)
                  print(val_nlli)

                  print(test_nlli)

                  test_nlli = self.evaluate_neg_loglikelihood2(test_data, config)
                  val_nlli = self.evaluate_neg_loglikelihood2(val_data, config)

                  train_nlli = self.evaluate_neg_loglikelihood2(data_X, config)

                  print(train_nlli)
                  print(val_nlli)

                  print(test_nlli)

                  # train_gen_para, train_jac = self.trainable_flow_model(inputs_tr_flow)
                  # self.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac, self.prior) / self.batch_size

                  # train_gen_para, train_jac = self.trainable_flow_model(inputs_tr_flow)
                  # self.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac, self.prior) / self.batch_size

                  # train_gen_para, train_jac = self.trainable_flow_model(inputs_tr_flow)
                  # self.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac, self.prior) / self.batch_size

                  # self.sampler_function(self.z)
                  # self.flow_inv_model

                  # train_gen_para, train_jac = self.trainable_flow_model(inputs_tr_flow)
                  # self.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac, self.prior) / self.batch_size

                  # self.flow_inv_model(self.z)
                  # use: self.flow_inv_model(self.z)

                  # train_nlli = self.evaluate_neg_loglikelihood2(data_X, config)
                  # use: train_nlli = self.evaluate_neg_loglikelihood2(data_X, config)

                  # we use: self.flow_inv_model(self.z)
                  # now use: self.flow_inv_model(self.z)

                  # first term: self.flow_inv_model(self.z)
                  # train_nlli = self.evaluate_neg_loglikelihood2(self.flow_inv_model(self.z), config)

                  # train_nlli = self.evaluate_neg_loglikelihood2(self.flow_inv_model(self.z), config)
                  
                  # train_nlli = self.evaluate_neg_loglikelihood2(self.flow_inv_model(self.z), config)
                  # train_nlli = self.evaluate_neg_loglikelihood2(self.flow_inv_model(self.z), config)
                  
                  #import torch

                  #nrand = 200
                  #netG = DCDCGANDCGenerator(nrand)

                  #print(config.batch_norm_adaptive)
                  #adfsadfs
                  
                  '''
                  dcgan2 = GANDCGAN2(
                     self.sess,
                     input_width=config.input_width,
                     input_height=config.input_height,
                     batch_size=config.batch_size,
                     sample_num=config.batch_size,
                     c_dim=config.c_dim,
                     z_dim=config.c_dim * config.input_height * config.input_width,
                     dataset_name=config.dataset,
                     checkpoint_dir=config.checkpoint_dir,
                     f_div=config.f_div,
                     prior=config.prior,
                     lr_decay=config.lr_decay,
                     min_lr=config.min_lr,
                     model_type=config.model_type,
                     loLog_dir=config.loLog_dir,
                     alpha=config.alpha,
                     batch_norm_adaptive=config.batch_norm_adaptive,
                     init_type=config.init_type,
                     reg=config.reg,
                     n_critic=0,
                     hidden_layers=config.hidden_layers,
                     no_of_layers=config.no_of_layers,
                     like_reg=config.like_reg,
                     df_dim=config.df_dim)
                  '''

                  #dcgan.train(FLAGS)
                  
                  #dcgan.train(FLAGS)
                  #dcgan2.train(config)

                  #dcgan2.train(config)
                  #dcgan2.train(config)
                  
                  #asdfszd

                  # train_nlli = self.evaluate_neg_loglikelihood2(self.flow_inv_model(self.z), config)
                  #train_nlli = self.evaluate_neg_loglikelihood22(self.flow_inv_model(self.z), config)

                  #train_nlli = self.evaluate_neg_loglikelihood22(self.flow_inv_model(self.z), config)

                  #train_nlli = self.evaluate_neg_loglikelihood22(self.flow_inv_model(self.z), config)
                  #train_nlli = self.evaluate_neg_loglikelihood22(self.flow_inv_model(self.z), config)

                  #print(train_nlli)
                  
                  #print(train_nlli)
                  #print(train_nlli)

                  data_X, _, _, _ = mnist_data.load_mnist()
                  #data_X = np.reshape(data_X, (-1, self.image_size))

                  #print(data_X.shape)
                  #asdfasdfs

                  #data_X, _, _, _ = mnist_data.load_mnist()
                  data_X = np.reshape(data_X, (-1, self.image_size))

                  #train_nlli = self.evaluate_neg_loglikelihood(data_X, config)
                  #print(train_nlli)

                  #train_nlli = self.evaluate_neg_loglikelihood2(data_X, config)
                  #print(train_nlli)

                  #print(data_X.shape)
                  
                  #print(data_X.shape)
                  #print(data_X.shape)

                  #sadfasdfa
                  
                  #args.device = device
                  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                  
                  #print(device)
                  
                  #print(device)
                  #print(device)
                  
                  epEpochs = 1000

                  loss_theLoss = torch.empty(epEpochs, device=device)
                  loss_theLoss0 = torch.empty(epEpochs, device=device)

                  loss_theLoss1 = torch.empty(epEpochs, device=device)
                  loss_theLoss2 = torch.empty(epEpochs, device=device)

                  loss_theLoss3 = torch.empty(epEpochs, device=device)
                  
                  X_training = data_X
                  bsz = len(X_training)

                  #batch_size = 1024
                  batch_size = 100
                  
                  nrand = 200
                  #netG = DCDCGANDCGenerator(nrand).to(device)

                  #netG = DCDCGANDCGenerator(nrand).to(device)
                  #netG = DCDCGANDCGenerator2(28, nrand, 28, 1).to(device)

                  #G = generator(input_size=100, n_class=28*28)
                  #netG = generator(input_size=nrand, n_class=28 * 28).to(device)

                  #netG = generator(input_size=nrand, n_class=28 * 28).to(device)

                  #netG = generator(input_size=nrand, n_class=28 * 28).to(device)
                  #netG = generator(input_size=nrand, n_class=28 * 28).to(device)

                  # Option 1
                  #netG = generator(input_size=nrand, n_class=28 * 28).to(device)

                  # Option 2
                  netG = GeGenerator(1, nrand, 64, 1).to(device)
                  
                  #netG = generator(input_size=nrand, n_class=28 * 28).to(device)
                  #netG = GeGenerator(1).to(device)
                  
                  for epoch in range(1, epEpochs + 1):
                    for i in range(0, len(X_training), bsz):
                      #sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize)

                      #netD.zero_grad()

                      #netD.zero_grad()
                      #netD.zero_grad()

                      stop = min(bsz, len(X_training[i:]))
                      #real_cpu = X_training[i:i + stop].to(device)

                      #img = torch.from_numpy(img).float().to(device)

                      #img = torch.from_numpy(img).float().to(device)
                      #img = torch.from_numpy(img).float().to(device)

                      #real_cpu = X_training[i:i + stop].to(device)
                      real_cpu = torch.from_numpy(X_training[i:i + stop]).float().to(device)

                      # print(real_cpu.shape)
                      # asdfasdf

                      # batch_size = real_cpu.size(0)
                      #batch_size = args.batchSize

                      #label = torch.full((batch_size,), real_label, device=device)

                      #label = torch.full((batch_size,), real_label, device=device)
                      #label = torch.full((batch_size,), real_label, device=device)

                      #noise_eta = torch.randn_like(real_cpu)
                      #noised_data = real_cpu + sigma_x.detach() * noise_eta

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
                      #sigma_optimizer.zero_grad()

                      #label.fill_(real_label)
                      #gen_input = torch.randn(batch_size, args.nz, 1, 1, device=device)

                      #gen_input = torch.randn(batch_size, nrand, 1, 1, device=device)
                      #out = netG(gen_input)

                      #gen_input = torch.randn(batch_size, nrand, device=device)
                      #out = netG(gen_input)

                      #gen_input = torch.randn(batch_size, nrand, 1, 1, device=device)

                      #gen_input = torch.randn(batch_size, nrand, 1, 1, device=device)
                      gen_input = torch.randn(batch_size, nrand, 1, 1, device=device, requires_grad=True)

                      #gen_input = torch.randn(batch_size, nrand, 1, 1, device=device)
                      out = netG(gen_input)

                      # print(out.shape)
                      # asdfasdf
                      
                      #print(out.shape)
                      #print(np.shape(data_X))

                      #print(out.shape)

                      #print(out.shape)
                      #print(out.shape)
                      
                      #torch.Size([1024, 784])
                      #torch.Size([1024, 1, 28, 28])

                      #print(out.shape)

                      #print(out.shape)
                      #print(out.shape)

                      #out = np.reshape(out, (-1, self.image_size))
                      out = torch.reshape(out, (-1, self.image_size))

                      #print(out.shape)
                      #print(out.shape)
                      
                      #print(out.shape)
                      #asfsafasdfaf

                      #noise_eta = torch.randn_like(out)
                      #g_fake_data = out + noise_eta * sigma_x

                      #dg_fake_decision = netD(g_fake_data)
                      #g_error_gan = criterion(dg_fake_decision, label)
                      #D_G_z2 = dg_fake_decision.mean().item()

                      #if args.lambda_ == 0:
                      #  g_error_gan.backward()
                      #  optimizerG.step()
                      #  sigma_optimizer.step()

                      #p_probP =

                      #p_probP =
                      #p_probP =

                      #train_nlli = self.evaluate_neg_loglikelihood2(data_X, config)

                      #train_nlli = self.evaluate_neg_loglikelihood2(data_X, config)
                      #train_nlli = self.evaluate_neg_loglikelihood2(data_X, config)

                      #train_nlli = self.evaluate_neg_loglikelihood2(data_X, config)
                      #p_probP = self.evaluate_neg_loglikelihood2(out, config)

                      #p_probP = self.evaluate_neg_loglikelihood2(out, config)
                      p_probP = self.evaluate_neg_loglikelihood22(out, config)
                      
                      #print(p_probP)

                      print(p_probP)
                      #print(p_probP)

                      #print(out)
                      #print(out.shape)

                      print(out.shape)
                      print(out.requires_grad)

                      # batch_images is batch_size x 784
                      batch_images = out
                      
                      # torch.from_numpy(np.asarray(x))
                      # use: torch.from_numpy(np.asarray(x))

                      myNik_nlli = torch.from_numpy(np.asarray((
                        self.sess.run([self.log_likelihood],
                                      feed_dict={self.log_like_batch: batch_images.cpu().detach().numpy()}))))

                      #myNik_nlli = torch.from_numpy((
                      #  self.sess.run([self.log_likelihood],
                      #                feed_dict={self.log_like_batch: batch_images.cpu().detach().numpy()})))

                      #myNik_nlli = torch.from_numpy(np.gradient(
                      #  self.sess.run([self.log_likelihood], feed_dict={self.log_like_batch: batch_images.cpu().detach().numpy()})))

                      print(myNik_nlli)
                      #print(myNik_nlli.shape)
                      
                      #print(myNik_nlli.shape)
                      #print(myNik_nlli.shape)

                      print(myNik_nlli.shape)
                      print(myNik_nlli.requires_grad)

                      
                      
                      sdfsdadsf
                      
                      #asdfasf
                      #adsfasfs
                      
                      #asdfasd
                      #adfaszkx
                      
                      

                      #_, _, _, p_probP = hmc.get_samples(
                      #  netG2, netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)),
                      #  gen_input.clone(), sigma_x.detach(), args.burn_in,
                      #  args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                      #  args.hmc_learning_rate, args.hmc_opt_accept)

                      #_, _, _, p_probP = hmc.get_samples(
                      #  netG2, netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)),
                      #  gen_input.clone(), sigma_x.detach(), args.burn_in,
                      #  args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                      #  args.hmc_learning_rate, args.hmc_opt_accept)

                      # _, _, _, p_probP = hmc.get_samples(
                      #    netG2, X_training[0:0+64].to(device),
                      #    gen_input.clone(), sigma_x.detach(), args.burn_in,
                      #    args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt,
                      #    args.hmc_learning_rate, args.hmc_opt_accept)

                      # print(p_probP.mean())
                      # print(p_probP.mean().grad)

                      # print(p_probP.mean().requires_grad)
                      # sadfasdfks

                      # print(p_probP.mean())
                      # print(p_probP.mean().requires_grad)

                      # asdfsadfs

                      # print(p_probP.mean())
                      # print(p_probP.mean().requires_grad)

                      # asdfasfdfs

                      #  -609010.3125
                      # -1401163.0000

                      # print(p_probP.mean())
                      # print(p_probP.mean().requires_grad)

                      # asdfasdfs

                      # print(netG(torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)).requires_grad)
                      # print(netG2(torch.randn(batch_size, args.nz, 1, 1, device=device)).requires_grad)

                      # print(p_probP.mean())
                      # print(p_probP.requires_grad)
                      # print(p_probP.shape)

                      # print(p_probP.mean())
                      # print(p_probP.mean().requires_grad)

                      # g_error = p_probP.mean() + (?)
                      # use: g_error = p_probP.mean() + (?)

                      # asdfsadf

                      # g_error = (?)
                      # g_error = p_probP.mean() + (?)

                      # print(p_probP.mean())
                      # print(p_probP.mean().requires_grad)

                      # g_error = p_probP.mean() +
                      # we use: g_error = p_probP.mean() +

                      # p_probP.mean() +
                      # g_error = p_probP.mean() +

                      # we now use: p_probP.mean()
                      # firstTerm_theFirstTerm = p_probP.mean()

                      varInIn = torch.randn(batch_size, args.nz, 1, 1, requires_grad=True, device=device)
                      varOutOut = netG(varInIn)

                      g_error, firstOnly_lossGen, secondOnly_lossGen, thirdOnly_lossGen = use_loss_fn2(p_probP.mean(),
                                                                                                       varOutOut,
                                                                                                       args, netG2,
                                                                                                       varInIn,
                                                                                                       real_cpu.to(
                                                                                                         device))

                      # print(firstOnly_lossGen)
                      # print(secondOnly_lossGen)

                      # print(thirdOnly_lossGen)
                      # print(g_error)

                      # asdfadsfas

                      # firstTerm_theFirstTerm = p_probP.mean()
                      # g_error = firstTerm_theFirstTerm +

                      # g_error => Use netG( 64, 100, 1, 1 )
                      # gen_input = torch.randn(batch_size, args.nz, 1, 1, device=device)

                      # (?)
                      # g_error = (?)

                      # g_error = g_error_gan - args.lambda_ * g_error_entropy
                      g_error.backward()

                      gradGrad_lossGen = 1.0 / torch.mean(netG.main[0].weight.grad).item()

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

                        loss_theLoss[epoch - 1] = g_error.item()
                        loss_theLoss0[epoch - 1] = firstOnly_lossGen.item()

                        loss_theLoss1[epoch - 1] = secondOnly_lossGen.item()
                        loss_theLoss2[epoch - 1] = thirdOnly_lossGen.item()

                        loss_theLoss3[epoch - 1] = gradGrad_lossGen

                        # print(
                        #    'Epoch [%d/%d] .. Batch [%d/%d] .. Loss: %.4f .. L0: %.4f .. L1: %.4f .. D(G(z)): %.4f / %.4f'
                        #    % (epoch, args.epochs, i, len(X_training), g_error.item(), firstOnly_lossGen.item(),
                        #       secondOnly_lossGen.item(), thirdOnly_lossGen.item(), thirdOnly_lossGen.item()))

                        # print(
                        #    'Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
                        #    % (epoch, args.epochs, i, len(X_training), errD.data, g_error_gan.data, D_x, D_G_z1, D_G_z2))

                        # print('Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
                        #        % (epoch, args.epochs, i, len(X_training), errD.data, g_error_gan.data, D_x, D_G_z1, D_G_z2))

                    # print('*'*100)
                    # print('End of epoch {}'.format(epoch))
                    # print('sigma min: {} .. sigma max: {}'.format(torch.min(sigma_x), torch.max(sigma_x)))
                    # print('*'*100)
                    # if args.lambda_ > 0:
                    #    print('| MCMC diagnostics ====> | stepsize: {} | min ar: {} | mean ar: {} | max ar: {} |'.format(
                    #                stepsize, acceptRate.min().item(), acceptRate.mean().item(), acceptRate.max().item()))

                    if epoch % args.save_imgs_every == 0:
                      import matplotlib.pyplot as plt
                      plt.figure()
                      plt.plot(loss_theLoss.cpu())
                      plt.xlim(0, epoch - 1)
                      plt.savefig('neNewLossLoss_plot')

                      plt.figure()
                      plt.plot(loss_theLoss0.cpu())
                      plt.xlim(0, epoch - 1)
                      plt.savefig('neNewLossLoss0_plot')

                      plt.figure()
                      plt.plot(loss_theLoss1.cpu())
                      plt.xlim(0, epoch - 1)
                      plt.savefig('neNewLossLoss1_plot')

                      plt.figure()
                      plt.plot(loss_theLoss2.cpu())
                      plt.xlim(0, epoch - 1)
                      plt.savefig('neNewLossLoss2_plot')

                      plt.figure()
                      plt.plot(loss_theLoss3.cpu())
                      plt.xlim(0, epoch - 1)
                      plt.savefig('neNewLossNewLoLossLoLossLoLossLoss2_plot')

                      plt.figure()
                      fig, axs = plt.subplots(2, 2)
                      axs[0, 0].plot(range(1, 1 + epoch), loss_theLoss[:epoch].cpu())
                      axs[0, 0].set_title('Loss')
                      axs[0, 1].plot(range(1, 1 + epoch), loss_theLoss0[:epoch].cpu(), 'tab:orange')
                      axs[0, 1].set_title('L0')
                      axs[1, 0].plot(range(1, 1 + epoch), loss_theLoss1[:epoch].cpu(), 'tab:green')
                      axs[1, 0].set_title('L1')
                      axs[1, 1].plot(range(1, 1 + epoch), loss_theLoss2[:epoch].cpu(), 'tab:red')
                      axs[1, 1].set_title('L2')
                      plt.savefig('neNewLossLossTotal_plot')

                      for ax in axs.flat:
                        ax.set(xlabel='x-label', ylabel='y-label')

                      # Hide x labels and tick labels for top plots and y ticks for right plots
                      for ax in axs.flat:
                        ax.label_outer()

                      fake = netG2(fixed_noise).detach()
                      vutils.save_image(fake, '%s/presgan_%s_fake_epoch_%03d.png' % (
                      args.results_folder, args.dataset, epoch),
                                        normalize=True, nrow=20)

                      fake = netG(fixed_noise).detach()
                      vutils.save_image(fake, '%s/presgan_%s_faFake_epoch_%03d.png' % (
                      args.results_folder, args.dataset, epoch),
                                        normalize=True, nrow=20)

                    if epoch % args.save_ckpt_every == 0:
                      torch.save(netG.state_dict(),
                                 os.path.join(args.results_folder,
                                              'netG_presgan_%s_epoch_%s.pth' % (args.dataset, epoch)))
                      torch.save(log_sigma,
                                 os.path.join(args.results_folder, 'log_sigma_%s_%s.pth' % (args.dataset, epoch)))



                  asdfszd

                  

                  # _, _, _, _, _, test_labels = mnist_data.load_mnist(send_labels=True)

                  # _, _, _, _, _, test_labels = mnist_data.load_mnist(send_labels=True)
                  # _, _, _, trTrTrTrainLabels, _, test_labels = mnist_data.load_mnist(send_labels=True)

                  # data_X, val_data, test_data, train_dist = mnist_data.load_mnist()

                  # data_X, val_data, test_data, train_dist = mnist_data.load_mnist()
                  # data_X, val_data, test_data, train_dist = mnist_data.load_mnist()

                  # data_X, val_data, test_data, train_dist = mnist_data.load_mnist()
                  # _, _, _, trTrTrTrainLabels, _, test_labels = mnist_data.load_mnist(send_labels=True)

                  data_X, _, _, trTrTrTrainLabels, _, test_labels = mnist_data.load_mniMnist(send_labels=True)

                  data = test_data
                  config.batch_size = 1

                  losses_NIKlosses = []

                  loLosses_NIKlosses = []
                  loLosses_NIKlosses2 = []

                  # loLosses_NIKlosses2 = []
                  loLosses_NIKlosses3 = []

                  # print(data_X)
                  # print(data_X.shape)

                  # print(data)
                  # print(data.shape)

                  # dsfsazdf

                  # print(data_X.shape)
                  # print(data.shape)

                  # data = tf.reshape(data_X, [-1, 784])

                  # data = tf.reshape(data_X, [-1, 784])
                  # data = tf.reshape(data_X, [-1, 784])

                  # (?)
                  # data = tf.reshape(data_X, [-1, 784])
                  # (?)

                  # print(data_X.shape)
                  # print(data.shape)

                  x = tf.reshape(data_X, [-1, 784])
                  # x = x.to(device)

                  y = trTrTrTrainLabels
                  # y = y.to(device)

                  # print(x.shape)
                  # print(y.shape)

                  # dasfsadfazs

                  # config.batch_size = 1024
                  # config.batch_size = 16384

                  # config.batch_size = 1024
                  # config.batch_size = 2048

                  # config.batch_size = 2048
                  # config.batch_size = 150

                  # config.batch_size = 2048
                  # config.batch_size = 2 * 2048

                  # config.batch_size = 2 * 2048
                  # config.batch_size = 2048

                  config.batch_size = 100

                  # config.batch_size = 1024
                  # ggenFGen2 = torch.randn([config.batch_size, nrand], device=device, requires_grad=True)

                  # genFGen2 = genGen.forward(ggenFGen2)
                  # genFGen2 = genGen.forward(ggenFGen2)

                  # ggenFGen2 = torch.randn([config.batch_size, nrand], device=device, requires_grad=True)
                  # genFGen2 = genGen.forward(ggenFGen2)

                  # ggenFGen2 = torch.randn([config.batch_size, nrand], device=device)
                  # genFGen2 = genGen.forward(ggenFGen2)

                  # lli = self.sess.run([self.log_likelihood],
                  #                    feed_dict={self.log_like_batch: batch_images})

                  # sample_z = np.random.logistic(loc=0., scale=1., size=(self.sample_num, self.z_dim))
                  # sample_z = np.random.logistic(loc=0., scale=1., size=(config.batch_size, self.z_dim))

                  batch_z = np.random.logistic(loc=0., scale=1.0, size=[config.batch_size, self.z_dim]) \
                      .astype(np.float32)

                  ggenFGen2 = batch_z
                  # ggenFGen2 = torch.randn([config.batch_size, 100, 1, 1], device=device)

                  # genFGen2 = self.sess.run([self.G],
                  #                         feed_dict={self.z: sample_z, self.batch_size: config.batch_size})

                  genFGen2 = self.sess.run(
                      [self.sampler],
                      feed_dict={
                          self.z: batch_z,
                      }
                  )

                  # np.concatenate((a, b), axis=0)

                  # np.concatenate((a, b), axis=0)
                  # np.concatenate((a, b), axis=0)

                  # print(np.shape(ggenFGen2))
                  # print(np.shape(genFGen2))

                  ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
                      .astype(np.float32)

                  ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
                  genFGen2 = np.concatenate((genFGen2, self.sess.run(
                      [self.sampler],
                      feed_dict={
                          self.z: ggenGen2FGen2,
                      }
                  )), axis=1)

                  config.batch_size += 100

                  # print(np.shape(ggenFGen2))
                  # print(np.shape(genFGen2))

                  # dasfasdfsdf

                  ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
                      .astype(np.float32)

                  ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
                  genFGen2 = np.concatenate((genFGen2, self.sess.run(
                      [self.sampler],
                      feed_dict={
                          self.z: ggenGen2FGen2,
                      }
                  )), axis=1)

                  config.batch_size += 100

                  ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
                      .astype(np.float32)

                  ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
                  genFGen2 = np.concatenate((genFGen2, self.sess.run(
                      [self.sampler],
                      feed_dict={
                          self.z: ggenGen2FGen2,
                      }
                  )), axis=1)

                  config.batch_size += 100

                  ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
                      .astype(np.float32)

                  ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
                  genFGen2 = np.concatenate((genFGen2, self.sess.run(
                      [self.sampler],
                      feed_dict={
                          self.z: ggenGen2FGen2,
                      }
                  )), axis=1)

                  config.batch_size += 100

                  ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
                      .astype(np.float32)

                  ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
                  genFGen2 = np.concatenate((genFGen2, self.sess.run(
                      [self.sampler],
                      feed_dict={
                          self.z: ggenGen2FGen2,
                      }
                  )), axis=1)

                  config.batch_size += 100

                  ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
                      .astype(np.float32)

                  ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
                  genFGen2 = np.concatenate((genFGen2, self.sess.run(
                      [self.sampler],
                      feed_dict={
                          self.z: ggenGen2FGen2,
                      }
                  )), axis=1)

                  config.batch_size += 100

                  ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
                      .astype(np.float32)

                  ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
                  genFGen2 = np.concatenate((genFGen2, self.sess.run(
                      [self.sampler],
                      feed_dict={
                          self.z: ggenGen2FGen2,
                      }
                  )), axis=1)

                  config.batch_size += 100

                  ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
                      .astype(np.float32)

                  ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
                  genFGen2 = np.concatenate((genFGen2, self.sess.run(
                      [self.sampler],
                      feed_dict={
                          self.z: ggenGen2FGen2,
                      }
                  )), axis=1)

                  config.batch_size += 100

                  ggenGen2FGen2 = np.random.logistic(loc=0., scale=1.0, size=[100, self.z_dim]) \
                      .astype(np.float32)

                  ggenFGen2 = np.concatenate((ggenFGen2, ggenGen2FGen2), axis=0)
                  genFGen2 = np.concatenate((genFGen2, self.sess.run(
                      [self.sampler],
                      feed_dict={
                          self.z: ggenGen2FGen2,
                      }
                  )), axis=1)

                  config.batch_size += 100

                  # print(np.shape(ggenFGen2))
                  # print(np.shape(genFGen2))

                  # dasfasdfsdf

                  print(np.shape(ggenFGen2))

                  print(np.shape(ggenFGen2))
                  print(np.shape(ggenFGen2))

                  # samples_curr = self.sess.run(
                  #  [self.sampler],
                  #  feed_dict={
                  #    self.z: batch_z, }
                  # )

                  # genFGen2 = self.sess.run([self.G],
                  #                         feed_dict={self.z: sample_z})

                  # genFGen2 = tf.reshape(genFGen2, [config.batch_size, 784])
                  genFGen2 = np.reshape(genFGen2, (config.batch_size, 784))

                  # print(genFGen2)
                  print(np.shape(genFGen2))

                  print(np.shape(genFGen2))
                  print(np.shape(genFGen2))

                  # print(genFGen2.shape)
                  # asdfasdf

                  print(x.shape)
                  print(y.shape)

                  # asdfasdfas

                  # genFGen2 = self.sess.run([self.G],
                  #                    feed_dict={self.log_like_batch: batch_images})

                  # with torch.no_grad():
                  #    ggenFGen2 = torch.randn([config.batch_size, 100, 1, 1], device=device)
                  #    # genFGen2 = netG.forward(ggenFGen2)

                  #    # genFGen2 = netG.forward(ggenFGen2)

                  #    # genFGen2 = netG.forward(ggenFGen2)
                  #    # genFGen2 = netG.forward(ggenFGen2)

                  #    genFGen2 = netG.forward(ggenFGen2)
                  #    # genFGen2 = netG2.forward(ggenFGen2)

                  # print(x.shape)
                  # print(y.shape)

                  # print(genFGen2.shape)
                  # print(config.batch_size)

                  # for i21 in range(len(y)):
                  #    if y[i21] == 0 and i21 == 0:
                  #        y[i21] = y[i21+1]
                  #        x[i21, :, :, :] = x[i21+1, :, :, :]
                  #    elif y[i21] == 0:
                  #        y[i21] = y[i21 - 1]
                  #        x[i21, :, :, :] = x[i21 - 1, :, :, :]

                  # import torch

                  # y2 = []
                  x2 = []
                  for i21 in range(len(y)):
                      if y[i21] == 1:
                          # y2.append(y[i21])
                          # x2.append(x[i21, :, :, :])

                          # x2.append(x[i21, :, :, :])
                          x2.append(x[i21, :])

                  # x2 = tf.stack(x2)
                  x2 = tf.stack(x2)
                  # y2 = torch.stack(y2)

                  # y3 = []
                  x3 = []
                  for i21 in range(len(y)):
                      if y[i21] == 2:
                          # y3.append(y[i21])
                          # x3.append(x[i21, :, :, :])

                          # x3.append(x[i21, :, :, :])
                          x3.append(x[i21, :])

                  # x3 = tf.stack(x3)
                  x3 = tf.stack(x3)
                  # y3 = torch.stack(y3)

                  # y4 = []
                  x4 = []
                  for i21 in range(len(y)):
                      if y[i21] == 3:
                          # y4.append(y[i21])
                          # x4.append(x[i21, :, :, :])

                          # x4.append(x[i21, :, :, :])
                          x4.append(x[i21, :])

                  x4 = tf.stack(x4)
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
                          # x5.append(x[i21, :, :, :])

                          # x5.append(x[i21, :, :, :])
                          x5.append(x[i21, :])

                  x5 = tf.stack(x5)
                  # y5 = torch.stack(y5)

                  # y6 = []
                  x6 = []
                  for i21 in range(len(y)):
                      if y[i21] == 5:
                          # y6.append(y[i21])
                          # x6.append(x[i21, :, :, :])

                          # x6.append(x[i21, :, :, :])
                          x6.append(x[i21, :])

                  x6 = tf.stack(x6)
                  # y6 = torch.stack(y6)

                  # y7 = []
                  x7 = []
                  for i21 in range(len(y)):
                      if y[i21] == 6:
                          # y7.append(y[i21])
                          # x7.append(x[i21, :, :, :])

                          # x7.append(x[i21, :, :, :])
                          x7.append(x[i21, :])

                  x7 = tf.stack(x7)
                  # y7 = torch.stack(y7)

                  # y8 = []
                  x8 = []
                  for i21 in range(len(y)):
                      if y[i21] == 7:
                          # y8.append(y[i21])
                          # x8.append(x[i21, :, :, :])

                          # x8.append(x[i21, :, :, :])
                          x8.append(x[i21, :])

                  x8 = tf.stack(x8)
                  # y8 = torch.stack(y8)

                  # y9 = []
                  x9 = []
                  for i21 in range(len(y)):
                      if y[i21] == 8:
                          # y9.append(y[i21])
                          # x9.append(x[i21, :, :, :])

                          # x9.append(x[i21, :, :, :])
                          x9.append(x[i21, :])

                  x9 = tf.stack(x9)
                  # y9 = torch.stack(y9)

                  # y99 = []
                  x99 = []
                  for i21 in range(len(y)):
                      if y[i21] == 9:
                          # y99.append(y[i21])
                          # x99.append(x[i21, :, :, :])

                          # x99.append(x[i21, :, :, :])
                          x99.append(x[i21, :])

                  x99 = tf.stack(x99)
                  # y99 = torch.stack(y99)

                  x999 = []
                  for i21 in range(len(y)):
                      if y[i21] == 0:
                          # x999.append(x[i21, :, :, :])

                          # x999.append(x[i21, :, :, :])
                          x999.append(x[i21, :])
                  x999 = tf.stack(x999)

                  # print(x9.shape)
                  # print(x99.shape)
                  # print(genFGen2.shape)

                  # print(x999.shape)
                  # asdfasdfs

                  '''
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
                  # x3 = x3.view(-1, 64 * 64)
        
                  # x3 = x3.view(-1, 64 * 64)
        
                  # x3 = x3.view(-1, 64 * 64)
                  # x3 = x3.view(-1, 64 * 64)
        
                  x3 = x3.view(-1, 64 * 64)
        
                  x2 = x2.view(-1, 64 * 64)
                  # x8 = x8.view(-1, 64 * 64)
                  '''

                  # print(config.batch_size)
                  # print(genFGen2.shape)

                  x999 = x999.eval()

                  x99 = x99.eval()
                  x9 = x9.eval()

                  x8 = x8.eval()
                  x7 = x7.eval()

                  x6 = x6.eval()
                  x5 = x5.eval()

                  x4 = x4.eval()
                  x3 = x3.eval()

                  x2 = x2.eval()

                  # with torch.no_grad():

                  # print(config.batch_size)
                  # asdfasdfs

                  with tf.device('/gpu:0'):
                      # second_term_loss32 = torch.empty(args.batch_size, device=device, requires_grad=False)
                      # second_term_loss32 = torch.empty(config.batch_size, device=device, requires_grad=False)
                      # second_term_loss32 = torch.empty(config.batch_size, device=device)
                      # second_term_loss32 = tf.zeros(config.batch_size)
                      second_term_loss32 = []
                      # for i in range(args.batch_size):
                      for i in range(config.batch_size):
                          print(i)

                          """
                          print(torch.mean(torch.sqrt((genFGen2[i, :] - xData).view(config.batch_size, -1).pow(2).sum(1))))
                          print(torch.mean(torch.sqrt((genFGen2[i, :] - genFGen2).view(config.batch_size, -1).pow(2).sum(1))))
                          print(torch.mean(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1))))
                          print('')
            
                          print(torch.mean(torch.norm((genFGen2[i, :] - xData).view(config.batch_size, -1), p=None, dim=1)))
                          print(torch.mean(torch.norm((genFGen2[i, :] - genFGen2).view(config.batch_size, -1), p=None, dim=1)))
                          print(torch.mean(torch.norm((genFGen3[i, :] - genFGen3), p=None, dim=1)))
                          print('')
                          """

                          # print(torch.mean(torch.sqrt((genFGen2[i, :] - xData).view(config.batch_size, -1).pow(2).sum(1))))
                          # print(torch.mean(torch.sqrt((genFGen2[i, :] - genFGen2).view(config.batch_size, -1).pow(2).sum(1))))
                          # print(torch.mean(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1))))
                          # print('')

                          # print(torch.sqrt((genFGen2[i, :] - xData).view(config.batch_size, -1).pow(2).sum(1)))
                          # print(torch.sqrt((genFGen2[i, :] - genFGen2).view(config.batch_size, -1).pow(2).sum(1)))
                          # print(torch.sqrt((genFGen3[i, :] - genFGen3).pow(2).sum(1)))
                          # print('')

                          # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p='fro', dim=1).requires_grad_()
                          # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()
                          # second_term_loss22 = torch.norm(genFGen2[i, :] - xData, p=None, dim=1).requires_grad_()**2
                          # second_term_loss22 = torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1))**2
                          # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_()**2

                          # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2

                          # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).pow(2).sum(1)).requires_grad_() ** 2
                          # second_term_loss22 = torch.sqrt(1e-17 + (genFGen2[i, :] - xData).view(config.batch_size, -1).pow(2).sum(1)).requires_grad_() ** 2

                          # second_term_loss22 = torch.sqrt(
                          #    1e-17 + (genFGen2[i, :] - xData).view(config.batch_size, -1).pow(2).sum(1)).requires_grad_() ** 2

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

                          # secondSecSec_term_loss32 = tf.zeros(10)
                          secondSecSec_term_loss32 = []

                          # secondSecSec_term_loss32 = tf.zeros(10)
                          # secondSecSec_term_loss32[8] = torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2

                          # secondSecSec_term_loss32[8] = torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2
                          # secondSecSecSec_term_loss32 = torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2

                          # secondSecSec_term_loss32[8] = torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2

                          # secondSecSec_term_loss32[8] = torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2
                          # secondSecSec_term_loss32[8] = torch.min(torch.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2)

                          # secondSecSec_term_loss32[7] = torch.min(torch.sqrt((genFGen2[i, :] - x9).pow(2).sum(1)) ** 2)
                          # secondSecSec_term_loss32[6] = torch.min(torch.sqrt((genFGen2[i, :] - x8).pow(2).sum(1)) ** 2)

                          # secondSecSec_term_loss32[5] = torch.min(torch.sqrt((genFGen2[i, :] - x7).pow(2).sum(1)) ** 2)
                          # secondSecSec_term_loss32[4] = torch.min(torch.sqrt((genFGen2[i, :] - x6).pow(2).sum(1)) ** 2)

                          # secondSecSec_term_loss32[3] = torch.min(torch.sqrt((genFGen2[i, :] - x5).pow(2).sum(1)) ** 2)
                          # secondSecSec_term_loss32[2] = torch.min(torch.sqrt((genFGen2[i, :] - x4).pow(2).sum(1)) ** 2)

                          # secondSecSec_term_loss32[1] = torch.min(torch.sqrt((genFGen2[i, :] - x3).pow(2).sum(1)) ** 2)
                          # secondSecSec_term_loss32[0] = torch.min(torch.sqrt((genFGen2[i, :] - x2).pow(2).sum(1)) ** 2)

                          # print(secondSecSec_term_loss32)
                          # print(torch.min(torch.sqrt((genFGen2[i, :] - x999).pow(2).sum(1)) ** 2))

                          # use: x999
                          # secondSecSec_term_loss32.append((tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2)).eval())

                          # print((
                          #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2)).eval()))

                          # x999 = x999.eval()

                          # print((
                          #   (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x999), 2), 1)) ** 2))))

                          # print((
                          #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2))))

                          # print(genFGen2[i, :])
                          # print(x999)

                          # print(genFGen2[i, :])

                          # with tf.device('/CPU:0'):
                          #    print((
                          #        (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2)).eval()))

                          # asdfasdfs

                          # secondSecSec_term_loss32.append(
                          #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2)).eval())

                          secondSecSec_term_loss32.append((
                              (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x999), 2), 1)) ** 2))))

                          # print(((tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2)).eval()))

                          # print(((tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2)).eval()))
                          # print(((tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2)).eval()))

                          secondSecSec_term_loss32.append((
                              (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x2), 2), 1)) ** 2))))

                          # secondSecSec_term_loss32.append(
                          #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x2), 2), 1)) ** 2)).eval())

                          secondSecSec_term_loss32.append((
                              (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x3), 2), 1)) ** 2))))

                          # secondSecSec_term_loss32.append(
                          #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x3), 2), 1)) ** 2)).eval())

                          # print(genFGen2[i, :].shape)
                          # print(x4.shape)

                          # print(x3.shape)
                          # print(x2.shape)

                          # asadfasfasf

                          secondSecSec_term_loss32.append((
                              (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x4), 2), 1)) ** 2))))

                          # secondSecSec_term_loss32.append(
                          #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x4), 2), 1)) ** 2)).eval())

                          secondSecSec_term_loss32.append((
                              (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x5), 2), 1)) ** 2))))

                          # secondSecSec_term_loss32.append(
                          #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x5), 2), 1)) ** 2)).eval())

                          secondSecSec_term_loss32.append((
                              (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x6), 2), 1)) ** 2))))

                          # secondSecSec_term_loss32.append(
                          #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x6), 2), 1)) ** 2)).eval())

                          secondSecSec_term_loss32.append((
                              (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x7), 2), 1)) ** 2))))

                          # secondSecSec_term_loss32.append(
                          #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x7), 2), 1)) ** 2)).eval())

                          secondSecSec_term_loss32.append((
                              (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x8), 2), 1)) ** 2))))

                          # secondSecSec_term_loss32.append(
                          #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x8), 2), 1)) ** 2)).eval())

                          secondSecSec_term_loss32.append((
                              (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x9), 2), 1)) ** 2))))

                          # secondSecSec_term_loss32.append(
                          #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x9), 2), 1)) ** 2)).eval())

                          secondSecSec_term_loss32.append((
                              (np.min(np.sqrt(np.sum(np.power((genFGen2[i, :] - x99), 2), 1)) ** 2))))

                          # secondSecSec_term_loss32.append(
                          #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x99), 2), 1)) ** 2)).eval())

                          # secondSecSec_term_loss32.append(
                          #  (tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x999), 2), 1)) ** 2)).eval())

                          # secondSecSec_term_loss32.append((tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x2), 2), 1)) ** 2)).eval())
                          # secondSecSec_term_loss32.append(tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x3), 2), 1)) ** 2).eval())

                          # secondSecSec_term_loss32.append(tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x4), 2), 1)) ** 2).eval())
                          # secondSecSec_term_loss32.append(tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((genFGen2[i, :] - x5), 2), 1)) ** 2).eval())

                          # secondSecSec_term_loss32[5] = tf.reduce_min(tf.sqrt((genFGen2[i, :] - x6).pow(2).sum(1)) ** 2)
                          # secondSecSec_term_loss32[6] = tf.reduce_min(tf.sqrt((genFGen2[i, :] - x7).pow(2).sum(1)) ** 2)

                          # secondSecSec_term_loss32[7] = tf.reduce_min(tf.sqrt((genFGen2[i, :] - x8).pow(2).sum(1)) ** 2)
                          # secondSecSec_term_loss32[8] = tf.reduce_min(tf.sqrt((genFGen2[i, :] - x9).pow(2).sum(1)) ** 2)

                          # secondSecSec_term_loss32[8] = torch.min(torch.sqrt((genFGen2[i, :] - x9).pow(2).sum(1)) ** 2)
                          # secondSecSec_term_loss32[9] = tf.reduce_min(tf.sqrt((genFGen2[i, :] - x99).pow(2).sum(1)) ** 2)

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
                          # second_term_loss32[i] = torch.argmin(secondSecSec_term_loss32)

                          # second_term_loss32[i] = torch.argmin(secondSecSec_term_loss32)

                          # secondSecSec_term_loss32 = tf.stack(secondSecSec_term_loss32)

                          # second_term_loss32[i] = torch.argmin(secondSecSec_term_loss32)
                          # second_term_loss32[i] = tf.argmin(tf.stack(secondSecSec_term_loss32))

                          # second_term_loss32[i] = tf.argmin(tf.stack(secondSecSec_term_loss32))
                          second_term_loss32.append(tf.argmin(tf.stack(secondSecSec_term_loss32)).eval())

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

                      second_term_loss32 = tf.stack(second_term_loss32)

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

                  second_term_loss32 = second_term_loss32.eval()

                  plt.plot(second_term_loss32)
                  plt.savefig('saveMySaSaSaSaveStore_second_term_loss32.png', bbox_inches='tight')

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
                      if second_term_loss32[ii_loop21] == 0:
                          counterFor0 += 1
                      elif second_term_loss32[ii_loop21] == 1:
                          counterFor1 += 1
                      elif second_term_loss32[ii_loop21] == 2:
                          counterFor2 += 1
                      elif second_term_loss32[ii_loop21] == 3:
                          counterFor3 += 1
                      elif second_term_loss32[ii_loop21] == 4:
                          counterFor4 += 1
                      elif second_term_loss32[ii_loop21] == 5:
                          counterFor5 += 1
                      elif second_term_loss32[ii_loop21] == 6:
                          counterFor6 += 1
                      elif second_term_loss32[ii_loop21] == 7:
                          counterFor7 += 1
                      elif second_term_loss32[ii_loop21] == 8:
                          counterFor8 += 1
                      elif second_term_loss32[ii_loop21] == 9:
                          counterFor9 += 1

                  plt.figure()
                  plt.plot(
                      [counterFor0, counterFor1, counterFor2, counterFor3, counterFor4, counterFor5, counterFor6,
                       counterFor7,
                       counterFor8, counterFor9])
                  plt.savefig('saveMySaSaveSaSaveSaSaveSaSaSaveStore_second_term_loss32.png', bbox_inches='tight')
                  plt.savefig('NumberMyOfOccOccurences_vs_ClassesClusters.png', bbox_inches='tight')

                  plt.figure()
                  plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [counterFor0, counterFor1, counterFor2, counterFor3, counterFor4, counterFor5, counterFor6,
                            counterFor7,
                            counterFor8, counterFor9], '--bo', linewidth=2, markersize=12)
                  plt.ylabel('Number of modes')
                  plt.xlabel('Modes')
                  plt.savefig('NuMyNumberOfOccurences_vs_ClassesClusters.png', bbox_inches='tight')

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
                  # plt.ylabel('Number of modes')
                  plt.ylabel('Probability')
                  plt.xlabel('Modes')
                  plt.savefig('NumMyNumNumNumberOfOccurences_vs_ClassesClusters.png', bbox_inches='tight')
                  plt.savefig('NumNikMyNumNumNumberOfOccurences_vs_ClassesClusters.png', bbox_inches='tight')

                  asdfkfs

                  sadf

                  dasfasdfd

                  # sadfsdfs

                  # sfsadfs
                  # asdfasd

                  log_like_batch_idxs = len(data) // config.batch_size
                  lli_list = []
                  # inter_list = []
                  for idx in xrange(0, log_like_batch_idxs):
                      # batch_images = data[idx * config.batch_size:(idx + 1) * config.batch_size]

                      # batch_images = data[idx * config.batch_size:(idx + 1) * config.batch_size]
                      batch_images = data[idx * config.batch_size:(idx + 1) * config.batch_size]

                      # batch_images = [batch_images, batch_images]
                      # print(batch_images)

                      batch_images = np.repeat(batch_images, 100, axis=0)
                      # print(batch_images)

                      # print(batch_images.shape)
                      # asdfsafs

                      # batch_images = data[idx * config.batch_size:(idx + 1) * config.batch_size]
                      batch_images = np.cast[np.float32](batch_images)

                      # batch_images = tf.reshape(tf.tile(batch_images, tf.constant([100], tf.int32)), [100, 784])
                      # batch_images = tf.tile(batch_images, tf.constant([100, 1], tf.int32))

                      # batch_images = tf.tile(batch_images, tf.constant([100, 1], tf.int32))
                      # batch_images = np.cast[np.float32](batch_images)

                      # print(batch_images.shape)
                      # adfsadfs

                      if self.model_type == "nice":
                          batch_images = batch_images[:, self.permutation]

                      # batch_images = tf.tile(batch_images, tf.constant([100, 1], tf.int32))
                      # batch_images = np.cast[np.float32](batch_images)

                      lli = self.sess.run([self.log_likelihood],
                                          feed_dict={self.log_like_batch: batch_images})

                      lli_list.append(lli)

                  # lli_list = np.exp(lli_list)
                  print(np.mean(lli_list))

                  # print(lli_list)
                  print(len(lli_list))

                  # -329.59705
                  # [[-3393.8728], [-3069.1543], [-3634.7078], [-3107.4153],

                  # -329.59866
                  # [[-3393.8723], [-3069.1543], [-3634.7078], [-3107.4153], [-3223.6433], [-3631.3137], [-3197.009],

                  # llillllii_list = lli_list.max()
                  # lllliilllilllii_list = np.exp(lli_list - llillllii_list)
                  # firstOnly_lossGen2 =  lllliilllilllii_list / lllliilllilllii_list.sum()

                  # firstOnly_lossGen2 = lli_list
                  # firstOnly_lossGen2 = np.exp(lli_list)

                  llillllii_list = np.max(lli_list)
                  lllliilllilllii_list = np.exp(lli_list - llillllii_list)
                  firstOnly_lossGen2 = lllliilllilllii_list / lllliilllilllii_list.sum()

                  # 0.5000553955240417
                  # 0.9026107910480833
                  # 0.49994460447595834
                  # 0.0974

                  # firstOnly_lossGen2 = lli_list
                  # 0.19189498646620234
                  # 0.8326816764694991
                  # 0.8081050135337977
                  # 0.2199494240272631

                  # firstOnly_lossGen2 = np.exp(lli_list)
                  loLosses_NIKlosses2 = firstOnly_lossGen2

                  # print(test_labels)
                  # print(test_labels.shape)

                  # print(test_labels)
                  # print(test_labels.shape)

                  # import numpy as np
                  tesTest_labels = np.array(test_labels)

                  # indices_one = tesTest_labels != 8
                  # indices_zero = tesTest_labels == 8

                  indices_one = tesTest_labels != 3
                  indices_zero = tesTest_labels == 3

                  tesTest_labels[indices_one] = 0  # replacing 1s with 0s
                  tesTest_labels[indices_zero] = 1  # replacing 0s with 1s

                  tesTest_labels = tesTest_labels.tolist()

                  # print(tesTest_labels)
                  # print(tesTest_labels.shape)

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

                  loLosses_NIKlosses3 = tesTest_labels

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

                  # print(loLoss2)
                  # print(loLossNoChange)

                  # print(len(loLoss2))
                  # print(len(loLossNoChange))

                  # adfasdfasdfsdfs

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

                  # 0.5000553955240417
                  # 0.9026107910480833
                  # 0.49994460447595834
                  # 0.0974

                  # 0.5000553955240417
                  # 0.9026107910480833
                  # 0.49994460447595834
                  # 0.0974

                  asdfas

                  sdfasfsfs

                  # curr_model_path = self.save(config.checkpoint_dir, counter)
                  curr_model_path = self.save(config.checkpoint_dir, 703)

                  val_nlli = self.evaluate_neg_loglikelihood(val_data, config)

                  # train_nlli = self.evaluate_neg_loglikelihood(train_data, config)
                  curr_inception_score = self.calculate_inception_and_mode_score()

                  print("[LogLi (%d,%d)]: val neg logli: %.8f, ince: %.8f, train lli: %.8f" % (epoch, idx, val_nlli, \
                                                                                               curr_inception_score[0],
                                                                                               np.mean(
                                                                                                   self.batch_train_nlli_list[
                                                                                                   -700:])))

                  self.writer.add_summary(tf.Summary( \
                      value=[tf.Summary.Value(tag="Val Neg Log-likelihood", simple_value=val_nlli)]), counter)
                  # self.writer.add_summary(tf.Summary(\
                  #         value=[tf.Summary.Value(tag="Train Neg Log-likelihood", simple_value=train_nlli)]), counter)
                  if val_nlli < self.best_val_nlli:
                      self.best_val_nlli = val_nlli
                      self.best_model_counter = counter
                      self.best_model_path = curr_model_path
                      # self.best_model_train_nlli = train_nlli
                  # self.train_nlli_list.append(train_nlli)
                  self.val_nlli_list.append(val_nlli)
                  self.counter_list.append(counter)

                  samples, d_loss, g_loss = self.sess.run(
                      [self.sampler, self.d_loss, self.g_loss],
                      feed_dict={
                          self.z: sample_z,
                          self.inputs: sample_inputs,
                      }
                  )
                  self.samples_list.append(samples)
                  self.samples_list[-1].shape[1]
                  manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                  manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                  self.inception_list.append(curr_inception_score)
                  save_images(samples, [manifold_h, manifold_w],
                              './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                  print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                  np.save("./" + config.sample_dir + '/graph_data',
                          [self.samples_list, self.val_nlli_list, self.best_val_nlli, self.best_model_counter, \
                           self.best_model_path, self.counter_list, [manifold_h, manifold_w], \
                           self.batch_train_nlli_list, self.inception_list, self.loss_list])

      np.save("./" + config.sample_dir + '/graph_data',
              [self.samples_list, self.val_nlli_list, self.best_val_nlli, self.best_model_counter, \
               self.best_model_path, self.counter_list, [manifold_h, manifold_w], \
               self.batch_train_nlli_list, self.inception_list, self.loss_list])
      self.test_model(test_data, config)

  def test_model(self, test_data, config):
    print("[*] Restoring best model counter: %d, val neg lli: %.8f" 
      % (self.best_model_counter, self.best_val_nlli))
    self.saver.restore(self.sess, self.best_model_path)
    print("[*] Best model restore from: " + self.best_model_path)
    print("[*] Evaluating on the test set")
    test_nlli = self.evaluate_neg_loglikelihood(test_data, config)
    print("[*] Test negative log likelihood: %.8f" % (test_nlli))

  def calculate_inception_and_mode_score(self):
    #to get mode scores add code to load your favourite mnist classifier in inception_score.py
    if self.dataset_name == "mnist": 
      return [0.0, 0.0, 0.0, 0.0]
    sess = self.sess
    all_samples = []
    for i in range(18):
        if self.prior == "uniform":
          batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
              .astype(np.float32)
        elif self.prior == "logistic":
          batch_z = np.random.logistic(loc=0.,scale=1.0,size=[self.batch_size, self.z_dim]) \
              .astype(np.float32)
        elif self.prior == "gaussian":
          batch_z = np.random.normal(0.0, 1.0, size=(self.batch_size , self.z_dim))
        else:
          print("ERROR: Unrecognized prior...exiting")
          exit(-1)
        samples_curr = self.sess.run(
            [self.sampler],
            feed_dict={
                self.z: batch_z,}
          )
        all_samples.append(samples_curr[0])
    all_samples = np.concatenate(all_samples, axis=0)
    # return all_samples
    all_samples = (all_samples*255.).astype('int32')
    
    return inception_score.get_inception_and_mode_score(list(all_samples), sess=sess)
  
  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      tf.set_random_seed(0)
      np.random.seed(0)
      if reuse:
        scope.reuse_variables()

      if self.dataset_name != "mnist":
        if self.f_div == "wgan":
          hn1 = image
         
          h0 = Layernorm('d_ln_1', [1,2,3], lrelu(conv2d(hn1, self.df_dim , name='d_h0_conv')))
          h1 = Layernorm('d_ln_2', [1,2,3], lrelu(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
          h2 = Layernorm('d_ln_3', [1,2,3], lrelu(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
          h3 = Layernorm('d_ln_4', [1,2,3], lrelu(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
          h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
      
          return tf.nn.sigmoid(h4), h4
        else:
          h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
          h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
          h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
          h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
          h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

          return tf.nn.sigmoid(h4), h4
      else:
        if self.f_div == "wgan":
          x = image

          h0 = lrelu(conv2d(x, self.c_dim, name='d_h0_conv'))

          h1 = lrelu(conv2d(h0, self.df_dim , name='d_h1_conv'))
          h1 = tf.reshape(h1, [self.batch_size, -1])      

          h2 = lrelu(linear(h1, self.dfc_dim, 'd_h2_lin'))

          h3 = linear(h2, 1, 'd_h3_lin')

          return tf.nn.sigmoid(h3), h3
        else:
          x = image
          
          h0 = lrelu(conv2d(x, self.c_dim, name='d_h0_conv'))
          
          h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim , name='d_h1_conv')))
          h1 = tf.reshape(h1, [self.batch_size, -1])      
          
          h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
          
          h3 = linear(h2, 1, 'd_h3_lin')
            
          return tf.nn.sigmoid(h3), h3


  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.input_height, self.input_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    return self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0

