from __future__ import division
from __future__ import print_function
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

#import real_nvp.model as nvp
import real_nvp.nn as nvp_op

#import imageio
#imageio.imwrite('filename.jpg', array)

#from ops import *
from utils2 import *

#from ops import *
from ops2 import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def gen_random(mode, size):
    if mode == 'normal01': return np.random.normal(0, 1, size=size)
    if mode == 'uniform_signed': return np.random.uniform(-1, 1, size=size)
    if mode == 'uniform_unsigned': return np.random.uniform(0, 1, size=size)


class DCDCDCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
                 batch_size=64, sample_num=64, output_height=64, output_width=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 max_to_keep=1,
                 input_fname_pattern='*.jpg', checkpoint_dir='ckpts', sample_dir='samples', out_dir='./out',
                 data_dir='./data'):
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
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.max_to_keep = max_to_keep

        if self.dataset_name == 'mnist':
            self.data_X, self.data_y = self.load_mnist()
            self.c_dim = self.data_X[0].shape[-1]
        else:
            data_path = os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern)
            self.data = glob(data_path)
            if len(self.data) == 0:
                raise Exception("[!] No data found in '" + data_path + "'")
            np.random.shuffle(self.data)
            imreadImg = imread(self.data[0])
            if len(imreadImg.shape) >= 3:  # check if image is a non-grayscale image by checking channel number
                self.c_dim = imread(self.data[0]).shape[-1]
            else:
                self.c_dim = 1

            if len(self.data) < self.batch_size:
                raise Exception("[!] Entire dataset size is less than the configured batch_size")

        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        else:
            self.y = None

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        #self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        #self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        #self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        #self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        #self.firstTerm = tf.placeholder(tf.float32, [1], name='first_term')

        #self.firstTerm = tf.placeholder(tf.float32, [1], name='first_term')
        self.firstTerm = tf.placeholder(tf.float32, name='first_term')

        self.secondTerm = tf.placeholder(tf.float32, name='first_term')
        self.thirdTerm = tf.placeholder(tf.float32, name='first_term')

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G = self.generator(self.z, self.y)
        #self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
        self.sampler = self.sampler(self.z, self.y)
        #self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

        #self.d_sum = histogram_summary("d", self.D)
        #self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        #self.d_loss_real = tf.reduce_mean(
        #    sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        #self.d_loss_fake = tf.reduce_mean(
        #    sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        #self.g_loss = tf.reduce_mean(
        #    sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        #self.g_loss = tf.reduce_mean(
        #    sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        #self.g_loss = tf.reduce_mean(
        #    sigmoid_cross_entropy_with_logits(self.G, tf.ones_like(self.G)))

        #self.g_loss = tf.reduce_mean(
        #    sigmoid_cross_entropy_with_logits(self.G, tf.ones_like(self.G)))

        #self.g_loss = (self.firstTerm) + (tf.reduce_mean()) + (tf.reduce_mean())

        #self.g_loss = (self.firstTerm) + (tf.reduce_mean()) + (tf.reduce_mean())
        #self.g_loss = (self.firstTerm) + (tf.reduce_mean()) + (tf.reduce_mean())

        #self.g_loss = (self.firstTerm) + (tf.reduce_mean()) + (tf.reduce_mean())
        #self.g_loss = (self.firstTerm) + (tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.G, tf.ones_like(self.G))))

        #self.g_loss = (self.firstTerm) + (tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.G, tf.ones_like(self.G))))
        #self.g_loss = (self.firstTerm) + (self.secondTerm) + (self.thirdTerm)

        #self.g_loss = (self.firstTerm) + (self.secondTerm) + (self.thirdTerm)

        #self.g_loss = (self.firstTerm) + (self.secondTerm) + (self.thirdTerm)
        #self.g_loss = (self.firstTerm) + (self.secondTerm) + (self.thirdTerm) + (tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.G, tf.ones_like(self.G))))

        #self.g_loss = (self.firstTerm) + (self.secondTerm) + (self.thirdTerm)
        self.g_loss = (self.firstTerm) + (self.secondTerm) + (self.thirdTerm) + tf.reduce_mean(self.G)

        #adfasdbf

        #asdfa
        #asdfzs

        #self.g_loss = (self.firstTerm) + (tf.reduce_mean()) + (tf.reduce_mean())
        #self.g_loss = (self.firstTerm)

        #adfasdbf

        #asdfa
        #asdfzs

        #self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        #self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        #self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        #self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        #self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)

    def train(self, config, dcgan, FLAGS31):
        #d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
        #    .minimize(self.d_loss, var_list=self.d_vars)

        #sdfgdsgdsz

        #asdf
        #asdfs

        #train_nlli = dcgan.evaluate_neg_loglikelihood(np.tile(train_data[0, :], (FLAGS.batch_size, 1)), FLAGS)

        #print(train_nlli)
        #asdfdasfz

        g_optim = tf.train.AdamOptimizer(config.learning_rate2, beta1=config.beta12) \
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        if config.G_img_sum2:
            #self.g_sum = merge_summary([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
            self.g_sum = merge_summary([self.z_sum, self.G_sum, self.g_loss_sum])
        else:
            #self.g_sum = merge_summary([self.z_sum, self.d__sum, self.d_loss_fake_sum, self.g_loss_sum])
            self.g_sum = merge_summary([self.z_sum, self.g_loss_sum])
        #self.d_sum = merge_summary(
        #    [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum])
        self.writer = SummaryWriter(os.path.join(self.out_dir, "logs"), self.sess.graph)

        sample_z = gen_random(config.z_dist2, size=(self.sample_num, self.z_dim))

        if config.dataset2 == 'mnist':
            sample_inputs = self.data_X[0:self.sample_num]
            sample_labels = self.data_y[0:self.sample_num]
        else:
            sample_files = self.data[0:self.sample_num]
            sample = [
                get_image(sample_file,
                          input_height=self.input_height,
                          input_width=self.input_width,
                          resize_height=self.output_height,
                          resize_width=self.output_width,
                          crop=self.crop,
                          grayscale=self.grayscale) for sample_file in sample_files]
            if (self.grayscale):
                sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch2):
            if config.dataset2 == 'mnist':
                batch_idxs = min(len(self.data_X), config.train_size2) // config.batch_size2
            else:
                self.data = glob(os.path.join(
                    config.data_dir2, config.dataset2, self.input_fname_pattern))
                np.random.shuffle(self.data)
                batch_idxs = min(len(self.data), config.train_size2) // config.batch_size2

            for idx in xrange(0, int(batch_idxs)):
                if config.dataset2 == 'mnist':
                    batch_images = self.data_X[idx * config.batch_size2:(idx + 1) * config.batch_size2]
                    batch_labels = self.data_y[idx * config.batch_size2:(idx + 1) * config.batch_size2]
                else:
                    batch_files = self.data[idx * config.batch_size2:(idx + 1) * config.batch_size2]
                    batch = [
                        get_image(batch_file,
                                  input_height=self.input_height,
                                  input_width=self.input_width,
                                  resize_height=self.output_height,
                                  resize_width=self.output_width,
                                  crop=self.crop,
                                  grayscale=self.grayscale) for batch_file in batch_files]
                    if self.grayscale:
                        batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                    else:
                        batch_images = np.array(batch).astype(np.float32)

                batch_z = gen_random(config.z_dist2, size=[config.batch_size2, self.z_dim]) \
                    .astype(np.float32)

                if config.dataset2 == 'mnist':
                    # Update D network
                    #_, summary_str = self.sess.run([d_optim, self.d_sum],
                    #                               feed_dict={
                    #                                   self.inputs: batch_images,
                    #                                   self.z: batch_z,
                    #                                   self.y: batch_labels,
                    #                               })
                    #self.writer.add_summary(summary_str, counter)

                    #train_nlli = dcgan.evaluate_neg_loglikelihood(np.tile(train_data[0, :], (FLAGS31.batch_size, 1)),
                    #                                              FLAGS31)

                    #print(batch_images)
                    #print(batch_images.shape)

                    #batch_images = np.reshape(batch_images, (-1, dcgan.image_size))
                    #print(batch_images.shape)

                    #batch_images = np.reshape(batch_images, (-1, dcgan.image_size))

                    #batch_images = np.reshape(batch_images, (-1, dcgan.image_size))
                    #batch_images = np.reshape(batch_images, (-1, dcgan.image_size))

                    #train_nlli = dcgan.evaluate_neg_loglikelihood(np.tile(batch_images, (FLAGS31.batch_size, 1)),
                    #                                              FLAGS31)

                    #print(train_nlli)
                    #asdfdasfz

                    #print(train_nlli)
                    #trTrain_nlli = tf.exp(train_nlli)

                    #train_nlli = dcgan.evaluate_neg_loglikelihood2(np.tile(batch_images, (FLAGS31.batch_size, 1)),
                    #                                               FLAGS31)

                    #print(train_nlli)
                    #print(trTrain_nlli)

                    #train_nlli = dcgan.evaluate_neg_loglikelihood(np.tile(batch_images, (FLAGS31.batch_size, 1)),
                    #                                               FLAGS31)

                    #trTrain_nlli = tf.exp(train_nlli)
                    #print('')

                    #print(train_nlli)
                    #print(trTrain_nlli)

                    #asdfasfzs

                    #train_gen_para, train_jac = self.trainable_flow_model(inputs_tr_flow)
                    #self.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac, self.prior) / self.batch_size

                    #train_gen_para, train_jac = dcgan.trainable_flow_model(inputs_tr_flow)
                    #train_gen_para, train_jac = dcgan.trainable_flow_model(inputs_tr_flow)

                    #train_gen_para, train_jac = dcgan.trainable_flow_model(inputs_tr_flow)
                    #train_gen_para, train_jac = dcgan.flow_model(inputs_tr_flow)

                    #train_gen_para, train_jac = dcgan.flow_model(inputs_tr_flow)
                    #train_gen_para, train_jac = dcgan.flow_model(inputs_tr_flow)

                    #train_gen_para, train_jac = dcgan.flow_model(inputs_tr_flow)
                    #train_gen_para, train_jac = dcgan.flow_model(self.generator(batch_z, self.y))

                    #train_gen_para, train_jac = dcgan.flow_model(self.generator(batch_z, self.y))

                    #train_gen_para, train_jac = dcgan.flow_model(self.generator(batch_z, self.y))
                    #train_gen_para, train_jac = dcgan.trainable_flow_model(self.generator(batch_z, self.y))

                    #train_gen_para, train_jac = dcgan.flow_model(self.generator(batch_z, self.y))
                    #train_gen_para, train_jac = dcgan.flow_model(self.G)

                    #batch_images = np.reshape(batch_images, (-1, dcgan.image_size))
                    #batch_images = np.reshape(batch_images, (-1, dcgan.image_size))

                    #batch_images = np.reshape(batch_images, (-1, dcgan.image_size))
                    #train_gen_para, train_jac = dcgan.flow_model(tf.convert_to_tensor(batch_images, np.float32))

                    #train_gen_para, train_jac = dcgan.flow_model(tf.convert_to_tensor(batch_images, np.float32))

                    #train_gen_para, train_jac = dcgan.flow_model(tf.convert_to_tensor(batch_images, np.float32))
                    #train_gen_para, train_jac = dcgan.flow_model(tf.convert_to_tensor(batch_images, np.float32))

                    myFake_images = self.sess.run([self.G], feed_dict={self.inputs: batch_images, self.z: batch_z,
                                                                       self.y: batch_labels})

                    #myFake_images = np.reshape(myFake_images, (-1, dcgan.image_size))
                    #print(np.shape(myFake_images))

                    myFake_images = np.squeeze(myFake_images)
                    myFake_images = np.reshape(myFake_images, (-1, dcgan.image_size))

                    #print(np.shape(batch_images))
                    #print(np.shape(myFake_images))

                    #print(np.shape(myFake_images))
                    #print(myFake_images.size())

                    #print(myFake_images.shape)
                    #asdfasdfas

                    #train_gen_para, train_jac = dcgan.flow_model(tf.convert_to_tensor(batch_images, np.float32))
                    train_gen_para, train_jac = dcgan.flow_model(tf.convert_to_tensor(myFake_images, np.float32))

                    # _, summary_str = self.sess.run([d_optim, self.d_sum],
                    #                               feed_dict={
                    #                                   self.inputs: batch_images,
                    #                                   self.z: batch_z,
                    #                                   self.y: batch_labels,
                    #                               })

                    #dcgan.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac,
                    #                                                   self.prior) / self.batch_size

                    # use: batch_z
                    # now use: batch_z

                    #dcgan.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac,
                    #                                                   batch_z) / config.batch_size2

                    #train_nlli = nvp_op.log_likelihood(train_gen_para, train_jac,
                    #                                                   batch_z) / config.batch_size2

                    train_nlli = nvp_op.log_likelihood(train_gen_para, train_jac,
                                                       FLAGS31.prior) / FLAGS31.batch_size

                    #print(train_nlli)
                    #print(train_nlli.Print())

                    #print(train_nlli.Print())
                    #print(train_nlli.eval())

                    #print(train_nlli)
                    #print(train_nlli.eval())

                    print('')
                    traTrain_nlli = tf.exp(train_nlli)

                    #print(traTrain_nlli)
                    #print(traTrain_nlli.eval())

                    #print(train_nlli)
                    #print('')

                    #print(train_nlli)
                    #print(traTrain_nlli)

                    #print('')
                    #print(batch_images)

                    #print(batch_z)
                    #print(batch_labels)

                    print('')

                    # Once you have launched a sess, you can use your_tensor.eval(session=sess)
                    # or sess.run(your_tensor) to get you feed tensor into the format
                    # of numpy.array and then feed it to your placeholder.

                    #_, summary_str = self.sess.run([g_optim, self.g_sum],
                    #                               feed_dict={
                    #                                   self.firstTerm: traTrain_nlli,
                    #                                   self.inputs: batch_images,
                    #                                   self.z: batch_z,
                    #                                   self.y: batch_labels,
                    #                               })
                    #self.writer.add_summary(summary_str, counter)

                    # Once you have launched a sess, you can use your_tensor.eval(session=sess)
                    # or sess.run(your_tensor) to get you feed tensor into the format
                    # of numpy.array and then feed it to your placeholder.

                    # now use: sess.run(your_tensor)
                    # use: your_tensor.eval(session=sess)

                    #_, summary_str = self.sess.run([g_optim, self.g_sum],
                    #                               feed_dict={
                    #                                   self.firstTerm: self.sess.run(traTrain_nlli),
                    #                                   self.inputs: batch_images,
                    #                                   self.z: batch_z,
                    #                                   self.y: batch_labels,
                    #                               })
                    #self.writer.add_summary(summary_str, counter)

                    """
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={
                                                       self.firstTerm: self.sess.run(traTrain_nlli),
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                       self.y: batch_labels,
                                                   })
                    self.writer.add_summary(summary_str, counter)
                    """

                    #_, summary_str = self.sess.run([g_optim, self.g_sum],
                    #                               feed_dict={
                    #                                   self.firstTerm: self.sess.run(traTrain_nlli),
                    #                                   self.inputs: batch_images,
                    #                                   self.z: batch_z,
                    #                                   self.y: batch_labels,
                    #                               })
                    #self.writer.add_summary(summary_str, counter)

                    # we use: self.sess.run(your_tensor)
                    # use: your_tensor.eval(session=self.sess)

                    #_, summary_str = self.sess.run([g_optim, self.g_sum],
                    #                                   feed_dict={
                    #                                       self.firstTerm: traTrain_nlli,
                    #                                       self.inputs: batch_images,
                    #                                       self.z: batch_z,
                    #                                       self.y: batch_labels,
                    #                                   })
                    #    self.writer.add_summary(summary_str, counter)

                    #_, summary_str = self.sess.run([g_optim, self.g_sum],
                    #                               feed_dict={
                    #                                   self.inputs: batch_images,
                    #                                   self.z: batch_z,
                    #                                   self.y: batch_labels,
                    #                               })
                    #self.writer.add_summary(summary_str, counter)

                    # Update D network
                    #_, summary_str = self.sess.run([d_optim, self.d_sum],
                    #                              feed_dict={
                    #                                  self.inputs: batch_images,
                    #                                  self.z: batch_z,
                    #                                  self.y: batch_labels,
                    #                              })
                    #self.writer.add_summary(summary_str, counter)

                    # Update G network
                    #_, summary_str = self.sess.run([g_optim, self.g_sum],
                    #                               feed_dict={
                    #                                   self.z: batch_z,
                    #                                   self.y: batch_labels,
                    #                               })
                    #self.writer.add_summary(summary_str, counter)

                    

                    #self.xData = self.inputs
                    # xData is now batch_images

                    #print(np.shape(batch_images))
                    # batch_images is (1024, 28, 28, 1)

                    #self.genFgenFGen2 = self.flow_inv_model(self.z)
                    # genFgenFGen2 is now myFake_images

                    #print(np.shape(myFake_images))
                    #asdfasfszsdf

                    #print(np.shape(myFake_images))
                    # here, myFake_images is (1024, 784)

                    #self.xData = tf.reshape(self.xData, [-1, 28 * 28])
                    xData = tf.reshape(batch_images, [-1, 28 * 28])

                    #self.xData = tf.reshape(self.xData, [-1, 28 * 28])
                    #self.genFGen2 = tf.reshape(self.genFgenFGen2, [-1, 28 * 28])

                    #self.genFGen2 = tf.reshape(self.genFgenFGen2, [-1, 28 * 28])
                    genFGen2 = myFake_images

                    #self.genFGen3 = self.z
                    # genFGen3 is now batch_z

                    #print(np.shape(batch_z))
                    # here, batch_z is (1024, 100)

                    #self.genFGen3 = self.z
                    #self.genFGen3 = tf.reshape(self.genFGen3, [-1, 28 * 28])

                    #self.genFGen3 = tf.reshape(self.genFGen3, [-1, 28 * 28])
                    genFGen3 = batch_z

                    #self.second_term_loss2 = tf.reduce_min(
                    #    tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen2[0, :] - self.xData), 2), 1)) ** 2)
                    #for i in range(1, self.batch_size):
                    #    self.second_term_loss2 += tf.reduce_min(
                    #        tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen2[i, :] - self.xData), 2), 1)) ** 2)
                    #self.second_term_loss2 /= self.batch_size

                    second_term_loss2 = tf.reduce_min(
                        tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((genFGen2[0, :] - xData), 2), 1)) ** 2)
                    for i in range(1, config.batch_size2):
                    #for i in range(1, config.batch_size2+1):
                        second_term_loss2 += tf.reduce_min(
                            tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((genFGen2[i, :] - xData), 2), 1)) ** 2)
                    second_term_loss2 /= config.batch_size2

                    #self.third_term_loss32 = tf.reduce_mean(
                    #    (tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen3[0, :] - self.genFGen3), 2), 1))) / (
                    #            1e-17 + tf.sqrt(
                    #        1e-17 + tf.reduce_sum(tf.pow((self.genFGen2[0, :] - self.genFGen2), 2), 1))))
                    #for i in range(1, self.batch_size):
                    #    self.third_term_loss32 += tf.reduce_mean(
                    #        (tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen3[i, :] - self.genFGen3), 2), 1))) / (
                    #                1e-17 + tf.sqrt(
                    #            1e-17 + tf.reduce_sum(tf.pow((self.genFGen2[i, :] - self.genFGen2), 2), 1))))
                    #self.third_term_loss12 = self.third_term_loss32 / self.batch_size

                    third_term_loss32 = tf.reduce_mean(
                        (tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((genFGen3[0, :] - genFGen3), 2), 1))) / (
                                1e-17 + tf.sqrt(
                            1e-17 + tf.reduce_sum(tf.pow((genFGen2[0, :] - genFGen2), 2), 1))))
                    for i in range(1, config.batch_size2):
                    #for i in range(1, config.batch_size2+1):
                        third_term_loss32 += tf.reduce_mean(
                            (tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((genFGen3[i, :] - genFGen3), 2), 1))) / (
                                    1e-17 + tf.sqrt(
                                1e-17 + tf.reduce_sum(tf.pow((genFGen2[i, :] - genFGen2), 2), 1))))
                    third_term_loss12 = third_term_loss32 / config.batch_size2

                    # range(1, config.batch_size2)
                    # or range(1, config.batch_size2+1)?

                    # use range(1, config.batch_size2+1)?
                    # now use range(1, config.batch_size2+1)?

                    #print(traTrain_nlli)
                    #print(second_term_loss2)

                    #print(third_term_loss12)
                    #print('')

                    #print(traTrain_nlli.eval())
                    #print(second_term_loss2.eval())

                    #print(third_term_loss12.eval())
                    #print('')

                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={
                                                       self.firstTerm: self.sess.run(traTrain_nlli),
                                                       self.secondTerm: self.sess.run(second_term_loss2),
                                                       self.thirdTerm: self.sess.run(third_term_loss12),
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                       self.y: batch_labels,
                                                   })
                    self.writer.add_summary(summary_str, counter)

                    asdfsfs

                    asfkz
                    askdfs

                    #train_gen_para, train_jac = self.trainable_flow_model(self.genFgenFGen2)

                    #self.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac,
                    #                                                  self.prior) / self.batch_size

                    #self.train_log_likelihood = (tf.reduce_mean(tf.exp(-self.train_log_likelihood / 10000000))) + (
                    #    self.second_term_loss2) + (self.third_term_loss12)

                    '''
                    self.xData = self.inputs
                    self.genFgenFGen2 = self.flow_inv_model(self.z)
                    self.xData = tf.reshape(self.xData, [-1, 28 * 28])
                    self.genFGen2 = tf.reshape(self.genFgenFGen2, [-1, 28 * 28])

                    self.genFGen3 = self.z
                    self.genFGen3 = tf.reshape(self.genFGen3, [-1, 28 * 28])

                    self.second_term_loss2 = tf.reduce_min(
                        tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen2[0, :] - self.xData), 2), 1)) ** 2)
                    for i in range(1, self.batch_size):
                        self.second_term_loss2 += tf.reduce_min(
                            tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen2[i, :] - self.xData), 2), 1)) ** 2)
                    self.second_term_loss2 /= self.batch_size

                    self.third_term_loss32 = tf.reduce_mean(
                        (tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen3[0, :] - self.genFGen3), 2), 1))) / (
                                1e-17 + tf.sqrt(
                            1e-17 + tf.reduce_sum(tf.pow((self.genFGen2[0, :] - self.genFGen2), 2), 1))))
                    for i in range(1, self.batch_size):
                        self.third_term_loss32 += tf.reduce_mean(
                            (tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen3[i, :] - self.genFGen3), 2), 1))) / (
                                    1e-17 + tf.sqrt(
                                1e-17 + tf.reduce_sum(tf.pow((self.genFGen2[i, :] - self.genFGen2), 2), 1))))
                    self.third_term_loss12 = self.third_term_loss32 / self.batch_size

                    train_gen_para, train_jac = self.trainable_flow_model(self.genFgenFGen2)

                    self.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac,
                                                                      self.prior) / self.batch_size

                    self.train_log_likelihood = (tf.reduce_mean(tf.exp(-self.train_log_likelihood / 10000000))) + (
                        self.second_term_loss2) + (self.third_term_loss12)
                    '''

                    """
                    #train_gen_para, train_jac = self.trainable_flow_model(inputs_tr_flow)
                    #self.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac, self.prior) / self.batch_size
                    
                    #z_myZ_myMyZ = np.random.logistic(loc=0., scale=1., size=(self.sample_num , self.z_dim))
                    #train_gen_para, train_jac = self.trainable_flow_model(self.flow_inv_model(z_myZ_myMyZ))
                
                    #print(self.inputs)
                    #print(self.sample_inputs)
                
                    #print(self.batch_size)
                    #print(self.sample_num)
                
                    #adfasdfsfsdfs
                
                    self.xData = self.inputs
                
                    #xData = xData.view(-1, 28 * 28)
                    #genFGen2 = genFGen2.view(-1, 28 * 28)
                    #genFGen3 = genFGen3.squeeze()
                
                    #self.genFgenFGen2 = self.flow_inv_model(self.z)
                
                    #self.genFgenFGen2 = self.flow_inv_model(self.z)
                    self.genFgenFGen2 = self.flow_inv_model(self.z)
                
                    #self.genFgenFGen2 = self.flow_inv_model(self.z)
                    #self.genFgenFGen2 = self.sampler_function(self.z)
                
                    #self.genFgenFGen2 = self.flow_inv_model(self.z)
                    #genFGen2 = genFgenFGen2
                
                    self.xData = tf.reshape(self.xData, [-1, 28*28])
                    self.genFGen2 = tf.reshape(self.genFgenFGen2, [-1, 28 * 28])
                
                    #print(self.z)
                    #adfasdfs
                
                    self.genFGen3 = self.z
                    self.genFGen3 = tf.reshape(self.genFGen3, [-1, 28 * 28])
                
                    #device = args.device
                    #second_term_loss2 = tf.zeros(1, device=device, requires_grad=False)
                    #print(tf.pow((genFGen2[0, :] - xData), 2))
                    #print(tf.reduce_sum(tf.pow((genFGen2[0, :] - xData), 2), 1))
                    #asdfadsfdsaf
                    #self.second_term_loss2 = tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((self.genFGen2[0, :] - self.xData), 2), 1)) ** 2)
                    self.second_term_loss2 = tf.reduce_min(
                      tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen2[0, :] - self.xData), 2), 1)) ** 2)
                    #for i in range(self.batch_size):
                    for i in range(1, self.batch_size):
                      #second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)
                      #self.second_term_loss2 += tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((self.genFGen2[i, :] - self.xData), 2), 1)) ** 2)
                      self.second_term_loss2 += tf.reduce_min(
                        tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen2[i, :] - self.xData), 2), 1)) ** 2)
                    self.second_term_loss2 /= self.batch_size
                    #second_term_loss2 = second_term_loss2.squeeze()
                
                    #third_term_loss32 = torch.empty(self.batch_size, device=device, requires_grad=False)
                    self.third_term_loss32 = tf.reduce_mean((tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen3[0, :] - self.genFGen3), 2), 1))) / (
                              1e-17 + tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen2[0, :] - self.genFGen2), 2), 1))))
                    #for i in range(self.batch_size):
                    for i in range(1, self.batch_size):
                      self.third_term_loss32 += tf.reduce_mean((tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen3[i, :] - self.genFGen3), 2), 1))) / (
                              1e-17 + tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen2[i, :] - self.genFGen2), 2), 1))))
                      #third_term_loss32[i] = torch.mean(third_term_loss22)
                    #third_term_loss12 = torch.mean(third_term_loss32)
                    self.third_term_loss12 = self.third_term_loss32 / self.batch_size
                
                    #print(third_term_loss12)
                
                    #print(second_term_loss2)
                    #print(third_term_loss12)
                
                    #asdfasdf
                
                    #train_gen_para, train_jac = self.trainable_flow_model(self.flow_inv_model(self.z))
                    #train_gen_para, train_jac = self.trainable_flow_model(genFgenFGen2)
                
                    #train_gen_para, train_jac = self.trainable_flow_model(genFgenFGen2)
                
                    #train_gen_para, train_jac = self.trainable_flow_model(genFgenFGen2)
                    #train_gen_para, train_jac = self.flow_model(genFgenFGen2)
                
                
                
                    #asdfzsfd
                
                    #dfasz
                    #zdfasf
                
                
                
                    #train_gen_para, train_jac = self.flow_model(genFgenFGen2)
                    #train_gen_para, train_jac = self.flow_model(self.genFgenFGen2)
                
                    #train_gen_para, train_jac = self.flow_model(self.genFgenFGen2)
                
                    #train_gen_para, train_jac = self.flow_model(self.genFgenFGen2)
                    train_gen_para, train_jac = self.trainable_flow_model(self.genFgenFGen2)
                
                    #train_gen_para, train_jac = self.trainable_flow_model(self.flow_inv_model(self.z))
                    self.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac, self.prior) / self.batch_size
                
                    #print((tf.reduce_mean(tf.exp(-self.train_log_likelihood))))
                    #asdfasdfasdfs
                
                    #self.train_log_likelihood = (tf.reduce_mean(tf.exp(-self.train_log_likelihood))) + (secondTerm) + (thirdTerm)
                    #self.train_log_likelihood = (tf.reduce_mean(tf.exp(-self.train_log_likelihood))) + (self.second_term_loss2) + (self.third_term_loss12)
                
                    #self.train_log_likelihood = (tf.reduce_mean(tf.exp(-self.train_log_likelihood))) + (self.second_term_loss2) + (self.third_term_loss12)
                
                    #self.train_log_likelihood = (tf.reduce_mean(tf.exp(-self.train_log_likelihood))) + (self.second_term_loss2) + (self.third_term_loss12)
                    self.train_log_likelihood = (tf.reduce_mean(tf.exp(-self.train_log_likelihood / 10000000))) + (self.second_term_loss2) + (
                        self.third_term_loss12)
                
                    #self.evaluate_neg_loglikelihood22(out, config)
                
                    #self.evaluate_neg_loglikelihood22(out, config)
                    #self.evaluate_neg_loglikelihood22(out, config)
                    """



                    asdfasfzs

                    asdfasdfasz
                    asdfasfasdfz

                    # -0.34090483
                    # -0.90332794

                    # -0.90332794
                    # 0.38768163

                    #asdfas
                    #asdfasf



                    # Update G network
                    #_, summary_str = self.sess.run([g_optim, self.g_sum],
                    #                               feed_dict={
                    #                                   self.z: batch_z,
                    #                                   self.y: batch_labels,
                    #                               })
                    #self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    #_, summary_str = self.sess.run([g_optim, self.g_sum],
                    #                               feed_dict={self.z: batch_z, self.y: batch_labels})
                    #self.writer.add_summary(summary_str, counter)

                    #errD_fake = self.d_loss_fake.eval({
                    #    self.z: batch_z,
                    #    self.y: batch_labels
                    #})
                    #errD_real = self.d_loss_real.eval({
                    #    self.inputs: batch_images,
                    #    self.y: batch_labels
                    #})
                    errG = self.g_loss.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })
                else:
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={self.inputs: batch_images, self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                    errG = self.g_loss.eval({self.z: batch_z})

                print("[%8d Epoch:[%2d/%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f" \
                      % (counter, epoch, config.epoch2, idx, batch_idxs,
                         time.time() - start_time, errG))

                #print("[%8d Epoch:[%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                #      % (counter, epoch, config.epoch2, idx, batch_idxs,
                #         time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, config.sample_freq2) == 0:
                    if config.dataset2 == 'mnist':
                        samples, g_loss = self.sess.run(
                            [self.sampler, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.y: sample_labels,
                            }
                        )

                        #samples, d_loss, g_loss = self.sess.run(
                        #    [self.sampler, self.d_loss, self.g_loss],
                        #    feed_dict={
                        #        self.z: sample_z,
                        #        self.inputs: sample_inputs,
                        #        self.y: sample_labels,
                        #    }
                        #)
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    './{}/train_{:08d}.png'.format(config.sample_dir2, counter))
                        print("[Sample] g_loss: %.8f" % (g_loss))
                        #print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    else:
                        try:
                            samples, d_loss, g_loss = self.sess.run(
                                [self.sampler, self.d_loss, self.g_loss],
                                feed_dict={
                                    self.z: sample_z,
                                    self.inputs: sample_inputs,
                                },
                            )
                            save_images(samples, image_manifold_size(samples.shape[0]),
                                        './{}/train_{:08d}.png'.format(config.sample_dir2, counter))
                            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                        except:
                            print("one pic error!...")

                if np.mod(counter, config.ckpt_freq2) == 0:
                    self.save(config.checkpoint_dir2, counter)

                counter += 1

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

                return tf.nn.sigmoid(h4), h4
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])
                h1 = concat([h1, y], 1)

                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = concat([h2, y], 1)

                h3 = linear(h2, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(
                    z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

                self.h0 = tf.reshape(
                    self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(self.h0))

                self.h1, self.h1_w, self.h1_b = deconv2d(
                    h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(
                    h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(
                    h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(
                    h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

                return tf.nn.tanh(h4)
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(
                    self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
                                                    [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(
                    deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                h0 = tf.reshape(
                    linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
                    [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

                return tf.nn.tanh(h4)
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(
                    deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def load_mnist(self):
        data_dir = os.path.join(self.data_dir, self.dataset_name+'_data')

        #data_dir = os.path.join(data_dir, '_data')

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        return X / 255., y_vec

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step, filename='model', ckpt=True, frozen=False):
        # model_name = "DCGAN.model"
        # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        filename += '.b' + str(self.batch_size)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if ckpt:
            self.saver.save(self.sess,
                            os.path.join(checkpoint_dir, filename),
                            global_step=step)

        if frozen:
            tf.train.write_graph(
                tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ["generator_1/Tanh"]),
                checkpoint_dir,
                '{}-{:06d}_frz.pb'.format(filename, step),
                as_text=False)

    def load(self, checkpoint_dir):
        # import re
        print(" [*] Reading checkpoints...", checkpoint_dir)
        # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        # print("     ->", checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            # counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

