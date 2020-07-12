import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#import os
import scipy.misc
import numpy as np
np.random.seed(0)

from model import DCGAN
#from utils import pp

import json
from moModel import DCDCDCGAN

import tensorflow as tf
tf.set_random_seed(0)

from utils2 import pp, visualize, to_json, show_all_variables, expand_path, timestamp
from utils import pp

# main.py --dataset mnist --input_height=28 --c_dim=1
# --checkpoint_dir checkpoint_mnist/flow --sample_dir samples_mnist/flow
# --model_type nice --log_dir logs_mnist/flow --prior logistic
# --beta1 0.5 --learning_rate 1e-4 --alpha 1e-7 --reg 10.0 --epoch 500
# --batch_size 100 --like_reg 1.0 --n_critic 5 --no_of_layers 5

flags = tf.app.flags
#flags.DEFINE_integer("epoch", 8, "Epoch to train [25]")
#flags.DEFINE_integer("epoch", 2, "Epoch to train [25]")
#flags.DEFINE_integer("epoch", 31, "Epoch to train [25]")
flags.DEFINE_integer("epoch", 9, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 28, "The size of image to use  [32]")
flags.DEFINE_integer("input_width", None, "The size of image to use If None, same value as input_height [None]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [mnist, multi-mnist, cifar-10]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_mnist/flow", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "logs_mnist/flow", "Directory name to save the logs [logs]")
flags.DEFINE_string("sample_dir", "samples_mnist/flow", "Directory name to save the image samples [samples]")
flags.DEFINE_string("f_div", "wgan", "f-divergence used for specifying the objective")
flags.DEFINE_string("prior", "logistic", "prior for generator")
flags.DEFINE_float("alpha", 1e-7, "alpha value (if applicable)")
flags.DEFINE_float("lr_decay", 1.0, "learning rate decay rate")
flags.DEFINE_float("min_lr", 0.0, "minimum lr allowed")
flags.DEFINE_float("reg", 10.0, "regularization parameter (only for wgan)")
flags.DEFINE_string("model_type", "nice", "model_type")
flags.DEFINE_string("init_type", "normal", "initialization for weights")
flags.DEFINE_integer("n_critic", 5, "no of discriminator iterations")
flags.DEFINE_integer("batch_norm_adaptive", 1, "type of batch norm used (only for real-nvp)")
flags.DEFINE_integer("no_of_layers", 5,"No of units between input and output in the m function for a coupling layer")
flags.DEFINE_integer("hidden_layers", 1000, "Size of hidden layers if applicable")
flags.DEFINE_integer("gpu_nr", 0, "gpu no used")
flags.DEFINE_float("like_reg", 1.0, "regularizing factor for likelihood")
flags.DEFINE_integer("df_dim", 64, "Dim depth of disc")


FLAGS = flags.FLAGS

def main(_):
  np.random.seed(0)
  tf.set_random_seed(0)
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
  run_config = tf.ConfigProto()

  #run_config.gpu_options.allow_growth=True
  #run_config.gpu_options.allow_growth=True

  #run_config.gpu_options.allow_growth=True
  #run_config.allow_soft_placement=True

  #run_config.allow_soft_placement=True
  #run_config.allow_soft_placement=True

  sess = None

  # # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
  # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  #
  # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



  # main.py - -dataset
  # mnist - -input_height = 28 - -output_height = 28 - -train

  # - -dataset
  # mnist - -input_height = 28 - -output_height = 28 - -train

  flags2 = tf.app.flags
  #flags2.DEFINE_integer("epoch2", 25, "Epoch to train [25]")
  flags2.DEFINE_integer("epoch2", 81, "Epoch to train [25]")
  flags2.DEFINE_float("learning_rate2", 0.0002, "Learning rate of for adam [0.0002]")
  flags2.DEFINE_float("beta12", 0.5, "Momentum term of adam [0.5]")
  flags2.DEFINE_float("train_size2", np.inf, "The size of train images [np.inf]")
  #flags2.DEFINE_integer("batch_size2", 1024, "The size of batch images [64]")
  #flags2.DEFINE_integer("batch_size2", 512, "The size of batch images [64]")
  flags2.DEFINE_integer("batch_size2", 256, "The size of batch images [64]")
  #flags2.DEFINE_integer("batch_size2", 128, "The size of batch images [64]")
  #flags2.DEFINE_integer("batch_size32", 50000, "The size of batch images [64]")
  #flags2.DEFINE_integer("batch_size32", 25000, "The size of batch images [64]")
  #flags2.DEFINE_integer("batch_size32", 12500, "The size of batch images [64]")
  #flags2.DEFINE_integer("batch_size32", 10000, "The size of batch images [64]")
  flags2.DEFINE_integer("batch_size32", 6250, "The size of batch images [64]")
  flags2.DEFINE_integer("input_height2", 28, "The size of image to use (will be center cropped). [108]")
  flags2.DEFINE_integer("input_width2", None,
                       "The size of image to use (will be center cropped). If None, same value as input_height [None]")
  flags2.DEFINE_integer("output_height2", 28, "The size of the output images to produce [64]")
  flags2.DEFINE_integer("output_width2", None,
                       "The size of the output images to produce. If None, same value as output_height [None]")
  flags2.DEFINE_string("dataset2", "mnist", "The name of dataset [celebA, mnist, lsun]")
  flags2.DEFINE_string("input_fname_pattern2", "*.jpg", "Glob pattern of filename of input images [*]")
  flags2.DEFINE_string("data_dir2", "./data", "path to datasets [e.g. $HOME/data]")
  flags2.DEFINE_string("out_dir2", "./out", "Root directory for outputs [e.g. $HOME/out]")
  flags2.DEFINE_string("out_name2", "",
                      "Folder (under out_root_dir) for all outputs. Generated automatically if left blank []")
  flags2.DEFINE_string("checkpoint_dir2", "checkpoint",
                      "Folder (under out_root_dir/out_name) to save checkpoints [checkpoint]")
  flags2.DEFINE_string("sample_dir2", "samples", "Folder (under out_root_dir/out_name) to save samples [samples]")
  flags2.DEFINE_boolean("train2", True, "True for training, False for testing [False]")
  flags2.DEFINE_boolean("crop2", False, "True for training, False for testing [False]")
  flags2.DEFINE_boolean("visualize2", False, "True for visualizing, False for nothing [False]")
  flags2.DEFINE_boolean("export2", False, "True for exporting with new batch size")
  flags2.DEFINE_boolean("freeze2", False, "True for exporting with new batch size")
  flags2.DEFINE_integer("max_to_keep2", 1, "maximum number of checkpoints to keep")
  flags2.DEFINE_integer("sample_freq2", 200, "sample every this many iterations")
  flags2.DEFINE_integer("ckpt_freq2", 200, "save checkpoint every this many iterations")
  flags2.DEFINE_integer("z_dim2", 100, "dimensions of z")
  flags2.DEFINE_string("z_dist2", "uniform_signed", "'normal01' or 'uniform_unsigned' or uniform_signed")
  flags2.DEFINE_boolean("G_img_sum2", False, "Save generator image summaries in log")
  # flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
  FLAGS2 = flags2.FLAGS

  pp.pprint(flags2.FLAGS.__flags)

  # expand user name and environment variables
  FLAGS2.data_dir2 = expand_path(FLAGS2.data_dir2)
  FLAGS2.out_dir2 = expand_path(FLAGS2.out_dir2)
  FLAGS2.out_name2 = expand_path(FLAGS2.out_name2)
  FLAGS2.checkpoint_dir2 = expand_path(FLAGS2.checkpoint_dir2)
  FLAGS2.sample_dir2 = expand_path(FLAGS2.sample_dir2)

  if FLAGS2.output_height2 is None: FLAGS2.output_height2 = FLAGS2.input_height2
  if FLAGS2.input_width2 is None: FLAGS2.input_width2 = FLAGS2.input_height2
  if FLAGS2.output_width2 is None: FLAGS2.output_width2 = FLAGS2.output_height2

  # output folders
  if FLAGS2.out_name2 == "":
      FLAGS2.out_name2 = '{} - {} - {}'.format(timestamp(), FLAGS2.data_dir2.split('/')[-1],
                                             FLAGS2.dataset2)  # penultimate folder of path
      if FLAGS2.train2:
          FLAGS2.out_name2 += ' - x{}.z{}.{}.y{}.b{}'.format(FLAGS2.input_width2, FLAGS2.z_dim2, FLAGS2.z_dist2,
                                                           FLAGS2.output_width2, FLAGS2.batch_size2)

  FLAGS2.out_dir2 = os.path.join(FLAGS2.out_dir2, FLAGS2.out_name2)
  FLAGS2.checkpoint_dir2 = os.path.join(FLAGS2.out_dir2, FLAGS2.checkpoint_dir2)
  FLAGS2.sample_dir2 = os.path.join(FLAGS2.out_dir2, FLAGS2.sample_dir2)

  if not os.path.exists(FLAGS2.checkpoint_dir2): os.makedirs(FLAGS2.checkpoint_dir2)
  if not os.path.exists(FLAGS2.sample_dir2): os.makedirs(FLAGS2.sample_dir2)

  with open(os.path.join(FLAGS2.out_dir2, 'FLAGS.json'), 'w') as f:
      flags_dict = {k: FLAGS2[k].value for k in FLAGS2}
      json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

  #run_config = tf.ConfigProto()
  #run_config.gpu_options.allow_growth = True

  #run_config = tf.ConfigProto()
  #run_config.gpu_options.allow_growth = True

  '''
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  run_config.allow_soft_placement=True

  sess = None

  # # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
  # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  #
  # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  '''

  """
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  run_config.allow_soft_placement=True

  sess = None

  # # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
  # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  #
  # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  """



  with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        c_dim=FLAGS.c_dim,
        z_dim=FLAGS.c_dim * FLAGS.input_height * FLAGS.input_width,
        dataset_name=FLAGS.dataset,
        checkpoint_dir=FLAGS.checkpoint_dir,
        f_div=FLAGS.f_div,
        prior=FLAGS.prior,
        lr_decay=FLAGS.lr_decay,
        min_lr=FLAGS.min_lr,
        model_type=FLAGS.model_type,
        log_dir=FLAGS.log_dir,
        alpha=FLAGS.alpha,
        batch_norm_adaptive=FLAGS.batch_norm_adaptive,
        init_type=FLAGS.init_type,
        reg=FLAGS.reg,
        n_critic=FLAGS.n_critic,
        hidden_layers=FLAGS.hidden_layers,
        no_of_layers=FLAGS.no_of_layers,
        like_reg=FLAGS.like_reg,
        df_dim=FLAGS.df_dim)

    '''
    dcdcdcgan = DCDCDCGAN(
        sess,
        dcDcgan = dcgan,
        input_width=FLAGS2.input_width2,
        input_height=FLAGS2.input_height2,
        output_width=FLAGS2.output_width2,
        output_height=FLAGS2.output_height2,
        batch_size=FLAGS2.batch_size2,
        sample_num=FLAGS2.batch_size32,
        y_dim=10,
        z_dim=FLAGS2.z_dim2,
        dataset_name=FLAGS2.dataset2,
        input_fname_pattern=FLAGS2.input_fname_pattern2,
        crop=FLAGS2.crop2,
        checkpoint_dir=FLAGS2.checkpoint_dir2,
        sample_dir=FLAGS2.sample_dir2,
        data_dir=FLAGS2.data_dir2,
        out_dir=FLAGS2.out_dir2,
        max_to_keep=FLAGS2.max_to_keep2)
    '''

    #dcdcdcgan = DCDCDCGAN(
    #    sess,
    #    dcDcgan = dcgan,
    #    input_width=FLAGS2.input_width2,
    #    input_height=FLAGS2.input_height2,
    #    output_width=FLAGS2.output_width2,
    #    output_height=FLAGS2.output_height2,
    #    batch_size=FLAGS2.batch_size2,
    #    sample_num=FLAGS2.batch_size32,
    #    y_dim=10,
    #    z_dim=FLAGS2.z_dim2,
    #    dataset_name=FLAGS2.dataset2,
    #    input_fname_pattern=FLAGS2.input_fname_pattern2,
    #    crop=FLAGS2.crop2,
    #    checkpoint_dir=FLAGS2.checkpoint_dir2,
    #    sample_dir=FLAGS2.sample_dir2,
    #    data_dir=FLAGS2.data_dir2,
    #    out_dir=FLAGS2.out_dir2,
    #    max_to_keep=FLAGS2.max_to_keep2)

    '''
    dcdcdcgan = DCDCDCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        c_dim=FLAGS.c_dim,
        z_dim=FLAGS.c_dim * FLAGS.input_height * FLAGS.input_width,
        dataset_name=FLAGS.dataset,
        checkpoint_dir=FLAGS.checkpoint_dir,
        f_div=FLAGS.f_div,
        prior=FLAGS.prior,
        lr_decay=FLAGS.lr_decay,
        min_lr=FLAGS.min_lr,
        model_type=FLAGS.model_type,
        log_dir=FLAGS.log_dir,
        alpha=FLAGS.alpha,
        batch_norm_adaptive=FLAGS.batch_norm_adaptive,
        init_type=FLAGS.init_type,
        reg=FLAGS.reg,
        n_critic=FLAGS.n_critic,
        hidden_layers=FLAGS.hidden_layers,
        no_of_layers=FLAGS.no_of_layers,
        like_reg=FLAGS.like_reg,
        df_dim=FLAGS.df_dim)
    '''

    #dcgan.train(FLAGS)

    dcgan.train(FLAGS)
    #print('asdfasf OK asdfasdf')

    #print('asdfasf OK asdfasdf')

    #print('asdfasf OK asdfasdf')
    #print('asdfasf OK asdfasdf')

    #print('')
    #dcdcdcgan.train(FLAGS2)

    #dcdcdcgan.train(FLAGS2)
    #print('asdfasdfdfasf OK OK OK asdfsdfsaasdf')

    #print('asdfasdfdfasf OK OK OK asdfsdfsaasdf')

    #print('asdfasdfdfasf OK OK OK asdfsdfsaasdf')
    #print('asdfasdfdfasf OK OK OK asdfsdfsaasdf')

    #print('asdfasdasdfasfdfasf This is OK. asdadsfasfsdfsaasdf')
    #print('asdfasdasdfasfdfasf This is OK. asdadsfasfsdfsaasdf')

    #print('asdfasdasdfasfdfasf This is OK. asdadsfasfsdfsaasdf')
    #print('asdfasdasdfasfdfasf This is OK. asdadsfasfsdfsaasdf')

    print('')

    #test_nlli = self.evaluate_neg_loglikelihood(test_data, config)

    #test_nlli = self.evaluate_neg_loglikelihood(test_data, config)
    #test_nlli = self.evaluate_neg_loglikelihood(test_data, config)

    import dataset_loaders.mnist_loader as mnist_data

    #data_X, val_data, test_data, train_dist = mnist_data.load_mnist()
    train_data, val_data, test_data, train_dist = mnist_data.load_mnist()

    #data_X, val_data, test_data, train_dist = mnist_data.load_mnist()
    test_data = np.reshape(test_data, (-1, dcgan.image_size))

    #data_X, val_data, test_data, train_dist = mnist_data.load_mnist()
    #test_nlli = dcgan.evaluate_neg_loglikelihood(test_data, FLAGS)

    #val_nlli = self.evaluate_neg_loglikelihood(val_data, config)

    #val_nlli=self.evaluate_neg_loglikelihood(val_data, config)
    #val_nlli=self.evaluate_neg_loglikelihood(val_data, config)

    #val_nlli = self.evaluate_neg_loglikelihood(val_data, config)
    #val_nlli = self.evaluate_neg_loglikelihood(val_data, config)

    #print('')
    #print('asdfaassdfassdasdfasfdfasf OK. This is OK. OK. asdadsfsasfssafasddfsaasdf')

    #print('asdfaassdfassdasdfasfdfasf OK. This is OK. OK. asdadsfsasfssafasddfsaasdf')

    #print('asdfaassdfassdasdfasfdfasf OK. This is OK. OK. asdadsfsasfssafasddfsaasdf')
    #print('asdfaassdfassdasdfasfdfasf OK. This is OK. OK. asdadsfsasfssafasddfsaasdf')

    #print(test_nlli)
    #print(test_nlli)

    #print('')
    #print(test_nlli)

    val_data = np.reshape(val_data, (-1, dcgan.image_size))
    #val_nlli = dcgan.evaluate_neg_loglikelihood(val_data, FLAGS)

    train_data = np.reshape(train_data, (-1, dcgan.image_size))
    #train_nlli = dcgan.evaluate_neg_loglikelihood(train_data, FLAGS)

    #print(val_nlli)
    #print(train_nlli)

    #print(val_nlli)
    #print(val_nlli)

    #print(train_nlli)

    #print(train_nlli)
    #print(train_nlli)

    print('')

    # test_data is (10000, 784)
    # the size is now (10000, 784)

    #nlli_test = sess.run([dcgan.log_likelihood],
    #                    feed_dict={dcgan.log_like_batch: test_data})

    #nlli_test = sess.run([dcgan.log_likelihood],
    #                     feed_dict={dcgan.log_like_batch: test_data[0,:].repeat(64, 1)})

    # torch.cat([input]*100)
    # use: torch.cat([input]*100)

    # np.tile(a,(3,1))
    # use: np.tile(a,(3,1))

    # test_data[0, :]
    # use: test_data[0, :]

    nlli_test = sess.run([dcgan.log_likelihood],
                         feed_dict={dcgan.log_like_batch: np.tile(test_data[0, :],(FLAGS.batch_size,1))})

    nlli_val = sess.run([dcgan.log_likelihood],
                         feed_dict={dcgan.log_like_batch: np.tile(val_data[0, :], (FLAGS.batch_size, 1))})

    nlli_train = sess.run([dcgan.log_likelihood],
                        feed_dict={dcgan.log_like_batch: np.tile(train_data[0, :], (FLAGS.batch_size, 1))})

    #train_nlli = dcgan.evaluate_neg_loglikelihood(train_data, FLAGS)

    #train_nlli = dcgan.evaluate_neg_loglikelihood(train_data, FLAGS)
    #train_nlli = dcgan.evaluate_neg_loglikelihood(train_data, FLAGS)

    nlli_test = np.squeeze(nlli_test)

    nlli_val = np.squeeze(nlli_val)
    nlli_train = np.squeeze(nlli_train)

    print(nlli_test)

    print(nlli_val)
    print(nlli_train)

    #test_nlli = dcgan.evaluate_neg_loglikelihood(test_data[0, :], FLAGS)
    #test_nlli = dcgan.evaluate_neg_loglikelihood(test_data[0, :].repeat(FLAGS.batch_size, 1), FLAGS)

    #test_nlli = dcgan.evaluate_neg_loglikelihood(test_data[0, :].repeat(FLAGS.batch_size, 1), FLAGS)
    test_nlli = dcgan.evaluate_neg_loglikelihood(np.tile(test_data[0, :], (FLAGS.batch_size, 1)), FLAGS)

    val_nlli = dcgan.evaluate_neg_loglikelihood(np.tile(val_data[0, :], (FLAGS.batch_size, 1)), FLAGS)
    train_nlli = dcgan.evaluate_neg_loglikelihood(np.tile(train_data[0, :], (FLAGS.batch_size, 1)), FLAGS)

    #val_nlli = dcgan.evaluate_neg_loglikelihood(val_data[0, :].repeat(FLAGS.batch_size, 1), FLAGS)
    #train_nlli = dcgan.evaluate_neg_loglikelihood(train_data[0, :].repeat(FLAGS.batch_size, 1), FLAGS)

    #print(test_nlli)
    #print(test_nlli)

    print('')
    print(test_nlli)

    print(val_nlli)
    print(train_nlli)

    #print(val_nlli)
    #print(val_nlli)

    #print(train_nlli)
    #print(train_nlli)

    #print(train_nlli)

    #print(train_nlli)
    #print(train_nlli)

    #adsfadsf

    #dsfasf
    #adsfas

    #print('')
    #print('')

    print('')

    #print('')
    #print('asdfaasdfsssasdfasdfassdasdfasfdfasf OK. This is OK. OK. asdfaasdfsssasdfasdfassdasdfasfdfasf')

    #print('asdfaassasdfasdfassdasdfasfdfasf OK. This is OK. OK. asdadsfsasfsasdfsfssafasddfsaasdf')
    #print('asdfaasdfsssasdfasdfassdasdfasfdfasf OK. This is OK. OK. asdfaasdfsssasdfasdfassdasdfasfdfasf')

    #print('')
    #print('asdfaasdfsssasdfasdfassdasdfasfdfasf OK. This is OK. OK. asdfaasdfsssasdfasdfassdasdfasfdfasf')

    #print('asdfaasdfsssasdfasdfassdasdfasfdfasf OK. This is OK. OK. asdfaasdfsssasdfasdfassdasdfasfdfasf')
    #print('asdfaasdfsssasdfasdfassdasdfasfdfasf OK. This is OK. OK. asdfaasdfsssasdfasdfassdasdfasfdfasf')

    #print('asdfaassasdfasdfassdasdfasfdfasf OK. This is OK. OK. asdadsfsasfsasdfsfssafasddfsaasdf')
    #print('asdfaassasdfasdfassdasdfasfdfasf OK. This is OK. OK. asdadsfsasfsasdfsfssafasddfsaasdf')

    # evaluate_neg_loglikelihood(self, data, config)
    # use: evaluate_neg_loglikelihood(self, data, config)

    #train_nlli = dcgan.evaluate_neg_loglikelihood(np.tile(train_data[0, :], (FLAGS.batch_size, 1)), FLAGS)
    #train_nlli = dcgan.evaluate_neg_loglikelihood(np.tile(train_data[0, :], (FLAGS.batch_size, 1)), FLAGS)

    #dcdcdcgan.train(FLAGS2)
    #dcdcdcgan.train(FLAGS2, dcgan, FLAGS)

    #dcdcdcgan.train(FLAGS2, dcgan, FLAGS)
    #dcdcdcgan.train(FLAGS2)

    dcdcdcgan = DCDCDCGAN(
        sess,
        dcDcgan=dcgan,
        input_width=FLAGS2.input_width2,
        input_height=FLAGS2.input_height2,
        output_width=FLAGS2.output_width2,
        output_height=FLAGS2.output_height2,
        batch_size=FLAGS2.batch_size2,
        sample_num=FLAGS2.batch_size32,
        y_dim=10,
        z_dim=FLAGS2.z_dim2,
        dataset_name=FLAGS2.dataset2,
        input_fname_pattern=FLAGS2.input_fname_pattern2,
        crop=FLAGS2.crop2,
        checkpoint_dir=FLAGS2.checkpoint_dir2,
        sample_dir=FLAGS2.sample_dir2,
        data_dir=FLAGS2.data_dir2,
        out_dir=FLAGS2.out_dir2,
        max_to_keep=FLAGS2.max_to_keep2)

    #dcdcdcgan.train(FLAGS2)

    #dcdcdcgan.train(FLAGS2)
    dcdcdcgan.train(FLAGS2)

    #dcdcdcgan.train(FLAGS2)

    #dcdcdcgan.train(FLAGS2)
    #dcdcdcgan.train(FLAGS2)

if __name__ == '__main__':
  tf.app.run()

