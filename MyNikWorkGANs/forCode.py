from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

# Import MNIST data

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

#from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Params

# Training Params
num_steps = 100000

#num_steps = 100000
learning_rate = 0.0002

#batch_size = 128

#batch_size = 128
batch_size = 1024

# Network Params

# Network Params
image_dim = 784 # 28*28 pixels

#noise_dim = 100 # Noise data points

#noise_dim = 100 # Noise data points
noise_dim = 200 # Noise data points

gen_hidden_dim = 256
disc_hidden_dim = 256

# A custom initialization (see Xavier Glorot init)

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Store layers weight & bias
weights = {
    'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
    'gen_out': tf.Variable(glorot_init([gen_hidden_dim, image_dim])),
}

#weights = {
#    'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
#    'gen_out': tf.Variable(glorot_init([gen_hidden_dim, image_dim])),
#    'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim])),
#    'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1])),
#}

biases = {
    'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
    'gen_out': tf.Variable(tf.zeros([image_dim])),
}

#biases = {
#    'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
#    'gen_out': tf.Variable(tf.zeros([image_dim])),
#    'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
#    'disc_out': tf.Variable(tf.zeros([1])),
#}

# Generator

# Generator
def generator(x):
    hidden_layer = tf.matmul(x, weights['gen_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

# Discriminator

# Discriminator
#def discriminator(x):
#    hidden_layer = tf.matmul(x, weights['disc_hidden1'])
#    hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
#    hidden_layer = tf.nn.relu(hidden_layer)
#    out_layer = tf.matmul(hidden_layer, weights['disc_out'])
#    out_layer = tf.add(out_layer, biases['disc_out'])
#    out_layer = tf.nn.sigmoid(out_layer)
#    return out_layer

# Build Networks

# Build Networks
# Network Inputs

# Network Inputs
gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')

#gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
#disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')

#disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')
disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')

# Build Generator Network

# Build Generator Network
gen_sample = generator(gen_input)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
#disc_real = discriminator(disc_input)
#disc_fake = discriminator(gen_sample)

# Build Loss

# Build Loss
#gen_loss = -tf.reduce_mean(tf.log(disc_fake))
#disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

#gen_loss = -tf.reduce_mean(tf.log(disc_fake))

#gen_loss = -tf.reduce_mean(tf.log(disc_fake))
#gen_loss = -tf.reduce_mean(tf.log(disc_fake))

#gen_loss = -tf.reduce_mean(tf.log(disc_fake))
#gen_loss = -tf.reduce_mean(tf.log(gen_sample))

#gen_loss = -tf.reduce_mean(tf.log(gen_sample))

#gen_loss = -tf.reduce_mean(tf.log(gen_sample))
gen_loss = -tf.reduce_mean(tf.log(gen_sample))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)

#optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
#optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Training Variables for each optimizer

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = [weights['gen_hidden1'], weights['gen_out'],
            biases['gen_hidden1'], biases['gen_out']]

# Discriminator Network Variables

# Discriminator Network Variables
#disc_vars = [weights['disc_hidden1'], weights['disc_out'],
#            biases['disc_hidden1'], biases['disc_out']]

# Create training operations

# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)

#train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
#train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

#init = tf.global_variables_initializer()

#init = tf.global_variables_initializer()
#init = tf.global_variables_initializer()



# Training
# Start training

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    #sess.run(init)

    #sess.run(init)
    #sess.run(init)

    for i in range(1, num_steps+1):
        # Prepare Data

        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Generate noise to feed to the generator
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

        # Train

        # Train
        feed_dict = {disc_input: batch_x, gen_input: z}
        _, gl = sess.run([train_gen, gen_loss],
                                feed_dict=feed_dict)

        #feed_dict = {disc_input: batch_x, gen_input: z}
        #_, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
        #                        feed_dict=feed_dict)

        #feed_dict = {disc_input: batch_x, gen_input: z}
        #_, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
        #                        feed_dict=feed_dict)

        if i % 100 == 0 or i == 1:
            #print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

            #print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
            print('Step %i: Generator Loss: %f' % (i, gl))

            #print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
            #print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

    # Generate images from noise, using the generator network.

    # Generate images from noise, using the generator network.
    f, a = plt.subplots(4, 10, figsize=(10, 4))

    for i in range(10):
        # Noise input

        # Noise input
        z = np.random.uniform(-1., 1., size=[4, noise_dim])

        g = sess.run([gen_sample], feed_dict={gen_input: z})
        g = np.reshape(g, newshape=(4, 28, 28, 1))

        # Reverse colours for better display

        # Reverse colours for better display
        g = -1 * (g - 1)

        for j in range(4):
            # Generate image from noise. Extend to 3 channels for matplot figure.

            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                             newshape=(28, 28, 3))
            a[j][i].imshow(img)

    f.show()
    plt.draw()

    #plt.draw()
    plt.waitforbuttonpress()

