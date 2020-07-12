import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.examples.tutorials.mnist import input_data
#import tensorflow_datasets as tfds

def load_mnist(send_labels=False):
    #mnist = input_data.read_data_sets("data/mnist_data/", one_hot=False)

    mnist = input_data.read_data_sets("data/mnist_data/", one_hot=False)
    #mnist = tfds.load(name="mnist", split=tfds.Split.TRAIN)

    np.random.seed(0)

    idx = mnist.train.labels == 0
    idx += mnist.train.labels == 1
    idx += mnist.train.labels == 2
    idx += mnist.train.labels == 3
    idx += mnist.train.labels == 4
    #idx += mnist.train.labels == 5
    idx += mnist.train.labels == 6
    idx += mnist.train.labels == 7
    idx += mnist.train.labels == 8
    idx += mnist.train.labels == 9

    '''
    idx = mnist.train.labels == 0
    idx += mnist.train.labels == 1
    idx += mnist.train.labels == 2
    idx += mnist.train.labels == 3
    idx += mnist.train.labels == 4
    idx += mnist.train.labels == 5
    idx += mnist.train.labels == 6
    idx += mnist.train.labels == 7
    idx += mnist.train.labels == 8
    idx += mnist.train.labels == 9
    #mnist.train.labels = mnist.train.labels[idx]
    #mnist.train.images = mnist.train.images[idx]
    '''

    # replace items[node.ind].v = node.v
    # with items[node.ind] = items[node.ind]._replace(v=node.v)

    #mnist.train.labels = mnist.train.labels[idx]

    #mnist.train.labels = mnist.train.labels[idx]
    #mnist.train = mnist.train._replace(labels=labels[idx])
    #mnist = mnist._replace(train.labels=train.labels[idx])

    #mnist.train.images = mnist.train.images[idx]

    #mnist.train.images = mnist.train.images[idx]
    #mnist.train = mnist.train._replace(images=images[idx])
    #mnist = mnist._replace(train.images=train.images[idx])

    #print(len(mnist.train.labels))
    #sadfsdfs

    #print(len(mnist.train.labels))
    #print(len(mnist.train.images))
    
    """
    idx = mnist.train.labels == 1
    idx += mnist.train.labels == 2
    idx += mnist.train.labels == 3
    idx += mnist.train.labels == 4
    idx += mnist.train.labels == 5
    idx += mnist.train.labels == 6
    idx += mnist.train.labels == 7
    idx += mnist.train.labels == 8
    idx += mnist.train.labels == 9
    mnist.train.labels = mnist.train.labels[idx]
    mnist.train.images = mnist.train.images[idx]
    """

    """
    idx = mnist.train.labels == 0
    idx += mnist.train.labels == 2
    idx += mnist.train.labels == 3
    idx += mnist.train.labels == 4
    idx += mnist.train.labels == 5
    idx += mnist.train.labels == 6
    idx += mnist.train.labels == 7
    idx += mnist.train.labels == 8
    idx += mnist.train.labels == 9
    mnist.train.labels = mnist.train.labels[idx]
    mnist.train.images = mnist.train.images[idx]
    """

    #train_data = mnist.train.images
    #train_labels = mnist.train.labels

    train_data = mnist.train.images[idx]
    train_labels = mnist.train.labels[idx]

    #print(train_data.shape)
    #print(train_labels.shape)

    #sadfsad

    #train_data = mnist.train.images[:,:]
    #train_labels = mnist.train.labels

    val_data = train_data[45000:, :]
    val_labels = train_labels[45000:]

    #val_data = mnist.train.images[45000:,:]
    #val_labels = mnist.train.labels[45000:]

    train_data = train_data[:45000, :]
    train_labels = train_labels[:45000]

    #train_data = mnist.train.images[:45000,:]
    #train_labels = mnist.train.labels[:45000]

    train_stats = np.zeros((10,), np.float32)
    for label in train_labels:
      train_stats[label] += 1.0/45000.0

    #val_data = mnist.train.images[50000:,:]
    #val_labels = mnist.train.labels[50000:]

    #train_data = mnist.train.images[:50000,:]
    #train_labels = mnist.train.labels[:50000]
    #train_stats = np.zeros((10,), np.float32)
    #for label in train_labels:
    #  train_stats[label] += 1.0/50000.0

    """
    val_data = mnist.train.images[50000:, :]
    val_labels = mnist.train.labels[50000:]

    train_data = mnist.train.images[:50000, :]
    train_labels = mnist.train.labels[:50000]
    train_stats = np.zeros((10,), np.float32)
    for label in train_labels:
        train_stats[label] += 1.0 / 50000.0
    """

    train_data = train_data.reshape((-1,28,28,1))

    val_data = np.concatenate([val_data, mnist.validation.images[:,:]])
    val_labels = np.concatenate([val_labels, mnist.validation.labels])
    val_data = val_data.reshape((-1,28,28,1))
    
    test_data = mnist.test.images[:,:]
    test_labels = mnist.test.labels
    test_data = test_data.reshape((-1,28,28,1))

    if send_labels:
      return train_data, val_data, test_data, train_labels, val_labels, test_labels
    else:
      return train_data, val_data, test_data, train_stats

