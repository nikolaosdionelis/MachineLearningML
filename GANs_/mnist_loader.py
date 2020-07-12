import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def load_mnist(send_labels=False):
    mnist = input_data.read_data_sets("data/mnist_data/", one_hot=False)
    np.random.seed(0)

    #train_data = mnist.train.images[::2, :]
    #train_labels = mnist.train.labels[::2]

    #mnist.train.images = mnist.train.images[::2, :]
    #mnist.train.labels = mnist.train.labels[::2]

    # items[node.ind].v = node.v
    # items[node.ind] = items[node.ind]._replace(v=node.v)

    #mnist = mnist._replace(train.images=train.images[::2, :])
    #mnist = mnist._replace(train.labels=train.labels[::2])

    #mnist.train.images = mnist.train.images[::2, :]
    #mnist.train.labels = mnist.train.labels[::2]

    #train_data = mnist.train.images[:, :]
    #train_labels = mnist.train.labels

    #print(train_data.shape)
    #print(train_labels.shape)

    #print(train_labels)
    #asdfsdkfs

    # train_data = mnist.train.images[:, :]
    # train_labels = mnist.train.labels

    #train_data = mnist.train.images[:, :]
    #train_labels = mnist.train.labels

    #val_data = mnist.train.images[50000:, :]
    #val_labels = mnist.train.labels[50000:]

    #val_data = mnist.train.images[50000:, :]
    #val_labels = mnist.train.labels[50000:]

    #val_data = mnist.train.images[50000::2, :]
    #val_labels = mnist.train.labels[50000::2]

    #val_data = mnist.train.images[50000::4, :]
    #val_labels = mnist.train.labels[50000::4]

    #val_data = mnist.train.images[50000::5, :]
    #val_labels = mnist.train.labels[50000::5]

    val_data = mnist.train.images[50000::8, :]
    val_labels = mnist.train.labels[50000::8]

    #print(val_data.shape)
    #print(val_labels.shape)

    #print(val_labels)
    #asdfasdf

    #train_data = mnist.train.images[:50000, :]
    #train_labels = mnist.train.labels[:50000]

    #train_data = mnist.train.images[:50000, :]
    #train_labels = mnist.train.labels[:50000]

    #train_data = mnist.train.images[:50000:2, :]
    #train_labels = mnist.train.labels[:50000:2]

    #train_data = mnist.train.images[:50000:4, :]
    #train_labels = mnist.train.labels[:50000:4]

    #train_data = mnist.train.images[:50000:5, :]
    #train_labels = mnist.train.labels[:50000:5]

    train_data = mnist.train.images[:50000:8, :]
    train_labels = mnist.train.labels[:50000:8]

    train_stats = np.zeros((10,), np.float32)
    for label in train_labels:
        train_stats[label] += 1.0 / 50000.0

    train_data = train_data.reshape((-1, 28, 28, 1))

    #print(train_data.shape)
    #print(train_labels.shape)

    #print(train_labels)
    #asdfsdkfs

    #print(train_data.shape)
    #print(train_labels.shape)

    #print(train_labels)
    #asdfsdkfs

    #print(val_data.shape)
    #print(val_labels.shape)

    #print(val_labels)
    #asdfsdkfs

    # [7 3 4 ... 0 4 0]
    # [7 3 4 ... 0 4 0]

    # list[::2], list[1::2]
    # use: list[::2], list[1::2]

    val_data = np.concatenate([val_data, mnist.validation.images[:, :]])
    val_labels = np.concatenate([val_labels, mnist.validation.labels])
    val_data = val_data.reshape((-1, 28, 28, 1))

    test_data = mnist.test.images[:, :]
    test_labels = mnist.test.labels
    test_data = test_data.reshape((-1, 28, 28, 1))

    if send_labels:
        return train_data, val_data, test_data, train_labels, val_labels, test_labels
    else:
        return train_data, val_data, test_data, train_stats

