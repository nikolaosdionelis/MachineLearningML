import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow_datasets as tfds

def load_mnist(send_labels=False):
    # mnist = input_data.read_data_sets("data/mnist_data/", one_hot=False)

    mnist = input_data.read_data_sets("data/mnist_data/", one_hot=False)
    # mnist = tfds.load(name="mnist", split=tfds.Split.TRAIN)

    np.random.seed(0)

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

    """
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
    """

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

    # mnist.train.labels = mnist.train.labels[idx]

    # mnist.train.labels = mnist.train.labels[idx]
    # mnist.train = mnist.train._replace(labels=labels[idx])
    # mnist = mnist._replace(train.labels=train.labels[idx])

    # mnist.train.images = mnist.train.images[idx]

    # mnist.train.images = mnist.train.images[idx]
    # mnist.train = mnist.train._replace(images=images[idx])
    # mnist = mnist._replace(train.images=train.images[idx])

    # print(len(mnist.train.labels))
    # sadfsdfs

    # print(len(mnist.train.labels))
    # print(len(mnist.train.images))

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

    # train_data = mnist.train.images
    # train_labels = mnist.train.labels

    train_data = mnist.train.images[idx]
    train_labels = mnist.train.labels[idx]

    # print(train_data.shape)
    # print(train_labels.shape)

    # sadfsad

    #print(train_data.shape)
    #print(train_labels.shape)

    #asdfasdfa

    #asdfs
    #asdfsz

    #mnist = input_data.read_data_sets("data/mnist_data/", one_hot=False)
    #np.random.seed(0)

    #train_data = mnist.train.images[:, :]
    #train_labels = mnist.train.labels

    train_data = mnist.train.images[::2, :]
    train_labels = mnist.train.labels[::2]

    #train_data = mnist.train.images[:, :]
    #train_labels = mnist.train.labels

    #print(train_data.shape)
    #print(train_labels.shape)

    #print(train_labels)

    # [7 3 4 ... 5 6 8]
    # [7 3 4 ... 5 6 8]

    # [7 4 1 ... 1 3 6]
    # [7 4 1 ... 1 3 6]

    #print(train_data.shape)
    #print(train_labels.shape)

    #asdfdsdfs

    #asdfs
    #asdfxs

    """
    nc = 1
        transform = transforms.Compose([
                transforms.Resize(imgsize), 
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))]) 

        mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)

        #idx = mnist.targets == 0
        #idx = mnist.targets == 1
        idx = mnist.targets == 0
        # idx += mnist.targets == 2
        # idx += mnist.targets == 3
        # idx += mnist.targets == 4
        # idx += mnist.targets == 5
        # idx += mnist.targets == 6
        # idx += mnist.targets == 7
        # idx += mnist.targets == 8
        # idx += mnist.targets == 9
        mnist.targets = mnist.targets[idx]
        mnist.data = mnist.data[idx]

        evens = list(range(0, len(mnist), 2))
        mnist = torch.utils.data.Subset(mnist, evens)

        evens = list(range(0, len(mnist), 2))
        mnist = torch.utils.data.Subset(mnist, evens)

        evens = list(range(0, len(mnist), 2))
        mnist = torch.utils.data.Subset(mnist, evens)

        evens = list(range(0, len(mnist), 2))
        mnist = torch.utils.data.Subset(mnist, evens)

        mnist2 = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)

        '''
        idx = mnist2.targets == 1
        # idx += mnist.targets == 2
        # idx += mnist.targets == 3
        # idx += mnist.targets == 4
        # idx += mnist.targets == 5
        # idx += mnist.targets == 6
        # idx += mnist.targets == 7
        # idx += mnist.targets == 8
        # idx += mnist.targets == 9
        mnist2.targets = mnist2.targets[idx]
        mnist2.data = mnist2.data[idx]

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        mnist = torch.utils.data.ConcatDataset([mnist, mnist2])

        mnist2 = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)
        '''

        #idx = mnist2.targets == 2
        idx = mnist2.targets == 1
        # idx += mnist.targets == 2
        # idx += mnist.targets == 3
        # idx += mnist.targets == 4
        # idx += mnist.targets == 5
        # idx += mnist.targets == 6
        # idx += mnist.targets == 7
        # idx += mnist.targets == 8
        # idx += mnist.targets == 9
        mnist2.targets = mnist2.targets[idx]
        mnist2.data = mnist2.data[idx]

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        mnist = torch.utils.data.ConcatDataset([mnist, mnist2])

        mnist2 = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)

        #idx = mnist2.targets == 3
        idx = mnist2.targets == 2
        # idx += mnist.targets == 2
        # idx += mnist.targets == 3
        # idx += mnist.targets == 4
        # idx += mnist.targets == 5
        # idx += mnist.targets == 6
        # idx += mnist.targets == 7
        # idx += mnist.targets == 8
        # idx += mnist.targets == 9
        mnist2.targets = mnist2.targets[idx]
        mnist2.data = mnist2.data[idx]

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        mnist = torch.utils.data.ConcatDataset([mnist, mnist2])

        mnist2 = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)

        #idx = mnist2.targets == 4
        idx = mnist2.targets == 3
        # idx += mnist.targets == 2
        # idx += mnist.targets == 3
        # idx += mnist.targets == 4
        # idx += mnist.targets == 5
        # idx += mnist.targets == 6
        # idx += mnist.targets == 7
        # idx += mnist.targets == 8
        # idx += mnist.targets == 9
        mnist2.targets = mnist2.targets[idx]
        mnist2.data = mnist2.data[idx]

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        mnist = torch.utils.data.ConcatDataset([mnist, mnist2])

        mnist2 = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)

        #idx = mnist2.targets == 5
        idx = mnist2.targets == 4
        # idx += mnist.targets == 2
        # idx += mnist.targets == 3
        # idx += mnist.targets == 4
        # idx += mnist.targets == 5
        # idx += mnist.targets == 6
        # idx += mnist.targets == 7
        # idx += mnist.targets == 8
        # idx += mnist.targets == 9
        mnist2.targets = mnist2.targets[idx]
        mnist2.data = mnist2.data[idx]

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        mnist = torch.utils.data.ConcatDataset([mnist, mnist2])

        mnist2 = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)

        #idx = mnist2.targets == 6
        idx = mnist2.targets == 5
        # idx += mnist.targets == 2
        # idx += mnist.targets == 3
        # idx += mnist.targets == 4
        # idx += mnist.targets == 5
        # idx += mnist.targets == 6
        # idx += mnist.targets == 7
        # idx += mnist.targets == 8
        # idx += mnist.targets == 9
        mnist2.targets = mnist2.targets[idx]
        mnist2.data = mnist2.data[idx]

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        mnist = torch.utils.data.ConcatDataset([mnist, mnist2])

        mnist2 = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)

        #idx = mnist2.targets == 7
        idx = mnist2.targets == 6
        # idx += mnist.targets == 2
        # idx += mnist.targets == 3
        # idx += mnist.targets == 4
        # idx += mnist.targets == 5
        # idx += mnist.targets == 6
        # idx += mnist.targets == 7
        # idx += mnist.targets == 8
        # idx += mnist.targets == 9
        mnist2.targets = mnist2.targets[idx]
        mnist2.data = mnist2.data[idx]

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        mnist = torch.utils.data.ConcatDataset([mnist, mnist2])

        mnist2 = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)

        #idx = mnist2.targets == 8
        idx = mnist2.targets == 7
        # idx += mnist.targets == 2
        # idx += mnist.targets == 3
        # idx += mnist.targets == 4
        # idx += mnist.targets == 5
        # idx += mnist.targets == 6
        # idx += mnist.targets == 7
        # idx += mnist.targets == 8
        # idx += mnist.targets == 9
        mnist2.targets = mnist2.targets[idx]
        mnist2.data = mnist2.data[idx]

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        mnist = torch.utils.data.ConcatDataset([mnist, mnist2])

        mnist2 = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)

        #idx = mnist2.targets == 9
        idx = mnist2.targets == 8
        # idx += mnist.targets == 2
        # idx += mnist.targets == 3
        # idx += mnist.targets == 4
        # idx += mnist.targets == 5
        # idx += mnist.targets == 6
        # idx += mnist.targets == 7
        # idx += mnist.targets == 8
        # idx += mnist.targets == 9
        mnist2.targets = mnist2.targets[idx]
        mnist2.data = mnist2.data[idx]

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        evens = list(range(0, len(mnist2), 2))
        mnist2 = torch.utils.data.Subset(mnist2, evens)

        mnist = torch.utils.data.ConcatDataset([mnist, mnist2])

        del mnist2

        print(len(mnist))
        # asdfdasfz

        print(len(mnist))
        print(len(mnist))

        print(len(mnist))
        print(len(mnist))

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
    """

    # train_data = mnist.train.images[:,:]
    # train_labels = mnist.train.labels

    # train_data = mnist.train.images[:,:]
    # train_labels = mnist.train.labels

    val_data = train_data[45000:, :]
    val_labels = train_labels[45000:]

    # val_data = mnist.train.images[45000:,:]
    # val_labels = mnist.train.labels[45000:]

    train_data = train_data[:45000, :]
    train_labels = train_labels[:45000]

    # train_data = mnist.train.images[:45000,:]
    # train_labels = mnist.train.labels[:45000]

    train_stats = np.zeros((10,), np.float32)
    for label in train_labels:
        train_stats[label] += 1.0 / 45000.0

    # val_data = mnist.train.images[50000:,:]
    # val_labels = mnist.train.labels[50000:]

    # train_data = mnist.train.images[:50000,:]
    # train_labels = mnist.train.labels[:50000]
    # train_stats = np.zeros((10,), np.float32)
    # for label in train_labels:
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

    train_data = train_data.reshape((-1, 28, 28, 1))

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

