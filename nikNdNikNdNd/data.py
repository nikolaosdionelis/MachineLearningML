import torch
import numpy as np
import pickle
import os
import torchvision
import random
import scipy
from torch.utils import data
from PIL import Image

from scipy.spatial.distance import pdist, squareform

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.transform import rescale

random.seed(2019)
np.random.seed(2019)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
CROPSIZE = 178
IMAGE_DIR = 'data/celeba/images'
ATTR_PATH = 'data/celeba/list_attr_celeba.txt'
SELECTED_ATTRS = ['Male']

def load_data(name, dataroot, batch_size, device, imgsize=None, 
                Ntrain=None, Ntest=None, n_mixtures=10, radius=3, std=0.05):
    
    print('Loading dataset {} ...'.format(name.upper()))
    data_path = dataroot+'/{}'.format(name)

    pkl_file = os.path.join(data_path, '{}_{}.pkl'.format(name, imgsize))
    if not os.path.exists(pkl_file):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        dat = create_data(
            name, data_path, batch_size, device, imgsize, Ntrain, Ntest, n_mixtures, radius, std)
        if name != 'celeba':
            with open(pkl_file, 'wb') as f:
                pickle.dump(dat, f)
    else:
        with open(pkl_file, 'rb') as f:
            dat = pickle.load(f)
    return dat 

def create_data(name, data_path, batch_size, device, imgsize, Ntrain, Ntest, n_mixtures, radius, std):
    if name == 'ring':
        delta_theta = 2*np.pi / n_mixtures
        centers_x = []
        centers_y = []
        for i in range(n_mixtures):
            centers_x.append(radius*np.cos(i*delta_theta))
            centers_y.append(radius*np.sin(i*delta_theta))

        centers_x = np.expand_dims(np.array(centers_x), 1)
        centers_y = np.expand_dims(np.array(centers_y), 1)
        centers = np.concatenate([centers_x, centers_y], 1)

        p = [1. / n_mixtures for _ in range(n_mixtures)]
        
        ith_center = np.random.choice(n_mixtures, Ntrain, p=p)
        sample_centers = centers[ith_center, :]
        sample_points = np.random.normal(loc=sample_centers, scale=std).astype('float32')

        dat = {'X_train': torch.from_numpy(sample_points)}
 
    elif name in ['mnist', 'stackedmnist']:
        nc = 1
        transform = transforms.Compose([
                transforms.Resize(imgsize), 
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))]) 

        mnist = torchvision.datasets.MNIST(root=data_path, download=True, transform=transform, train=True)
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

        if name == 'stackedmnist':
            nc = 3
            if Ntrain is None or Ntest is None:
                raise NotImplementedError('You must set Ntrain and Ntest!')
            X_training, X_test, Y_training, Y_test = stack_mnist(data_path, Ntrain, Ntest, imgsize)

            dat = {'X_train': X_training, 'Y_train': Y_training, 'X_test': X_test, 'Y_test': Y_test, 'nc': nc}

    elif name == 'cifar10':
        nc = 3
        transform = transforms.Compose([
                transforms.Resize(imgsize), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        cifar = torchvision.datasets.CIFAR10(root=data_path, download=True, transform=transform, train=True)
        train_loader = DataLoader(cifar, batch_size=1, shuffle=True, num_workers=0)
        X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
        Y_training = torch.zeros(len(train_loader))

        for i, x in enumerate(train_loader):
            X_training[i, :, :, :] = x[0]
            Y_training[i] = x[1]
            if i % 10000 == 0:
                print('i: {}/{}'.format(i, len(train_loader)))

        cifar = torchvision.datasets.CIFAR10(root=data_path, download=True, transform=transform, train=False)
        test_loader = DataLoader(cifar, batch_size=1, shuffle=False, num_workers=0)
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

    elif name == "celeba":
        nc = 3 
        transform = []
        transform.append(transforms.CenterCrop(CROPSIZE))
        transform.append(transforms.Resize(imgsize))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform)

        train_dataset = CelebA(IMAGE_DIR, ATTR_PATH, SELECTED_ATTRS, transform, 'train')
        train_loader = data.DataLoader(dataset=train_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=4)

        X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
        Y_training = torch.zeros(len(train_loader))

        for i, x in enumerate(train_loader):
            X_training[i, :, :, :] = x[0]
            Y_training[i] = x[1]
            if i % 10000 == 0:
                print('i: {}/{}'.format(i, len(train_loader)))

        test_dataset = CelebA(IMAGE_DIR, ATTR_PATH, SELECTED_ATTRS, transform, 'test')
        test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=4)

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

    else:
        raise NotImplementedError('Dataset not supported yet.')

    return dat


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def stack_mnist(data_dir, num_training_sample, num_test_sample, imageSize):
    # Load MNIST images... 60K in train and 10K in test
    fd = open(os.path.join(data_dir, 'raw/train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)
    fd.close()

    fd = open(os.path.join(data_dir, 'raw/t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
    fd.close()

    # Load MNIST labels
    fd = open(os.path.join(data_dir, 'raw/train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)
    fd.close()

    fd = open(os.path.join(data_dir, 'raw/t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)
    fd.close()

    # Form training and test using MNIST images
    ids = np.random.randint(0, trX.shape[0], size=(num_training_sample, 3))
    X_training = np.zeros(shape=(ids.shape[0], imageSize, imageSize, ids.shape[1]))
    Y_training = np.zeros(shape=(ids.shape[0]))
    for i in range(ids.shape[0]):
        cnt = 0
        for j in range(ids.shape[1]):
            xij = trX[ids[i, j], :, :, 0]
            xij = rescale(xij, (imageSize/28., imageSize/28.))
            X_training[i, :, :, j] = xij
            cnt += trY[ids[i, j]] * (10**j)
        Y_training[i] = cnt
        if i % 10000 == 0:
            print('i: {}/{}'.format(i, ids.shape[0]))
    X_training = X_training/255.

    ids = np.random.randint(0, teX.shape[0], size=(num_test_sample, 3))
    X_test = np.zeros(shape=(ids.shape[0], imageSize, imageSize, ids.shape[1]))
    Y_test = np.zeros(shape=(ids.shape[0]))
    for i in range(ids.shape[0]):
        cnt = 0
        for j in range(ids.shape[1]):
            xij = teX[ids[i, j], :, :, 0]
            xij = rescale(xij, (imageSize/28., imageSize/28.))
            X_test[i, :, :, j] = xij
            cnt += teY[ids[i, j]] * (10**j)
        Y_test[i] = cnt
        if i % 1000 == 0:
            print('i: {}/{}'.format(i, ids.shape[0]))
    X_test = X_test/255.

    X_training = torch.FloatTensor(2 * X_training - 1).permute(0, 3, 2, 1)
    X_test = torch.FloatTensor(2 * X_test - 1).permute(0, 3, 2, 1)
    Y_training = torch.LongTensor(Y_training)
    Y_test = torch.LongTensor(Y_test)
    return X_training, X_test, Y_training, Y_test
