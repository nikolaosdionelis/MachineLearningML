import random
import numpy as np
import torch
from torchvision.datasets import MNIST, SVHN, CIFAR10
from torchvision import transforms
import torchvision.utils as vutils

import sklearn.datasets

from utils import tensor2Var
import cv2

def construct_edge_data(data):

    def convert2edge(x):
        x = np.transpose(x, [1, 2, 0])
        gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        # edge = cv2.Canny(gray, 50, 100)

        edge = cv2.Canny(gray, 100, 200)

        return edge[np.newaxis, :]

    def transform(x):
        return ((x * 0.5 + 0.5) * 255).astype(np.uint8)

    def inverse_transform(x):
        return (x.astype(np.float32) / 255 - 0.5) / 0.5

    transform_data = transform(data)

    edge_data = [convert2edge(transform_data[i]) for i in range(data.shape[0])]

    return inverse_transform(np.stack(edge_data))

class DataLoader(object):
    def __init__(self, data, label, batch_size, clip=False):
        self.images = torch.from_numpy(data)
        self.labels = torch.from_numpy(label)
        self.batch_size = batch_size

        self.unlimit_gen = self.generator(True, clip)
        self.len = self.images.size(0)

    # 'clip' is to drop the last data which is smaller than batch size
    def generator(self, inf=False, clip=False, shuffle=True):
        while True:
            indices = np.arange(self.images.size(0))
            if shuffle:
                np.random.shuffle(indices)
            indices = torch.from_numpy(indices)
            for start in range(0, indices.size(0), self.batch_size):
                end = min(start + self.batch_size, indices.size(0))

                if clip:
                    if end - start == self.batch_size:
                        ret_images, ret_labels = self.images[indices[start: end]], \
                        self.labels[indices[start: end]]
                        yield ret_images, ret_labels
                else:
                    ret_images, ret_labels = self.images[indices[start: end]], \
                        self.labels[indices[start: end]]
                    yield ret_images, ret_labels
            if not inf: break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self, shuffle=True):
        return self.generator(shuffle=shuffle)

    def __len__(self):
        return self.len

class edge_DataLoader(object):
    def __init__(self, data, edge_data, label, batch_size, clip=False):
        self.images = torch.from_numpy(data)
        self.edges = torch.from_numpy(edge_data)
        self.labels = torch.from_numpy(label)
        self.batch_size = batch_size

        self.unlimit_gen = self.generator(True, clip)
        self.len = self.images.size(0)

    # 'clip' is to drop the last data which is smaller than batch size
    def generator(self, inf=False, clip=False):
        while True:
            indices = np.arange(self.images.size(0))
            np.random.shuffle(indices)
            indices = torch.from_numpy(indices)
            for start in range(0, indices.size(0), self.batch_size):
                end = min(start + self.batch_size, indices.size(0))

                if clip:
                    if end - start == self.batch_size:
                        ret_images, ret_edges, ret_labels = self.images[indices[start: end]], \
                            self.edges[indices[start:end]], self.labels[indices[start: end]]
                        yield ret_images, ret_edges, ret_labels
                else:
                    ret_images, ret_edges, ret_labels = self.images[indices[start: end]], \
                            self.edges[indices[start:end]], self.labels[indices[start: end]]
                    yield ret_images, ret_edges, ret_labels
            if not inf: break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self):
        return self.generator()

    def __len__(self):
        return self.len


class label_dict:
    def __init__(self, data, labels):
        self.dict = self.create_dict(data, labels)

    def create_dict(self, data, labels):
        _dict = {}

        max_label = labels.max()
        min_label = labels.min()

        for l in range(min_label, max_label+1):
            _dict[l] = data[labels==l]

        return _dict

    def sample(self, label, size):
        max_length = self.dict[label].shape[0]

        assert size <= max_length

        selected_indices = random.sample(range(max_length), size)

        return self.dict[label][selected_indices]

    def sample_all(self, size):
        return_data = []
        for l in self.dict.keys():
            return_data.append(self.sample(l, size))

        return np.vstack(return_data)


def create_data_from_dataset(raw_loader, indices):
    images, labels = [], []
    for idx in indices:
        image, label = raw_loader[idx]
        images.append(image)
        labels.append(label)

    images = np.stack(images, 0).astype(np.float32)
    labels = np.hstack(labels).astype(np.int64)

    return images, labels

class P_d_bar():
    def __init__(self, args, data, label, batch_size):
        self._type = args.p_d_bar_type
        self.beta_2 = args.beta_2
        self.beta_1 = args.beta_1
        self.p_d_1 = DataLoader(data, label, batch_size, True)
        self.p_d_2 = DataLoader(data, label, batch_size, True)

        choices = ['normal', 'uniform', 'inter', 'huge_normal']

        assert self._type in choices, 'Type %s not exists in class P_d_bar' % self._type 

    def construct_p_d(self, data_1, data_2):
        beta = self.beta_2
        beta_1 = self.beta_1
        beta_2 = self.beta_2

        if self._type == 'normal':
            noise = torch.FloatTensor(data_1.size()).normal_(0, beta)

        elif self._type == 'uniform':
            noise = torch.FloatTensor(data_1.size()).uniform_(-beta, beta)
    
        elif self._type == 'inter':
            if len(data_1.size()) == 2:
                _beta = torch.FloatTensor(data_1.size(0), 1).uniform_(beta_1, beta_2)
            elif len(data_1.size()) == 4:
                _beta = torch.FloatTensor(data_1.size(0), 1, 1, 1).uniform_(beta_1, beta_2)
            _beta = tensor2Var(_beta)

            out = _beta * data_1 + (1 - _beta) * data_2

            return out
            
        elif self._type == 'huge_normal':
            noise = torch.randn(data_1.size()) * (1 + beta)

        return data_1 + tensor2Var(noise)

    def get_data(self):
        data_1, _ = self.p_d_1.next()
        data_2, _ = self.p_d_2.next()

        data_1 = tensor2Var(data_1)
        data_2 = tensor2Var(data_2)

        return data_1, data_2

    def sample(self):
        data_1, data_2 = self.get_data()

        return self.construct_p_d(data_1, data_2)

    def sample_feat(self, feat_extracter):
        data_1, data_2 = self.get_data()

        feat_1, feat_2 = feat_extracter(data_1), feat_extracter(data_2)

        return self.construct_p_d(feat_1, feat_2)


class edge_P_d_bar():
    def __init__(self, args, data, edge, label, batch_size):
        self._type = args.p_d_bar_type
        self.beta_2 = args.beta_2
        self.beta_1 = args.beta_1

        self.p_d_1 = DataLoader(data, edge, label, batch_size, True)
        self.p_d_2 = DataLoader(data, edge, label, batch_size, True)

        choices = ['normal', 'uniform', 'inter', 'huge_normal']

        assert self._type in choices, 'Type %s not exists in class P_d_bar' % self._type 

    def construct_p_d(self, data_1, data_2):
        beta = self.beta_2
        beta_1 = self.beta_1
        beta_2 = self.beta_2

        if self._type == 'normal':
            noise = torch.FloatTensor(data_1.size()).normal_(0, beta)

        elif self._type == 'uniform':
            noise = torch.FloatTensor(data_1.size()).uniform_(-beta, beta)
    
        elif self._type == 'inter':
            if len(data_1.size()) == 2:
                _beta = torch.FloatTensor(data_1.size(0), 1).uniform_(beta_1, beta_2)
            elif len(data_1.size()) == 4:
                _beta = torch.FloatTensor(data_1.size(0), 1, 1, 1).uniform_(beta_1, beta_2)
            _beta = tensor2Var(_beta)

            out = _beta * data_1 + (1 - _beta) * data_2

            return out
            
        elif self._type == 'huge_normal':
            noise = torch.randn(data_1.size()) * (1 + beta)

        return data_1 + tensor2Var(noise)

    def get_data(self):
        data_1, _ = self.p_d_1.next()
        data_2, _ = self.p_d_2.next()

        data_1 = tensor2Var(data_1)
        data_2 = tensor2Var(data_2)

        return data_1, data_2

    def sample(self):
        data_1, data_2 = self.get_data()

        return self.construct_p_d(data_1, data_2)

    def sample_feat(self, feat_extracter):
        data_1, data_2 = self.get_data()

        feat_1, feat_2 = feat_extracter(data_1), feat_extracter(data_2)

        return self.construct_p_d(feat_1, feat_2)
    
def get_mnist_loaders(args):
    # transform = transforms.Compose([transforms.ToTensor()])

    transform = transforms.Compose([transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    training_set = MNIST(args.data_root, train=True, download=True, transform=transform)
    dev_set = MNIST(args.data_root, train=False, download=True, transform=transform, 
        target_transform=lambda x: 1 if x == args.train_class else 0)

    if args.seed != -1:
        np.random.seed(args.seed)
    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)

    mask[np.where(labels == args.train_class)[0][:]] = True
    
    train_class_indices = indices[mask]

    print('train class size: %d' % train_class_indices.shape[0])

    train_data, train_label = create_data_from_dataset(training_set, train_class_indices)

    dev_data, dev_label = create_data_from_dataset(dev_set, np.arange(len(dev_set)))

    train_loader = DataLoader(train_data, train_label, args.train_batch_size, True)

    p_d = DataLoader(train_data, train_label, args.train_batch_size, True)
    p_d_2 = DataLoader(train_data, train_label, args.train_batch_size, True)

    p_d_bar = P_d_bar(args, train_data, train_label, args.train_batch_size)

    dev_loader = DataLoader(dev_data, dev_label, args.dev_batch_size)

    return train_loader, p_d, p_d_bar, dev_loader, p_d_2

def get_svhn_loaders(args):
    transform = transforms.Compose([transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = SVHN(args.data_root, split='train', download=True, transform=transform)
    dev_set = SVHN(args.data_root, split='test', download=True, transform=transform)

    def preprocess(data_set):
        for i in range(len(data_set.data)):
            if data_set.labels[i] == 10:
                data_set.labels[i] = 0
    preprocess(training_set)
    preprocess(dev_set)

    if args.seed != -1:
        np.random.seed(args.seed)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        # print(np.where(labels == i)[0].shape)
        mask[np.where(labels == i)[0][: args.size_valid_data // 10]] = True
    valid_indices, train_indices = indices[mask], indices[~ mask]
    print('train size: %d, valid size: %d, dev size: %d' % 
        (train_indices.shape[0], valid_indices.shape[0], len(dev_set)))

    train_data, train_label = create_data_from_dataset(training_set, train_indices)


    if len(valid_indices) == 0:
        valid_data, valid_label = create_data_from_dataset(training_set, train_indices)
    else:
        valid_data, valid_label = create_data_from_dataset(training_set, valid_indices)

    dev_data, dev_label = create_data_from_dataset(dev_set, np.arange(len(dev_set)))

    train_loader = DataLoader(train_data, train_label, args.train_batch_size)

    ###
    total_data = np.vstack([train_data, valid_data])
    total_label = np.hstack([train_label, valid_label])
    ###

    valid_loader = DataLoader(valid_data, valid_label, args.train_batch_size)
    p_d = DataLoader(train_data, train_label, args.train_batch_size)
    p_d_2 = DataLoader(train_data, train_label, args.train_batch_size)

    p_d_bar = P_d_bar(args, train_data, train_label, args.train_batch_size)
    dev_loader = DataLoader(dev_data, dev_label, args.dev_batch_size)

    return train_loader, valid_loader, p_d, p_d_bar, dev_loader, p_d_2

def get_cifar_loaders(args):
    transform = transforms.Compose([transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = CIFAR10('cifar', train=True, download=True, transform=transform)
    dev_set = CIFAR10('cifar', train=False, download=True, transform=transform,
        target_transform=lambda x: 1 if x == args.train_class else 0)

    if args.seed != -1:
        np.random.seed(args.seed)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    
    mask[np.where(labels == args.train_class)[0][:]] = True
    
    train_class_indices = indices[mask]

    print('train class size: %d' % train_class_indices.shape[0])

    train_data, train_label = create_data_from_dataset(training_set, train_class_indices)

    # edge_train_data = construct_edge_data(train_data)

    # print(edge_train_data.shape)

    dev_data, dev_label = create_data_from_dataset(dev_set, np.arange(len(dev_set)))

    # edge_dev_data = construct_edge_data(dev_data)

    train_loader = DataLoader(train_data, train_label, args.train_batch_size)

    p_d = DataLoader(train_data, train_label, args.train_batch_size)
    p_d_2 = DataLoader(train_data, train_label, args.train_batch_size)

    p_d_bar = P_d_bar(args, train_data, train_label, args.train_batch_size)

    dev_loader = DataLoader(dev_data, dev_label, args.dev_batch_size)

    return train_loader, p_d, p_d_bar, dev_loader, p_d_2

###############################################################################################
### TOY DATASET

def make_toy_data(dataset='4gaussians', data_size=30000):
    batch_size = 5
    data_list = []
    label_list = []
    seed = 1
    np.random.seed(seed)
    if dataset == '4gaussians':
        centers_x = [-1, -1, 1, 1]
        centers_y = [1, -1, -1, 1]
        for i in range(data_size // 4):
            for j in range(len(centers_x)):
                x = centers_x[j]
                y = centers_y[j]
                # point = np.random.randn(2) * 0.05
                point = np.random.randn(2) * 0.3
                point[0] += x
                point[1] += y

                data_list.append(point)

                if j == 0 or j == 2:
                    label_list.append(0)
                else:
                    # label_list.append(1)
                    label_list.append(0)

        data = np.array(data_list, dtype='float32')
        label = np.array(label_list, dtype=int)

    elif dataset == 'swissroll':
        data_list = []
        size = 0

        # while size < data_size:
        data = sklearn.datasets.make_swiss_roll(
            n_samples=data_size,
            noise=0.4)[0]
        data = data.astype('float32')[:, [0, 2]]
        data /= 7.5  # stdev plus a little

            # data_list.append(data)
            # size += batch_size
        label = np.zeros((data.shape[0], ), dtype=int)

    elif dataset == 'circles':
        size = 0
        sample_size = 20000 if 20000 > data_size else data_size * 2

        indices = np.arange(data_size)
        np.random.shuffle(indices)
        data, label = sklearn.datasets.make_circles(
            n_samples=sample_size,
            noise=0.06,
            factor=0.4,
            random_state=seed
        )
        data = data[indices, :]
        label = label[indices]
        data = data.astype('float32')
        data *= 3.0
    elif dataset == 'circles_2':
        size = 0
        sample_size = 20000 if 20000 > data_size else data_size * 2
        indices = np.arange(data_size * 2 // 3)
        np.random.shuffle(indices)
        data = sklearn.datasets.make_circles(
            n_samples=sample_size,
            noise=0.03,
            factor=0.3,
        )[0][indices, :]

        data = data.astype('float32')
        data *= 2
        
        data_list.append(data)
        label_list.append(np.zeros(shape=(data.shape[0], ), dtype=int ))

        indices = np.arange(data_size // 3)
        np.random.shuffle(indices)

        data = sklearn.datasets.make_circles(
            n_samples=sample_size,
            noise=0.06,
            factor=1,
        )[0][indices, :]
        data = data.astype('float32')
        data *= 1.3
        
        data_list.append(data)
        label_list.append(np.ones(shape=(data.shape[0], ), dtype=int ))

        data = np.vstack(data_list)
        label = np.hstack(label_list)
    elif dataset == 'self_classify':
        data = np.loadtxt('2d_data.txt', delimiter=',')

        label = data[:, 2].astype(int)
        data = data[:, :2].astype('float32')

        mean = np.mean(data, 0)
        std = np.std(data, 0)

        data = (data - mean) / std
    elif dataset == 'self_circles':
        data = np.loadtxt('2d_circles.txt', delimiter=',')

        label = data[:, 2].astype(int)
        data = data[:, :2].astype('float32')

        mean = np.mean(data, 0)
        std = np.std(data, 0)

        data = (data - mean) / std

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    return data[indices], label[indices]


def get_toy_loaders(args):

    label_data, label_label = make_toy_data(dataset=args.dataset, data_size=args.size_labeled_data)
    unlabel_data, unlabel_label = make_toy_data(dataset=args.dataset, data_size=args.size_unlabeled_data)
    dev_data, dev_label = make_toy_data(dataset=args.dataset, data_size=args.size_test_data)

    print('labeled size: %d, unlabeled size: %d, dev size: %d' % (len(label_data), len(unlabel_data), 
        len(dev_data)))

    labeled_loader = DataLoader(label_data, label_label, args.train_batch_size)

    unlabeled_loader = DataLoader(label_data, label_label, args.train_batch_size)
    unlabeled_loader2 = DataLoader(label_data, label_label, args.train_batch_size)
    unlabeled_loader3 = DataLoader(label_data, label_label, args.train_batch_size)
    dev_loader = DataLoader(dev_data, dev_label, args.dev_batch_size)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, unlabeled_loader3, dev_loader