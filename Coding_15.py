# Deep Generative Models
# GANs and VAEs, Generative Models

# random noise
# from random noise to a tensor

# We use batch normalisation.
# GANs are very difficult to train. Super-deep models. This is why we use batch normalisation.

# Anomaly detection (AD)
# Unsupervised machine learning

# GANs for super-resolution
# Generative Adversarial Networks, GANs

# GANs and LSTM RNNs
# use LSTM RNNs together with GANs

# combine the power of LSTM RNNs and GANs
# it is possible to use LSTM RNN together with GANs

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models
# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# the BigGAN dataset
# BigGAN => massive dataset
# latent space, BigGAN, GANs

# down-sampling, sub-sample, pooling
# throw away samples, pooling, max-pooling

# partial derivatives
# loss function and partial derivatives

# https://github.com/Students-for-AI/The-Academy-of-AI
# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models

# Generator G and Discriminator D
# the loss function of the Generator G

# up-convolution
# We use a filter we do up-convolution with.

# use batch normalisation
# GANs are very difficult to train and this is why we use batch normalisation.

# We normalize across a batch.
# Mean across a batch. We use batches. Normalize across a batch.

# the ReLU activation function
# ReLU is the most common activation function. We use ReLU.

# use: https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# use PyTorch
import torch

#import torch
import torchvision

from torchvision import datasets, transforms

# use matplotlib
import matplotlib.pyplot as plt

batch_size = 128

# download the training dataset
train_data = datasets.FashionMNIST(root='fashiondata/',
                                   transform=transforms.ToTensor(),
                                   train=True,
                                   download=True)

# we create the train data loader
train_loader = torch.utils.data.DataLoader(train_data,
                                           shuffle=True,
                                           batch_size=batch_size)

# class for D and G
# we train the discriminator and the generator

# we make the discriminator
class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # one-channel, stride of 2
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        # do 1D convolution

        # do 2D convolution
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)

        # fully connected fc
        self.fc = torch.nn.Linear(128*7*7, 1)
        # the output is a single number, one number

        # we need fc
        # we need a fully connected layer

        # batch normalisation layer
        self.bn1 = torch.nn.BatchNorm2d(64)
        # after the 1D convolution

        # second batch normalization layer
        self.bn2 = torch.nn.BatchNorm2d(128)
        # after the 2D convolution

        # activation function
        #self.af = torch.nn.Sigmoid()
        self.af = torch.nn.ReLU()

        # for the output
        self.s = torch.nn.Sigmoid()

    def forward(selfself, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.af(x)
        x = self.conv2(x)

        x = self.bn2(x)
        x = self.af(x)

        # reshape
        x = x.view(-1, 128*7*7)
        # we do not care about the rows, hence "-1"

        # we do not care about the batch size
        # we do not care about the rows, hence "-1"

        # fully connected (fc)
        x = self.fc(x)

        x = self.s(x)

        return x

# this was for the discriminator
# we now do the same for the generator

# Generator G
class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # random noise
        # create random noise

        # 128 to 1256
        self.dense1 = torch.nn.Linear(128, 256)

        self.dense2 = torch.nn.Linear(256, 1024)
        self.dense3 = torch.nn.Linear(1024, 128*7*7)

        # convolution layer
        self.uconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # we use a stride of 2

        # second convolution layer
        self.uconv2 = torch.nn.ConvTranspose2d(64, 1, 4, 2, 1)

        # batch normalization
        self.bn1 = torch.nn.BatchNorm1d(256)

        # second batch normalization layer
        self.bn2 = torch.nn.BatchNorm1d(1024)
        # this is after dense2

        # this is after dense3
        self.bn3 = torch.nn.BatchNorm1d(128*7*7)

        self.bn4 = torch.nn.BatchNorm2d(64)

        # use ReLU
        self.af = torch.nn.ReLU()

        self.s = torch.nn.Sigmoid()

        # grayscale images
        # we use grayscale images

    # forward function
    def forward(self, z):
        #z = self.dense1(z)
        #z = self.bn1(z)
        #z = self.af(z)

        z = self.af(self.bn1(self.dense1(z)))

        #z = self.dense2(z)
        #z = self.bn2(z)
        #z = self.af(z)

        z = self.af(self.bn2(self.dense2(z)))

        z = self.af(self.bn3(self.dense3(z)))

        # up-convolution
        z = self.af(self.bn4(self.uconv1(z)))

        # stable training
        # batch normalization for stable training

        z = self.s(self.uconv2(z))

        return z

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb
# use: https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

d = Discriminator()
g = Generator()

epochs = 100

dlr = 0.0003
glr = 0.0003

#d_optimizer = torch.optim.Adam(d.parameters(), lr=dlr)
#g_optimizer = torch.

# instantiate the model
d = Discriminator()
g = Generator()

# training hyperparameters
epochs = 100

# training hyperparameters
dlr = 0.0003
glr = 0.0003

# we use Adam
d_optimizer = torch.optim.Adam(d.parameters(), lr=dlr)
g_optimizer = torch.optim.Adam(g.parameters(), lr=glr)

dcosts = []
gcosts = []

plt.ion()
fig = plt.figure()
loss_ax = fig.add_subplot(121)

loss_ax.set_xlabel('Batch')
loss_ax.set_ylabel('Cost')

loss_ax.set_ylim(0, 0.2)
generated_img = fig.add_subplot(122)

plt.show()

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models
# use: https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

def train(epochs):
    for epoch in range(epochs):
        #for batch_idx, (real_images, _) enumerate(train_loader):

        for batch_idx, (real_images, _) in enumerate(train_loader):
            # random noise
            z = torch.randn(batch_size, 128)

            # latent space
            # our latent space is 128

            # generate images
            generated_images = g(z)

            gen_pred = d(generated_images)

            real_pred = d(real_images)

            # cost function
            # create loss function

            # sum over batches
            #dcost = -torch.sum(torch.log(real_pred))

            dcost = -torch.sum(torch.log(real_pred)) - torch.sum(torch.log(1 - real_pred))

            # we sum over the batches
            gcost = -torch.sum(torch.log(gen_pred)) / batch_size
            # use: . / batch_size

            d_optimizer.zero_grad()

            # delete stuff from the computational graph
            dcost.backward(retain_graph=True)

            d_optimizer.step()

            g_optimizer.zero_grad()
            gcost.backward()
            g_optimizer.step()

            # batch normalization
            # different between training and testing

            # batch normalization is different between training and testing

            # running average during testing
            # we use the running average during testing

            if batch_idx == 10000:
                # batch normalization is different between training and testing
                g.eval()

                noise_input = torch.randn(1,128)
                generated_image = g(noise_input)

                # use .squeeze()
                generated_img.imshow(generated_image.detach().squeeze())

                # batch normalization is different between training and testing
                g.train()

            dcost /= batch_size
            gcost /= batch_size

            # for every epoch, print
            print('Epoch:', epoch, '\tBatch:', batch_idx)

            dcosts.append(dcost.item())
            gcosts.append(gcost.item())

            loss_ax.plot(dcosts, 'r')
            loss_ax.plot(gcosts, 'b')

            fig.canvas.draw()

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

train(epochs)


