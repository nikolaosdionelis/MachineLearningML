import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import torch.nn.init as nn_init

from torch.nn.utils import spectral_norm

from utils import tensor2Var

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name, map_location=lambda storage, loc: storage))
    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, input):
        if self.training:
            noise = Variable(input.data.new(input.size()).normal_(std=self.sigma))
            return input + noise
        else:
            return input

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
    
    def forward(self, input):
        return self.func(input)

class WN_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_Linear, self).__init__(in_features, out_features, bias=bias)
        if train_scale:
            self.weight_scale = Parameter(torch.ones(self.out_features))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_features))

        self.train_scale = train_scale 
        self.init_mode = False
        self.init_stdv = init_stdv

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(0, std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)

        # normalize weight matrix and linear projection
        norm_weight = self.weight * (weight_scale.unsqueeze(1) / torch.sqrt((self.weight ** 2).sum(1, keepdim=True) + 1e-6)).expand_as(self.weight)
        activation = F.linear(input, norm_weight)

        if self.init_mode == True:
            mean_act = activation.mean(0).squeeze(0)
            activation = activation - mean_act.expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(0) + 1e-6).squeeze(0)
            activation = activation * inv_stdv.expand_as(activation)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias.expand_as(activation)

        return activation

class WN_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if train_scale:
            self.weight_scale = Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_channels))
        
        self.train_scale = train_scale 
        self.init_mode = False
        self.init_stdv = init_stdv

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)
        # normalize weight matrix and linear projection [out x in x h x w]
        # for each output dimension, normalize through (in, h, w) = (1, 2, 3) dims
        norm_weight = self.weight * (weight_scale[:,None,None,None] / torch.sqrt((self.weight ** 2).sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True) + 1e-6)).expand_as(self.weight)
        activation = F.conv2d(input, norm_weight, bias=None, 
                              stride=self.stride, padding=self.padding, 
                              dilation=self.dilation, groups=self.groups)

        if self.init_mode == True:
            mean_act = activation.mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act[None,:,None,None].expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(3).mean(2).mean(0) + 1e-6).squeeze()
            activation = activation * inv_stdv[None,:,None,None].expand_as(activation)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias[None,:,None,None].expand_as(activation)

        return activation

class WN_ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias)
        if train_scale:
            self.weight_scale = Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_channels))
        
        self.train_scale = train_scale 
        self.init_mode = False
        self.init_stdv = init_stdv

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input, output_size=None):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)
        # normalize weight matrix and linear projection [in x out x h x w]
        # for each output dimension, normalize through (in, h, w)  = (0, 2, 3) dims
        norm_weight = self.weight * (weight_scale[None,:,None,None] / torch.sqrt((self.weight ** 2).sum(3, keepdim=True).sum(2, keepdim=True).sum(0, keepdim=True) + 1e-6)).expand_as(self.weight)
        output_padding = self._output_padding(input, output_size, 
            self.stride, self.padding, self.kernel_size)
        activation = F.conv_transpose2d(input, norm_weight, bias=None, 
                                        stride=self.stride, padding=self.padding, 
                                        output_padding=output_padding, groups=self.groups)

        if self.init_mode == True:
            mean_act = activation.mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act[None,:,None,None].expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(3).mean(2).mean(0) + 1e-6).squeeze()
            activation = activation * inv_stdv[None,:,None,None].expand_as(activation)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias[None,:,None,None].expand_as(activation)

        return activation

def identity(_input):
    return _input

class feature_Discriminator(myModel):
    def __init__(self, args, input_size):
        super(feature_Discriminator, self).__init__()

        self.input_size = input_size

        # if args.spectral_norm:
        #     func = spectral_norm
        # else:
        func = identity

        self.feat_net = nn.Sequential(
            func(nn.Linear(self.input_size, 400)), nn.ReLU(),
            func(nn.Linear(400, 200)), nn.ReLU(),
            func(nn.Linear(200, 100)), nn.ReLU(),
        )

        self.critic = nn.Sequential(
            func(nn.Linear(100, 1))
            )
        self.discriminator = nn.Sequential(
            func(nn.Linear(100, 1)),
            nn.Sigmoid()
            )
        # for p in self.modules():
        #     if isinstance(p, nn.Linear):
        #         nn.init.kaiming_normal(p.weight.data)

        # self._initialize_weights()

    def forward(self, X, require='critic'):
        if X.dim() == 4:
            X = X.view(X.size(0), -1)

        if require == 'critic':
            return self.critic(self.feat_net(X)).view(-1)
        elif require == 'discriminator':
            return self.discriminator(self.feat_net(X)).view(-1)
        elif require == 'feat':
            return self.feat_net(X)


def activation(args):
    if args.class_activation == 'selu':
        return nn.SELU()
    elif args.class_activation == 'relu':
        return nn.LeakyReLU(0.2)



class VAE(myModel):
    def __init__(self, args, feature_size):
        super(VAE, self).__init__()

        self.conv_feature = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), # 32
            nn.BatchNorm2d(32),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(32, 64, 5, 2, 2), # 16
            nn.BatchNorm2d(64),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 5, 2, 2), # 8
            nn.BatchNorm2d(128),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            )

        self.mean_layer = nn.Sequential(
            nn.Conv2d(128, feature_size, 4, 1, 0)
            )
        self.std_layer = nn.Sequential(
            nn.Conv2d(128, feature_size, 4, 1, 0)
            )
        self.decode_layer = nn.Sequential(
            nn.ConvTranspose2d(feature_size, 128, 5, 2, 0), # 8
            nn.BatchNorm2d(128),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, 5, 2, 2), # 16
            nn.BatchNorm2d(64),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 32, 5, 2, 2), # 32
            nn.BatchNorm2d(32),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, 3, 5, 2, 2), # 64
            nn.Tanh()
            )
    def encode(self, inputs):
        assert len(inputs.size()) == 4

        feature = self.conv_feature(inputs)

        # print(feature.shape)

        return self.mean_layer(feature), self.std_layer(feature)

    def reparameterize(self, mean_input, logvar_input):
        if self.training:
            std = torch.exp(0.5 * logvar_input)
            epsilon = tensor2Var(torch.randn(std.size()))
            # print(std)
            return mean_input + std * epsilon
        else:
            return mean_input

    def get_features(self, inputs):
        mean, logvar = self.encode(inputs)
        embed_reparameterized = self.reparameterize(mean, logvar)

        return embed_reparameterized

    def decode(self, inputs):
        assert len(inputs.size()) == 4
        return self.decode_layer(inputs)[:, :, 1:, 1:]
    def forward(self, inputs):
        mean, logvar = self.encode(inputs)
        embed_reparameterized = self.reparameterize(mean, logvar)
        return self.decode(embed_reparameterized), mean, logvar

class AE(myModel):
    def __init__(self, args, feature_size):
        super(AE, self).__init__()

        self.conv_feature = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), # 32
            nn.BatchNorm2d(32),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(32, 64, 5, 2, 2), # 16
            nn.BatchNorm2d(64),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 5, 2, 2), # 8
            nn.BatchNorm2d(128),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            )

        self.mean_layer = nn.Sequential(
            nn.Conv2d(128, feature_size, 4, 1, 0),
            # nn.Tanh()
            )

        self.decode_layer = nn.Sequential(
            nn.ConvTranspose2d(feature_size, 128, 5, 2, 0), # 8
            nn.BatchNorm2d(128),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, 5, 2, 2), # 16
            nn.BatchNorm2d(64),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 32, 5, 2, 2), # 32
            nn.BatchNorm2d(32),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, 3, 5, 2, 2), # 64
            nn.Tanh()
            )
    def encode(self, inputs):
        assert len(inputs.size()) == 4

        feature = self.conv_feature(inputs)

        # print(feature.shape)

        return self.mean_layer(feature), 0

    def reparameterize(self, mean_input, logvar_input):
        return mean_input

    def get_features(self, inputs):
        mean, _ = self.encode(inputs)

        return mean

    def decode(self, inputs):
        assert len(inputs.size()) == 4
        return self.decode_layer(inputs)[:, :, 1:, 1:]

    def forward(self, inputs):
        mean, _ = self.encode(inputs)
        embed_reparameterized = self.reparameterize(mean, _)
        return self.decode(embed_reparameterized), mean, _

class WN_Classifier(myModel):
    def __init__(self, args, feature_size):
        super(WN_Classifier, self).__init__()

        print('===> Init small-conv for {}'.format(args.dataset))

        self.noise_size = args.noise_size
        self.num_label  = args.num_label
        self.n_channels = args.n_channels

        if args.dataset == 'svhn':
            n_filter_1, n_filter_2 = 64, 128
        elif args.dataset == 'cifar':
            n_filter_1, n_filter_2 = 96, 192
        else:
            raise ValueError('dataset not found: {}'.format(args.dataset))

        # Assume X is of size [batch x 3 x 32 x 32]
        self.feat_net = nn.Sequential(

            WN_Conv2d(args.n_channels, n_filter_1, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_1     , n_filter_1, 3, 2, 1), nn.LeakyReLU(0.2),

            WN_Conv2d(n_filter_1     , n_filter_2, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2     , n_filter_2, 3, 2, 1), nn.LeakyReLU(0.2),

            WN_Conv2d(n_filter_2     , n_filter_2, 3, 1, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2     , n_filter_2, 1, 1, 0), nn.LeakyReLU(0.2),

            Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
        )

        if args.linear:
            self.out_class = WN_Linear(n_filter_2, self.num_label, train_scale=True, init_stdv=0.1)
        else:
            self.out_class = nn.Sequential(
                WN_Linear(n_filter_2, n_filter_2), nn.LeakyReLU(0.2),
                WN_Linear(n_filter_2, n_filter_2), nn.LeakyReLU(0.2),
                WN_Linear(n_filter_2, n_filter_2), nn.LeakyReLU(0.2),
                WN_Linear(n_filter_2, self.num_label, train_scale=True, init_stdv=0.1)
                )

        # for p in self.modules():
        #     if isinstance(p, nn.Linear) or isinstance(p, nn.Conv2d):
        #         nn.init.kaiming_normal(p.weight.data)

# class WN_Classifier(myModel):
#     def __init__(self, args, feature_size):
#         super(WN_Classifier, self).__init__()

#         print('===> Init small-conv for {}'.format(args.dataset))

#         self.noise_size = args.noise_size
#         self.num_label  = args.num_label

#         if args.dataset == 'svhn':
#             n_filter_1, n_filter_2 = 64, 128
#         elif args.dataset == 'cifar':
#             n_filter_1, n_filter_2 = 96, 192
#         else:
#             raise ValueError('dataset not found: {}'.format(args.dataset))

#         # Assume X is of size [batch x 3 x 32 x 32]
#         self.feat_net = nn.Sequential(

#             nn.Sequential(GaussianNoise(0.05), nn.Dropout2d(0.15)) if args.dataset == 'svhn' \
#                 else nn.Sequential(GaussianNoise(0.05), nn.Dropout2d(0.2)),

#             WN_Conv2d(         3, n_filter_1, 3, 1, 1), nn.LeakyReLU(0.2),
#             WN_Conv2d(n_filter_1, n_filter_1, 3, 1, 1), nn.LeakyReLU(0.2),
#             WN_Conv2d(n_filter_1, n_filter_1, 3, 2, 1), nn.LeakyReLU(0.2),

#             nn.Dropout2d(0.5) if args.dataset == 'svhn' else nn.Dropout(0.5),

#             WN_Conv2d(n_filter_1, n_filter_2, 3, 1, 1), nn.LeakyReLU(0.2),
#             WN_Conv2d(n_filter_2, n_filter_2, 3, 1, 1), nn.LeakyReLU(0.2),
#             WN_Conv2d(n_filter_2, n_filter_2, 3, 2, 1), nn.LeakyReLU(0.2),

#             nn.Dropout2d(0.5) if args.dataset == 'svhn' else nn.Dropout(0.5),

#             WN_Conv2d(n_filter_2, n_filter_2, 3, 1, 0), nn.LeakyReLU(0.2),
#             WN_Conv2d(n_filter_2, n_filter_2, 1, 1, 0), nn.LeakyReLU(0.2),
#             WN_Conv2d(n_filter_2, n_filter_2, 1, 1, 0), nn.LeakyReLU(0.2),

#             Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
#         )

#         self.out_class = nn.Sequential(
#             WN_Linear(n_filter_2, self.num_label, train_scale=True, init_stdv=0.1)
#             )


#         # for p in self.modules():
#         #     if isinstance(p, nn.Linear) or isinstance(p, nn.Conv2d):
#         #         nn.init.kaiming_normal(p.weight.data)


    def forward(self, X, require='class'):
        if X.dim() == 2:
            X = X.view(X.size(0), self.n_channels, self.image_size, self.image_size)
        
        if require == 'class':
            return self.out_class(self.feat_net(X)).view(X.shape[0], -1)

        elif require == 'feat':
            return self.feat_net(X)

class Generator(myModel):
    def __init__(self, args):
        super(Generator, self).__init__()

        self.noise_size = args.noise_size
        self.image_size = args.image_size


        self.core_net = nn.Sequential(
            nn.Linear(self.noise_size, 4 * 4 * 512, bias=False), nn.BatchNorm1d(4 * 4 * 512), nn.ReLU(), 
            Expression(lambda tensor: tensor.view(tensor.size(0), 512, 4, 4)),
            nn.ConvTranspose2d(512, 256, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            # WN_ConvTranspose2d(128,   3, 5, 2, 2, 1, train_scale=True, init_stdv=0.1), nn.Tanh(),
            nn.ConvTranspose2d(128,   3, 5, 2, 2, 1), nn.Tanh(),
        )
        

        # for p in self.modules():
        #     if isinstance(p, nn.Linear) or isinstance(p, nn.Conv2d):
        #         nn.init.kaiming_normal(p.weight.data)

    def forward(self, noise):        
        output = self.core_net(noise)

        return output

class Discriminator(myModel):
    def __init__(self, args, feature_size):
        super(Discriminator, self).__init__()

        print('===> Init small-conv for {}'.format(args.dataset))

        self.noise_size = args.noise_size
        self.num_label  = 1
        self.n_channels = args.n_channels

        if args.dataset == 'svhn':
            n_filter_1, n_filter_2 = 64, 128
        elif args.dataset == 'cifar':
            n_filter_1, n_filter_2 = 96, 192
        else:
            raise ValueError('dataset not found: {}'.format(args.dataset))

        # Assume X is of size [batch x 3 x 32 x 32]
        self.feat_net = nn.Sequential(
            nn.Conv2d(args.n_channels, n_filter_1, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(n_filter_1     , n_filter_1, 3, 2, 1), nn.LeakyReLU(0.2),

            nn.Conv2d(n_filter_1     , n_filter_2, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(n_filter_2     , n_filter_2, 3, 2, 1), nn.LeakyReLU(0.2),

            nn.Conv2d(n_filter_2     , n_filter_2, 3, 1, 0), nn.LeakyReLU(0.2),
            nn.Conv2d(n_filter_2     , n_filter_2, 1, 1, 0), nn.LeakyReLU(0.2),

            Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
        )

        self.out_layer = nn.Linear(n_filter_2, self.num_label)

    def forward(self, X, require='class'):
        if X.dim() == 2:
            X = X.view(X.size(0), self.n_channels, self.image_size, self.image_size)
        
        if require == 'critic':
            return self.out_layer(self.feat_net(X))
        elif require == 'feat':
            return self.feat_net(X)

class feature_Generator(myModel):
    def __init__(self, args, input_size):
        super(feature_Generator, self).__init__()

        self.noise_size = args.noise_size
        self.image_size = args.image_size


        self.core_net = nn.Sequential(
            nn.Linear(     input_size,       1024), nn.BatchNorm1d(1024), nn.ReLU(), 
            nn.Linear(           1024,        512), nn.BatchNorm1d(512), nn.ReLU(), 
            nn.Linear(            512,        256), nn.BatchNorm1d(256), nn.ReLU(), 
            nn.Linear(            256, input_size),
            Expression(lambda tensor: tensor.view(tensor.size(0), -1, 1, 1))
        )
        

        # for p in self.modules():
        #     if isinstance(p, nn.Linear) or isinstance(p, nn.Conv2d):
        #         nn.init.kaiming_normal(p.weight.data)

    def forward(self, X):        
        if X.dim() == 4:
            X = X.view(X.size(0), -1)

        output = self.core_net(X)

        return output

class Classifier(myModel):
    def __init__(self, args, input_size):
        super(Classifier, self).__init__()

        self.input_size = input_size

        if args.spectral_norm:
            func = spectral_norm
        else:
            func = identity

        self.feat_net = nn.Sequential(
            func(nn.Linear(self.input_size, 400)), nn.ReLU(),
            func(nn.Linear(400, 200)), nn.ReLU(),
            func(nn.Linear(200, 100)), nn.ReLU(),
        )

        self.discriminator = nn.Sequential(
            func(nn.Linear(100, 1)),
            nn.Sigmoid()
            )

        # for p in self.discriminator.parameters():
        #     # print(p)
        #     # print(type(p))
        #     p.data = - torch.abs(p.data) * 100

            # print(p)

        # self._initialize_weights()

    def forward(self, X, require='critic'):
        if X.dim() == 4:
            X = X.view(X.size(0), -1)

        return self.discriminator(self.feat_net(X)).view(-1)



if __name__ == '__main__':
    m = VAE(128)

    i = torch.rand(28, 3, 32, 32)

    o, mean, std = m(i)
    print(o.shape, mean.shape, std.shape)
