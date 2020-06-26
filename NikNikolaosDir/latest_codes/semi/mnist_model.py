import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn

from torch.nn.utils import spectral_norm

from torch import Tensor

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name, map_location=lambda storage, loc: storage))
    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

def identity(_input):
    return _input

class Discriminator(myModel):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.input_size = args.image_size * args.image_size

        if args.spectral_norm:
            func = spectral_norm
        else:
            func = identity

        self.feat_net = nn.Sequential(
            func(nn.Linear(self.input_size, 1000)), nn.ReLU(),
            func(nn.Linear(1000, 500)), nn.ReLU(),
            func(nn.Linear( 500, 250)), nn.ReLU(),
            func(nn.Linear( 250, 250)), nn.ReLU(),
            func(nn.Linear( 250, 250)), nn.ReLU(),
        )

        self.critic = nn.Sequential(
            func(nn.Linear(250, 1))
            )
        self.discriminator = nn.Sequential(
            func(nn.Linear(250, 1)),
            nn.Sigmoid()
            )
        # for p in self.modules():
        #     if isinstance(p, nn.Linear):
        #         nn.init.kaiming_normal(p.weight.data)
    def forward(self, X, require='critic'):
        if X.dim() == 4:
            X = X.view(X.size(0), -1)

        if require == 'critic':
            return self.critic(self.feat_net(X)).view(-1)
        elif require == 'discriminator':
            return self.discriminator(self.feat_net(X)).view(-1)
        elif require == 'feat':
            return self.feat_net(X)

class feature_Generator(myModel):
    def __init__(self, args, output_size):
        super(feature_Generator, self).__init__()

        self.noise_size = args.noise_size
        self.output_size = output_size

        dim = 256

        self.core_net = nn.Sequential(
            nn.Linear(self.noise_size, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, self.output_size), nn.ReLU()
            )

        # for p in self.modules():
        #     if isinstance(p, nn.Linear):
        #         nn.init.kaiming_normal(p.weight.data)

    def forward(self, noise):
        
        return self.core_net(noise)

class feature_Discriminator(myModel):
    def __init__(self, args, input_size):
        super(feature_Discriminator, self).__init__()

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

class feature_Discriminator_drop(myModel):
    def __init__(self, args, input_size):
        super(feature_Discriminator_drop, self).__init__()

        self.input_size = input_size

        if args.spectral_norm:
            func = spectral_norm
        else:
            func = identity

        self.feat_net = nn.Sequential(
            func(nn.Linear(self.input_size, 400)), nn.ReLU(),
            func(nn.Linear(400, 200)), nn.ReLU(),
            nn.Dropout(0.5),
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
        elif require == 'both':
            feat = self.feat_net(X)
            out = self.critic(feat).view(-1)

            return feat, out

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
    
    def forward(self, input):
        return self.func(input)

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

class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, b=True, g=True):
        super(BatchNorm1d, self).__init__()
        self.b = b
        self.g = g
        self.core = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=(b and g))

        if (not b) and g:
            self.g = Parameter(torch.Tensor(num_features))
        elif (not g) and b:
            self.b = Parameter(torch.Tensor(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        if (not self.b) and self.g:
            self.g.data.fill_(1)
        elif (not self.g) and self.b:
            self.b.data.zero_()

    def forward(self, input):
        output = self.core(input)
        if (not self.b) and self.g:
            output = output * self.g.expand_as(output)
        elif (not self.g) and self.b:
            output = output + self.b.expand_as(output)

        return output

class WN_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, train_scale=False):
        super(WN_Linear, self).__init__(in_features, out_features, bias=bias)
        if train_scale:
            self.weight_scale = Parameter(torch.ones(self.out_features))
        else:
            self.register_buffer('weight_scale', Variable(torch.ones(self.out_features)))
        self.init_mode = False

    def reset_parameters(self):
        self.weight.data.normal_(0, 0.1)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        # normalize weight matrix and linear projection
        norm_weight = self.weight * (self.weight_scale.unsqueeze(1) / \
            torch.sqrt((self.weight ** 2).sum(1, keepdim=True) + 1e-6)).expand_as(self.weight)
        activation = F.linear(input, norm_weight)

        if self.init_mode == True:
            mean_act = activation.mean(0).squeeze(0)
            activation = activation - mean_act.expand_as(activation)

            stdv_act = torch.sqrt((activation ** 2).mean(0) + 1e-6).squeeze(0)
            activation = activation / stdv_act.expand_as(activation)

            self.weight_scale.data = self.weight_scale.data / stdv_act.data
            self.bias.data = - mean_act.data / stdv_act.data

        else:
            if self.bias is not None:
                activation = activation + self.bias.expand_as(activation)

        return activation
            

class WN_Generator(myModel):
    def __init__(self, args):
        super(WN_Generator, self).__init__()

        self.noise_size = args.noise_size
        self.output_size = args.image_size * args.image_size

        self.core_net = nn.Sequential(
            nn.Linear(self.noise_size, 500, bias=False), 
            # nn.Linear(noise_size, 500, bias=True),
            nn.BatchNorm1d(500), 
            nn.Softplus(), 
            nn.Linear(500, 500, bias=False),   
            # nn.Linear(500, 500, bias=True), 
            nn.BatchNorm1d(500), 
            nn.Softplus(), 
            WN_Linear(500, self.output_size, train_scale=True),    
            nn.Sigmoid()
        )
        # self._initialize_weights()
        # for p in self.modules():
        #     if isinstance(p, nn.Linear):
        #         nn.init.kaiming_normal(p.weight.data)

    def forward(self, noise):
        output = self.core_net(noise)

        return output

class WN_Classifier(myModel):
    def __init__(self, args, feature_size):
        super(WN_Classifier, self).__init__()

        self.input_size = args.image_size*args.image_size
        self.num_label  = args.num_label


        self.feat_net = nn.Sequential(
            GaussianNoise(0.3), WN_Linear(self.input_size, 1000), nn.ReLU(),
            GaussianNoise(0.5), WN_Linear(1000, 500), nn.ReLU(),
            GaussianNoise(0.5), WN_Linear( 500, 250), nn.ReLU(),
            GaussianNoise(0.5), WN_Linear( 250, 250), nn.ReLU(),
            GaussianNoise(0.5), WN_Linear( 250, feature_size), nn.ReLU(),
        )


        self.out_class = nn.Sequential(
            GaussianNoise(0.5), 
            WN_Linear(feature_size, self.num_label, train_scale=True)
        )

        # self._initialize_weights()
        # for p in self.modules():
        #     if isinstance(p, nn.Linear):
        #         nn.init.kaiming_normal(p.weight.data)

    def forward(self, X, require='class'):
        if X.dim() == 4:
            X = X.view(X.size(0), -1)

        if require == 'class':
            return self.out_class(self.feat_net(X))
        elif require == 'feat':
            return self.feat_net(X)

if __name__ == '__main__':
    l = WN_Linear(3, 5)
    a = torch.rand(3)
    a_v = torch.autograd.Variable(a)
    l(a_v)
