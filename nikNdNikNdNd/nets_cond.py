import torch.nn as nn
import torch.nn.functional as F
import torch
    
class Generator(nn.Module):
    def __init__(self, imgSize, nz, ngf, nc, num_classes=10):
        super(Generator, self).__init__()

        self.ngf = ngf
        self.nz = nz
        self.fc_labels = nn.Linear(num_classes, 1000) # 1000
        self.fc_combined = nn.Linear(nz + num_classes, (ngf*8) * 4 * 4) # nz + 1000
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        
        self.main = nn.Sequential(
            # # input is Z, going into a convolution
            # nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False), # in_channels, out_channels, kernel_size, stride=1, padding=0,
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),

            # input is reshaped fc(Z+Y), state (nfg*16) * 2 * 2 ?
            # nn.ConvTranspose2d( ngf*16 , ngf * 8, 4, 1, 0, bias=False), # in_channels, out_channels, kernel_size, stride=1, padding=0,
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,    ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, y): # input: (64, 100, 1, 1) -- (bsz, nz, 1, 1)
        #import pdb; pdb.set_trace()

        bsz = input.size(0)
        # transorm y: 10 -> 1000
        # y_ = self.fc_labels(y) # y should be (64)
        # y_ = F.relu(y_)
        y_ = y
        # concat x and y and transform: 1100 -> ngf*8*4*4
        x = torch.cat([input.squeeze(), y_], 1)
        x = self.fc_combined(x) # [64, 1100] -> [64, 8192]
        x = x.view(bsz, self.ngf*8, 4, 4) # -> [64, 512, 4, 4]

        x = self.bn1(x)
        x = F.relu(x)
        
        # pass through deconv
        output = self.main(x)
        return output


class Discriminator(nn.Module):
    def __init__(self, imgSize, ndf, nc, num_classes=10):
        super(Discriminator, self).__init__()

        self.ndf = ndf
        self.nc = nc
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid() # [64, 1, 1, 1]
        )
        # 64*16*16 + 800 -> 1024
        self.fc_labels = nn.Linear(num_classes, 1000)
        # self.fc1  = nn.Linear((ndf*8) * 4 * 4 + 1000, 1024) 4
        # self.fc2 = nn.Linear(1024, 1)
        self.fc1  = nn.Linear((ndf*8) * 4 * 4 + num_classes, 128)
        self.fc2 = nn.Linear(128, 1)

        # self.fc_combined = nn.Linear(nz + 1000, (ngf*16) * 2 * 2)
        # self.bc1 = nn.BatchNorm2d(ngf * 16)
        

    def forward(self, input, y): # input: torch.Size([64, 1, 64, 64])
        bsz = input.size(0)
        x = self.main(input)

        x = x.view(bsz, (self.ndf*8) * 4 * 4)
        # y_ = self.fc_labels(y)
        # y_ = F.relu(y_)
        y_ = y
        x = torch.cat([x, y_], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze()

