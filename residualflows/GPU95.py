import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time

def main():
    mode = 'test'
    model = models.resnet50()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    N = 1280
    dataset = datasets.FakeData(size=N, transform=transforms.ToTensor())
    if mode=='test': # switch to evaluate mode
        model.eval()
    model.to('cuda')
    for num_workers in [1, 2, 4, 8]: # 4 < 2 for test
        for batch_size in [1, 2, 4, 8, 16, 32]:
            loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True)
            if mode=='test':
                for i, (data, target) in enumerate(loader):
                    if i==1:
                        tm = time.time()
                    data = data.to('cuda', non_blocking=True)
                    output = model(data)
            else: # mode=='train':
                for i, (data, target) in enumerate(loader):
                    if i==1:
                        tm = time.time()
                    data = data.to('cuda', non_blocking=True)
                    target = target.to('cuda', non_blocking=True).long()
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            tm = time.time() - tm
            print('Mode=%s: NumWorkers=%2d  BatchSize=%2d  Time=%6.3fs  Imgs/s=%6.2f' % (mode, num_workers, batch_size, tm, N/tm))
            torch.cuda.empty_cache() # doesn't seem to be working...

if __name__ == '__main__':
    main()

