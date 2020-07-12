import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import IPython
#IPython.display.Image(filename='data/images/2012_004258.jpg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.figure()
#plt.imshow(mpimg.imread('data/images/2012_004258.jpg'))

#plt.imshow(mpimg.imread('data/images/2012_004258.jpg'))
#IPython.display.Image(filename='data/images/2012_004258.jpg')

#plt.imshow(mpimg.imread('data/images/2012_004258.jpg'))
#plt.imshow(mpimg.imread('data/images/2012_004258.jpg'))

#plt.imshow(mpimg.imread('data/images/2012_004258.jpg'))

#plt.imshow(mpimg.imread('data/images/2012_004258.jpg'))
#plt.imshow(mpimg.imread('data/images/2008_007739.jpg'))

#plt.imshow(mpimg.imread('data/images/2012_004258.jpg'))
#plt.imshow(mpimg.imread('data/images/2008_007739.jpg'))

#plt.imshow(mpimg.imread('data/images/2012_004258.jpg'))
#plt.imshow(mpimg.imread('../../home/ndioneli/data/images/2012_004258.jpg'))

#plt.imshow(mpimg.imread('../../home/ndioneli/data/images/2012_004258.jpg'))
#plt.imshow(mpimg.imread('../../home/ndioneli/data/images/2012_004258.jpg'))

#plt.imshow(mpimg.imread('../../home/ndioneli/data/images/2012_004258.jpg'))
#plt.imshow(mpimg.imread('../../home/ndioneli/data/images/2008_007739.jpg'))

#plt.imshow(mpimg.imread('../../home/ndioneli/data/images/2012_004258.jpg'))
#plt.imshow(mpimg.imread('../../home/ndioneli/data/images/2008_007739.jpg'))

#plt.imshow(mpimg.imread('../../home/ndioneli/data/images/2008_007739.jpg'))
#plt.show()

#plt.show()
#plt.imshow(mpimg.imread('data/images/2012_004258.jpg'))

#IPython.display.Image(filename='data/images/2012_004258.jpg')
#IPython.display.Image(filename='data/images/2008_007739.jpg')

#IPython.display.Image(filename='data/images/2008_007739.jpg')

#plt.figure()
#plt.imshow(mpimg.imread('data/images/2008_007739.jpg'))

#plt.imshow(mpimg.imread('data/images/2008_007739.jpg'))
#IPython.display.Image(filename='data/images/2008_007739.jpg')

#plt.show()
#IPython.display.Image(filename='data/images/2008_007739.jpg')

list_aeroplane = []
f = open("data/aeroplane.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_aeroplane.append(x[:11])

#print(list_aeroplane)
print(len(list_aeroplane))

list_bicycle = []
f = open("data/bicycle.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_bicycle.append(x[:11])

#print(list_bicycle)
print(len(list_bicycle))

#print(list_bicycle)
#print(list_aeroplane)

#print(list_bicycle)
#print(list_bicycle)

list_bird = []
f = open("data/bird.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_bird.append(x[:11])

#print(list_bird)
print(len(list_bird))

list_boat = []
f = open("data/boat.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_boat.append(x[:11])

#print(list_boat)
print(len(list_boat))

list_bottle = []
f = open("data/bottle.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_bottle.append(x[:11])

#print(list_bottle)
print(len(list_bottle))

list_bus = []
f = open("data/bus.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_bus.append(x[:11])

#print(list_bus)
print(len(list_bus))

list_car = []
f = open("data/car.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_car.append(x[:11])

#print(list_car)
print(len(list_car))

list_cat = []
f = open("data/cat.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_cat.append(x[:11])

#print(list_cat)
print(len(list_cat))

list_chair = []
f = open("data/chair.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_chair.append(x[:11])

#print(list_chair)
print(len(list_chair))

list_cow = []
f = open("data/cow.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_cow.append(x[:11])

#print(list_cow)
print(len(list_cow))

list_diningtable = []
f = open("data/diningtable.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_diningtable.append(x[:11])

#print(list_diningtable)
print(len(list_diningtable))

list_dog = []
f = open("data/dog.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_dog.append(x[:11])

#print(list_dog)
print(len(list_dog))

list_horse = []
f = open("data/horse.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_horse.append(x[:11])

#print(list_horse)
print(len(list_horse))

list_motorbike = []
f = open("data/motorbike.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_motorbike.append(x[:11])

#print(list_motorbike)
print(len(list_motorbike))

list_person = []
f = open("data/person.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_person.append(x[:11])

#print(list_person)
print(len(list_person))

list_pottedplant = []
f = open("data/pottedplant.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_pottedplant.append(x[:11])

#print(list_pottedplant)
print(len(list_pottedplant))

list_sheep = []
f = open("data/sheep.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_sheep.append(x[:11])

#print(list_sheep)
print(len(list_sheep))

list_sofa = []
f = open("data/sofa.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_sofa.append(x[:11])

#print(list_sofa)
print(len(list_sofa))

list_train = []
f = open("data/train.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_train.append(x[:11])

#print(list_train)
print(len(list_train))

list_tvmonitor = []
f = open("data/tvmonitor.txt","r")

if f.mode == 'r':
    f1 = f.readlines()

    for x in f1:
        if x[12] != '-' and x[12] != '0':
            #print(x)
            #print(x[:11])

            #print(x[:11])
            list_tvmonitor.append(x[:11])

#print(list_tvmonitor)
print(len(list_tvmonitor))

list_allTotal = [len(list_aeroplane), len(list_bicycle), len(list_bird), len(list_boat), \
                 len(list_bottle), len(list_bus), len(list_car), len(list_cat), len(list_chair), \
                 len(list_cow), len(list_diningtable), len(list_dog), len(list_horse), len(list_motorbike), \
                 len(list_person), len(list_pottedplant), len(list_sheep), len(list_sofa), len(list_train), len(list_tvmonitor)]

import numpy as np
#prob_list_allTotal = list_allTotal / np.sum(list_allTotal)

#prob_list_allTotal = list_allTotal / np.sum(list_allTotal)
prob_list_allTotal = list_allTotal / np.sum(list_allTotal)

names_list_allTotal = range(0,20)

#names_list_allTotal = ['aeroplane', 'bicycle', 'bird', 'boat', \
#                 'bottle', 'bus', 'car', 'cat', 'chair', \
#                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', \
#                 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

print('')

print(names_list_allTotal)
print(list_allTotal)

#plt.plot(names_list_allTotal, list_allTotal, '--bo')
#plt.show()

#plt.plot(names_list_allTotal, list_allTotal, '--bo')
#plt.plot(names_list_allTotal, prob_list_allTotal, '--bo')

#plt.plot(names_list_allTotal, prob_list_allTotal, '--bo')
#plt.plot(names_list_allTotal, list_allTotal, '--bo')

#plt.xticks(names_list_allTotal)
#plt.show()

#plt.xlabel('Class Index')
#plt.ylabel('Probability of Occurrence')

#plt.ylabel('Probability of Occurrence')
#plt.ylabel('(Raw) Number of Occurrence')

#plt.show()
#import torch

#from PIL import Image
#im = Image.open('um_000000.png')

#from PIL import Image
#from PIL import Image

#im = Image.open('um_000000.png')
#im = Image.open('../../home/ndioneli/data/images/2012_004258.jpg')

#im = Image.open('../../home/ndioneli/data/images/2012_004258.jpg')
#im = mpimg.imread('../../home/ndioneli/data/images/2012_004258.jpg')

#im = mpimg.imread('../../home/ndioneli/data/images/2012_004258.jpg')
#im = mpimg.imread('../../home/ndioneli/data/images/2008_007739.jpg')

#im = mpimg.imread('../../home/ndioneli/data/images/2008_007739.jpg')

#im = mpimg.imread('../../home/ndioneli/data/images/2008_007739.jpg')
im = mpimg.imread('./data/images/2008_007739.jpg')

#plt.imshow(mpimg.imread('../../home/ndioneli/data/images/2012_004258.jpg'))
#plt.imshow(mpimg.imread('../../home/ndioneli/data/images/2008_007739.jpg'))

dtData = np.asarray(im)

#print(dtData)
#print(dtData.shape)

#   [114  79  57]
#   [112  78  53]
#   [109  77  52]]]
# (333, 500, 3)

print('')

import PIL.Image as Image
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F

#img = Image.open('data/Places365_val_00000001.jpg')
#img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')

#img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
#img = Image.open('../../home/ndioneli/data/images/2012_004258.jpg')

#img = Image.open('../../home/ndioneli/data/images/2012_004258.jpg')
#img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')

#img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')

#img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('./data/images/2008_007739.jpg')

import torch
img = ToTensor()(img)

#print(img.shape)
#print(img.size)

#print(img)
#print(img.shape)

#img = ToTensor()(img)
#print(img.shape)

#img = img.view(-1, img.shape[1] * img.shape[2])
#print(img.shape)

#img = ToTensor()(img)
#out = F.interpolate(img, size=256)

# The resize operation on tensor
#out = F.interpolate(img, size=256)

#a = torch.rand(1,2,3,4)
#print(a.size())

#print('')
#print(a.size())

#print(a.transpose(0,3).transpose(1,2).size())
#print(a.permute(3,2,1,0).size())

#print(img.size())
#print(img.permute(0,2,1).size())

#img = img.permute(0,2,1)
#print(img.size())

out = F.interpolate(img, size=256)
#out = F.interpolate(img, size=([256, 256]))

#ToPILImage()(out).save('test.png', mode='png')
#ToPILImage()(out).save('test3.png', mode='png')

#print(out)
#print(out.shape)

#img = img.permute(0,2,1)
#print(img.size())

out = out.permute(0,2,1)
#print(out.size())

out = F.interpolate(out, size=256)
#print(out.shape)

out = out.permute(0,2,1)
#ToPILImage()(out).save('teTest3.png', mode='png')

#ToPILImage()(out).save('teTest3.png', mode='png')
#ToPILImage()(out).save('teTest4.png', mode='png')

#print(out.size())
#asdfasdfasdf

print(out.size())
#asdffqwe

#plt.imshow(mpimg.imread('../../home/ndioneli/data/images/2012_004258.jpg'))
#plt.imshow(mpimg.imread('../../home/ndioneli/data/images/2008_007739.jpg'))

m = 1
#n = 3*256*256

# Like CIFAR-10
n = 3*32*32

#d = np.min(list_allTotal)
#d = np.sum(list_allTotal)

d = np.sum(list_allTotal)
#d = np.sum(list_allTotal[:6])

print(d)
#asdfasdf

# X is nxd
# and Y is mxd

print('')

#x = x.unsqueeze(1).expand(n, m, d)
#y = y.unsqueeze(0).expand(n, m, d)

#toUse_list_aeroplane = []
#img = Image.open('../../home/ndioneli/data/images/' + list_aeroplane[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
#img = Image.open('../../home/ndioneli/data/images/' + list_aeroplane[0] + '.jpg')

#img = Image.open('../../home/ndioneli/data/images/' + list_aeroplane[0] + '.jpg')

#img = Image.open('../../home/ndioneli/data/images/' + list_aeroplane[0] + '.jpg')
img = Image.open('./data/images/' + list_aeroplane[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_aeroplane.append(arrayOfImage)
toUse_list_aeroplane = arrayOfImage

for i in range(1,list_allTotal[0]):
    #print(list_aeroplane[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    #img = Image.open('../../home/ndioneli/data/images/'+list_aeroplane[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/'+list_aeroplane[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/'+list_aeroplane[i]+'.jpg')
    img = Image.open('./data/images/' + list_aeroplane[i] + '.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_aeroplane.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_aeroplane.append(arrayOfImage)
    toUse_list_aeroplane = torch.cat((toUse_list_aeroplane, arrayOfImage), 0)

print(toUse_list_aeroplane.shape)



#toUse_list_bicycle = []
#img = Image.open('../../home/ndioneli/data/images/' + list_bicycle[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
#img = Image.open('../../home/ndioneli/data/images/' + list_bicycle[0] + '.jpg')

#img = Image.open('../../home/ndioneli/data/images/' + list_bicycle[0] + '.jpg')

#img = Image.open('../../home/ndioneli/data/images/' + list_bicycle[0] + '.jpg')
img = Image.open('./data/images/' + list_bicycle[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_bicycle.append(arrayOfImage)
toUse_list_bicycle = arrayOfImage

for i in range(1,list_allTotal[1]):
    #print(list_bicycle[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    #img = Image.open('../../home/ndioneli/data/images/'+list_bicycle[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/'+list_bicycle[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/'+list_bicycle[i]+'.jpg')
    img = Image.open('./data/images/' + list_bicycle[i] + '.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_bicycle.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_bicycle.append(arrayOfImage)
    toUse_list_bicycle = torch.cat((toUse_list_bicycle, arrayOfImage), 0)

print(toUse_list_bicycle.shape)



#toUse_list_bird = []
#img = Image.open('../../home/ndioneli/data/images/' + list_bird[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_bird[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_bird.append(arrayOfImage)
toUse_list_bird = arrayOfImage

for i in range(1,list_allTotal[2]):
    #print(list_bird[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_bird[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_bird.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_bird.append(arrayOfImage)
    toUse_list_bird = torch.cat((toUse_list_bird, arrayOfImage), 0)

print(toUse_list_bird.shape)



#toUse_list_boat = []
#img = Image.open('../../home/ndioneli/data/images/' + list_boat[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_boat[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_boat.append(arrayOfImage)
toUse_list_boat = arrayOfImage

for i in range(1,list_allTotal[3]):
    #print(list_boat[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_boat[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_boat.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_boat.append(arrayOfImage)
    toUse_list_boat = torch.cat((toUse_list_boat, arrayOfImage), 0)

print(toUse_list_boat.shape)



#toUse_list_bottle = []
#img = Image.open('../../home/ndioneli/data/images/' + list_bottle[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_bottle[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_bottle.append(arrayOfImage)
toUse_list_bottle = arrayOfImage

for i in range(1,list_allTotal[4]):
    #print(list_bottle[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_bottle[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_bottle.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_bottle.append(arrayOfImage)
    toUse_list_bottle = torch.cat((toUse_list_bottle, arrayOfImage), 0)

print(toUse_list_bottle.shape)



#toUse_list_bus = []
#img = Image.open('../../home/ndioneli/data/images/' + list_bus[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_bus[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_bus.append(arrayOfImage)
toUse_list_bus = arrayOfImage

for i in range(1,list_allTotal[5]):
    #print(list_bus[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_bus[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_bus.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_bus.append(arrayOfImage)
    toUse_list_bus = torch.cat((toUse_list_bus, arrayOfImage), 0)

print(toUse_list_bus.shape)



#toUse_list_car = []
#img = Image.open('../../home/ndioneli/data/images/' + list_car[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_car[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_car.append(arrayOfImage)
toUse_list_car = arrayOfImage

for i in range(1,list_allTotal[6]):
    #print(list_car[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_car[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_car.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_car.append(arrayOfImage)
    toUse_list_car = torch.cat((toUse_list_car, arrayOfImage), 0)

print(toUse_list_car.shape)



#toUse_list_cat = []
#img = Image.open('../../home/ndioneli/data/images/' + list_cat[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_cat[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_cat.append(arrayOfImage)
toUse_list_cat = arrayOfImage

for i in range(1,list_allTotal[7]):
    #print(list_cat[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_cat[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_cat.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_cat.append(arrayOfImage)
    toUse_list_cat = torch.cat((toUse_list_cat, arrayOfImage), 0)

print(toUse_list_cat.shape)



#toUse_list_chair = []
#img = Image.open('../../home/ndioneli/data/images/' + list_chair[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_chair[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_chair.append(arrayOfImage)
toUse_list_chair = arrayOfImage

for i in range(1,list_allTotal[8]):
    #print(list_chair[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_chair[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_chair.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_chair.append(arrayOfImage)
    toUse_list_chair = torch.cat((toUse_list_chair, arrayOfImage), 0)

print(toUse_list_chair.shape)



#toUse_list_cow = []
#img = Image.open('../../home/ndioneli/data/images/' + list_cow[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_cow[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_cow.append(arrayOfImage)
toUse_list_cow = arrayOfImage

for i in range(1,list_allTotal[9]):
    #print(list_cow[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_cow[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_cow.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_cow.append(arrayOfImage)
    toUse_list_cow = torch.cat((toUse_list_cow, arrayOfImage), 0)

print(toUse_list_cow.shape)



#toUse_list_diningtable = []
#img = Image.open('../../home/ndioneli/data/images/' + list_diningtable[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_diningtable[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_diningtable.append(arrayOfImage)
toUse_list_diningtable = arrayOfImage

for i in range(1,list_allTotal[10]):
    #print(list_diningtable[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_diningtable[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_diningtable.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_diningtable.append(arrayOfImage)
    toUse_list_diningtable = torch.cat((toUse_list_diningtable, arrayOfImage), 0)

print(toUse_list_diningtable.shape)



#toUse_list_dog = []
#img = Image.open('../../home/ndioneli/data/images/' + list_dog[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_dog[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_dog.append(arrayOfImage)
toUse_list_dog = arrayOfImage

for i in range(1,list_allTotal[11]):
    #print(list_dog[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_dog[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_dog.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_dog.append(arrayOfImage)
    toUse_list_dog = torch.cat((toUse_list_dog, arrayOfImage), 0)

print(toUse_list_dog.shape)



#toUse_list_horse = []
#img = Image.open('../../home/ndioneli/data/images/' + list_horse[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_horse[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_horse.append(arrayOfImage)
toUse_list_horse = arrayOfImage

for i in range(1,list_allTotal[12]):
    #print(list_horse[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_horse[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_horse.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_horse.append(arrayOfImage)
    toUse_list_horse = torch.cat((toUse_list_horse, arrayOfImage), 0)

print(toUse_list_horse.shape)



#toUse_list_motorbike = []
#img = Image.open('../../home/ndioneli/data/images/' + list_motorbike[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_motorbike[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_motorbike.append(arrayOfImage)
toUse_list_motorbike = arrayOfImage

for i in range(1,list_allTotal[13]):
    #print(list_motorbike[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_motorbike[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_motorbike.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_motorbike.append(arrayOfImage)
    toUse_list_motorbike = torch.cat((toUse_list_motorbike, arrayOfImage), 0)

print(toUse_list_motorbike.shape)



#toUse_list_person = []
#img = Image.open('../../home/ndioneli/data/images/' + list_person[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_person[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_person.append(arrayOfImage)
toUse_list_person = arrayOfImage

for i in range(1,list_allTotal[14]):
    #print(list_person[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_person[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_person.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_person.append(arrayOfImage)
    toUse_list_person = torch.cat((toUse_list_person, arrayOfImage), 0)

print(toUse_list_person.shape)



#toUse_list_pottedplant = []
#img = Image.open('../../home/ndioneli/data/images/' + list_pottedplant[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_pottedplant[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_pottedplant.append(arrayOfImage)
toUse_list_pottedplant = arrayOfImage

for i in range(1,list_allTotal[15]):
    #print(list_pottedplant[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_pottedplant[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_pottedplant.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_pottedplant.append(arrayOfImage)
    toUse_list_pottedplant = torch.cat((toUse_list_pottedplant, arrayOfImage), 0)

print(toUse_list_pottedplant.shape)



#toUse_list_sheep = []
#img = Image.open('../../home/ndioneli/data/images/' + list_sheep[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_sheep[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_sheep.append(arrayOfImage)
toUse_list_sheep = arrayOfImage

for i in range(1,list_allTotal[16]):
    #print(list_sheep[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_sheep[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_sheep.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_sheep.append(arrayOfImage)
    toUse_list_sheep = torch.cat((toUse_list_sheep, arrayOfImage), 0)

print(toUse_list_sheep.shape)



#toUse_list_sofa = []
#img = Image.open('../../home/ndioneli/data/images/' + list_sofa[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_sofa[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_sofa.append(arrayOfImage)
toUse_list_sofa = arrayOfImage

for i in range(1,list_allTotal[17]):
    #print(list_sofa[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_sofa[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_sofa.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_sofa.append(arrayOfImage)
    toUse_list_sofa = torch.cat((toUse_list_sofa, arrayOfImage), 0)

print(toUse_list_sofa.shape)



#toUse_list_train = []
#img = Image.open('../../home/ndioneli/data/images/' + list_train[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_train[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_train.append(arrayOfImage)
toUse_list_train = arrayOfImage

for i in range(1,list_allTotal[18]):
    #print(list_train[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_train[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_train.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_train.append(arrayOfImage)
    toUse_list_train = torch.cat((toUse_list_train, arrayOfImage), 0)

print(toUse_list_train.shape)



#toUse_list_tvmonitor = []
#img = Image.open('../../home/ndioneli/data/images/' + list_tvmonitor[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_tvmonitor[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_tvmonitor.append(arrayOfImage)
toUse_list_tvmonitor = arrayOfImage

for i in range(1,list_allTotal[19]):
    #print(list_tvmonitor[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_tvmonitor[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_tvmonitor.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_tvmonitor.append(arrayOfImage)
    toUse_list_tvmonitor = torch.cat((toUse_list_tvmonitor, arrayOfImage), 0)

print(toUse_list_tvmonitor.shape)



# leave-one-out
# use leave-one-out

# leave-out class
# the leave-out class

# load to GPU
# use the GPU

# load the variables to the GPU
# GPU - Use the GPU. - Load to GPU.



'''
x = torch.cat((toUse_list_aeroplane, toUse_list_bicycle, toUse_list_bird,
               toUse_list_boat, toUse_list_bottle, toUse_list_bus), 0)

# X is nxd
# and Y is mxd

y = torch.cat((torch.zeros(toUse_list_aeroplane.size(0),1),
               torch.ones(toUse_list_bicycle.size(0),1),
               2*torch.ones(toUse_list_bird.size(0),1),
               3*torch.ones(toUse_list_boat.size(0),1),
               4*torch.ones(toUse_list_bottle.size(0),1),
               5*torch.ones(toUse_list_bus.size(0),1)), 0)

#y = torch.cat((torch.zeros(toUse_list_aeroplane.size()),
#               torch.ones(toUse_list_bicycle.size()),
#               2*torch.ones(toUse_list_bird.size()),
#               3*torch.ones(toUse_list_boat.size()),
#               4*torch.ones(toUse_list_bottle.size()),
#               5*torch.ones(toUse_list_bus.size())), 0)

x = x.permute(1, 0)
y = y.permute(1, 0)

print('')
print(x.shape)

#print(x.shape)
print(y.shape)

#print(x.shape)
#asdfasdfas

x = x.unsqueeze(1).expand(n, m, d)
y = y.unsqueeze(0).expand(n, m, d)

dist = torch.pow(x - y, 2).sum(2)

print(dist)
print(dist.shape)
'''



"""
#toUse_list_aeroplane = []
#img = Image.open('../../home/ndioneli/data/images/' + list_aeroplane[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = Image.open('../../home/ndioneli/data/images/' + list_aeroplane[0] + '.jpg')

# img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=32)
out = out.permute(0, 2, 1)

out = F.interpolate(out, size=32)
out = out.permute(0, 2, 1)

# print(out.size())
# ToPILImage()(out).save('teTest5.png', mode='png')

# print(out.size())
# asdffqwasdzf

# arrayOfImage = out.view(n,1)
#arrayOfImage = out.reshape(n)

#arrayOfImage = out.reshape(n)
arrayOfImage = out.reshape(1,n)

# print(arrayOfImage.size())
# asdfasdfz

#toUse_list_aeroplane.append(arrayOfImage)
toUse_list_aeroplane = arrayOfImage

for i in range(1,d):
    #print(list_aeroplane[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_aeroplane[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=32)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=32)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    #arrayOfImage = out.reshape(n)

    #arrayOfImage = out.reshape(n)
    arrayOfImage = out.reshape(1,n)

    #print(arrayOfImage.size())
    #asdfasdfz

    #print(toUse_list_aeroplane.shape)
    #print(arrayOfImage.shape)

    #asdfasdfas

    #toUse_list_aeroplane.append(arrayOfImage)
    toUse_list_aeroplane = torch.cat((toUse_list_aeroplane, arrayOfImage), 0)

#print(toUse_list_aeroplane)
#print(len(toUse_list_aeroplane))

#print(toUse_list_aeroplane.shape)
#print(len(toUse_list_aeroplane))

#print(toUse_list_aeroplane.shape)
#asdfsdfasfzdsf

#print(toUse_list_aeroplane.shape)
print(toUse_list_aeroplane.shape)
"""

'''
toUse_list_aeroplane = []
for i in range(d):
    #print(list_aeroplane[i])
    #asdfasdf

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = Image.open('../../home/ndioneli/data/images/'+list_aeroplane[i]+'.jpg')

    #img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
    img = ToTensor()(img)

    out = F.interpolate(img, size=256)
    out = out.permute(0, 2, 1)

    out = F.interpolate(out, size=256)
    out = out.permute(0, 2, 1)

    #print(out.size())
    #ToPILImage()(out).save('teTest5.png', mode='png')

    #print(out.size())
    #asdffqwasdzf

    #arrayOfImage = out.view(n,1)
    arrayOfImage = out.reshape(n)

    print(arrayOfImage.size())
    asdfasdfz

    toUse_list_aeroplane.append(arrayOfImage)
'''

"""
list_allTotal = [len(list_aeroplane), len(list_bicycle), len(list_bird), len(list_boat), \
                 len(list_bottle), len(list_bus), len(list_car), len(list_cat), len(list_chair), \
                 len(list_cow), len(list_diningtable), len(list_dog), len(list_horse), len(list_motorbike), \
                 len(list_person), len(list_pottedplant), len(list_sheep), len(list_sofa), len(list_train), len(list_tvmonitor)]

import numpy as np
#prob_list_allTotal = list_allTotal / np.sum(list_allTotal)

#prob_list_allTotal = list_allTotal / np.sum(list_allTotal)
prob_list_allTotal = list_allTotal / np.sum(list_allTotal)

names_list_allTotal = range(0,20)

#names_list_allTotal = ['aeroplane', 'bicycle', 'bird', 'boat', \
#                 'bottle', 'bus', 'car', 'cat', 'chair', \
#                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', \
#                 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

print(names_list_allTotal)
print(list_allTotal)
"""

'''
img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')
img = ToTensor()(img)

out = F.interpolate(img, size=256)
out = out.permute(0,2,1)

out = F.interpolate(out, size=256)
out = out.permute(0,2,1)

print(out.size())
#asdffqwasdf
'''

'''
img = Image.open('../../home/ndioneli/data/images/2008_007739.jpg')

import torch
img = ToTensor()(img)

#print(img.shape)
#print(img.size)

#print(img)
#print(img.shape)

#img = ToTensor()(img)
#print(img.shape)

#img = img.view(-1, img.shape[1] * img.shape[2])
#print(img.shape)

#img = ToTensor()(img)
#out = F.interpolate(img, size=256)

# The resize operation on tensor
#out = F.interpolate(img, size=256)

#a = torch.rand(1,2,3,4)
#print(a.size())

#print('')
#print(a.size())

#print(a.transpose(0,3).transpose(1,2).size())
#print(a.permute(3,2,1,0).size())

#print(img.size())
#print(img.permute(0,2,1).size())

#img = img.permute(0,2,1)
#print(img.size())

out = F.interpolate(img, size=256)
#out = F.interpolate(img, size=([256, 256]))

#ToPILImage()(out).save('test.png', mode='png')
#ToPILImage()(out).save('test3.png', mode='png')

#print(out)
#print(out.shape)

#img = img.permute(0,2,1)
#print(img.size())

out = out.permute(0,2,1)
#print(out.size())

out = F.interpolate(out, size=256)
#print(out.shape)

out = out.permute(0,2,1)
#ToPILImage()(out).save('teTest3.png', mode='png')

#ToPILImage()(out).save('teTest3.png', mode='png')
#ToPILImage()(out).save('teTest4.png', mode='png')

#print(out.size())
#asdfasdfasdf

print(out.size())
#asdffqwe

#plt.imshow(mpimg.imread('../../home/ndioneli/data/images/2012_004258.jpg'))
#plt.imshow(mpimg.imread('../../home/ndioneli/data/images/2008_007739.jpg'))
'''

