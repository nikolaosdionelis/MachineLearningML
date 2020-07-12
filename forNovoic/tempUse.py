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

for i in range(1,d):
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
