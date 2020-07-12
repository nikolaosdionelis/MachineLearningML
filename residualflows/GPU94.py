
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

dataset = datasets.FakeData(
    size=1000,
    transform=transforms.ToTensor())
loader = DataLoader(
    dataset,
    num_workers=1,
    pin_memory=True
)

model.to('cuda')

for data, target in loader:
    data = data.to('cuda', non_blocking=True)
    target = target.to('cuda', non_blocking=True)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

print('Done')

