import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

import lab5
from dataset import PetNoseDataset
from test import test_model


args = lab5.create_arguments()
args.add_argument('-no_test', action='store_true', help='skip testing after training')
args.add_argument('-b', type=int, default=40, help='batch size')
args.add_argument('-e', type=int, default=10, help='# of epochs')
args.add_argument('-lr', type=float, default=1e-3, help='learning rate')
args.add_argument('-plot_file', type=str, default=None, help='loss plot output file')
opts = args.parse_args()

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = lab5.create_model()
model.train()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = PetNoseDataset(root='./oxford-iiit-pet-noses', train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=opts.b, shuffle=True)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)

num_epochs = opts.e

losses = []
for epoch in range(num_epochs):
    loss_epoch = 0
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()

    losses.append(loss_epoch / len(train_loader))
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {losses[-1]}')

torch.save(model.state_dict(), 'model.pth')

if opts.plot_file is not None:
    plt.figure(2, figsize=(12, 7))
    plt.clf()
    plt.plot(losses, label='train')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc=1)
    plt.savefig(opts.plot_file)

if not opts.no_test:
    test_model(model, device, should_display=False)
