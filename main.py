import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm


class params:
    batch_size = 4096
    latent_dim = 4
    epochs = 25
    lr = 1e-4


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        ksp = [3, 2, 1]  # kernel size, stride, padding

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, *ksp),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, *ksp),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, *ksp),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, *ksp),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

        self.fc1a = nn.Linear(7 * 7 * 32, latent_dim)
        self.fc1b = nn.Linear(7 * 7 * 32, latent_dim)

        self.fc2 = nn.Linear(latent_dim, 7 * 7 * 32)

    def encode(self, x):
        conv_out = self.encoder_conv(x)
        return self.fc1a(conv_out), self.fc1b(conv_out)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):
        out = self.fc2(z)
        out = out.view(out.size(0), 32, 7, 7)
        return self.decoder_conv(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def criterion(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(
        recon_x.view(-1, 784), x.view(-1, 784), reduction="sum"
    )

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

kwargs = {"batch_size": params.batch_size}
if is_cuda:
    kwargs.update({"num_workers": 1, "pin_memory": True, "shuffle": True})

transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.MNIST("../../data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("../../data", train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, **kwargs)

metric_names = ["loss"]

model = VAE(params.latent_dim)
model.to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {num_params}")

optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

writer = SummaryWriter(log_dir="./logs")

temp_ipt = torch.rand(5, 1, 28, 28, device=device)
writer.add_graph(model, temp_ipt)

for epoch in range(params.epochs):
    model.train()

    train_metrics = [0]

    for idx, (data, _) in tqdm(
        enumerate(train_loader), f"Epoch: {epoch + 1}", total=len(train_loader)
    ):
        data = data.to(device)

        optimizer.zero_grad()
        output, mu, logvar = model(data)

        loss = criterion(output, data, mu, logvar)
        loss.backward()
        optimizer.step()

        train_metrics[0] += loss.item()

    train_metrics = np.array(train_metrics) / len(train_loader)
    for name, val in zip(metric_names, train_metrics):
        writer.add_scalar("train/" + name, val, epoch)

    test_metrics = [0]
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output, mu, logvar = model(data)

            test_metrics[0] += criterion(output, data, mu, logvar).item()

    test_metrics = np.array(test_metrics) / len(test_loader)
    for name, val in zip(metric_names, test_metrics):
        writer.add_scalar("test/" + name, val, epoch)
