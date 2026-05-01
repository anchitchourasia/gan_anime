import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_size = 100
image_size = 64
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# ==========================================
# 2. Dataset Class (Optimized for Large Data)
# ==========================================
class AnimeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

# ==========================================
# 3. Model Architectures (DCGAN)
# ==========================================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x).view(-1)

# ==========================================
# 4. Main Execution Function
# ==========================================
def train():
    # Transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # Path to images (Updated for Kaggle standard)
    dataset_path = '/kaggle/input/datasets/splcher/animefacedataset/images'
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} not found. Please update path in script.")
        return

    dataset = AnimeDataset(dataset_path, transform=transform)
    train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Init Models
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training Loop
    epochs = 20
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(train_dl):
            real_images = real_images.to(device)
            b_size = real_images.size(0)

            # Train Discriminator
            netD.zero_grad()
            label = torch.full((b_size,), 1.0, device=device)
            output = netD(real_images)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, latent_size, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0.0)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            label.fill_(1.0)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

        print(f'Epoch [{epoch+1}/{epochs}] Loss_D: {errD_real+errD_fake:.4f} Loss_G: {errG:.4f}')
        
    # Save the final model
    torch.save(netG.state_dict(), 'generator_v1.pth')
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()