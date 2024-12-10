import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# Define Transformer-based Generator
class TransformerGenerator(nn.Module):
    def _init_(self, img_size, patch_size, embed_dim, num_heads, condition_dim, num_layers):
        super(TransformerGenerator, self)._init_()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Linear(patch_size * patch_size * 3, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4, dropout=0.1),
            num_layers=num_layers,
        )
        self.condition_embed = nn.Linear(condition_dim, embed_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),  # Upsample to 32x32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Upsample to 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Upsample to 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Upsample to 256x256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),  # Final layer for RGB output
            nn.Tanh(),
        )

    def forward(self, img, conditions):
        batch_size = img.size(0)
        patches = img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, self.patch_size * self.patch_size * 3)
        patch_embeddings = self.patch_embed(patches)
        embeddings = patch_embeddings + self.positional_encoding[:, :patch_embeddings.size(1), :]
        condition_embeddings = self.condition_embed(conditions).unsqueeze(1)
        embeddings = torch.cat((condition_embeddings, embeddings), dim=1)
        transformer_output = self.transformer(embeddings)
        decoded = transformer_output[:, 1:].reshape(batch_size, self.img_size // self.patch_size, self.img_size // self.patch_size, -1)
        decoded = decoded.permute(0, 3, 1, 2)
        output = self.decoder(decoded)
        return output

# Parameters
img_size = 128
patch_size = 16
embed_dim = 256
num_heads = 8
condition_dim = 10
num_layers = 6

# Initialize Generator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = TransformerGenerator(img_size, patch_size, embed_dim, num_heads, condition_dim, num_layers).to(device)
generator.eval()  # Set to evaluation mode

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Load images using DataLoader
input_folder = "/home/mangesh_singh/testing_project/finalized"  # Replace with the path to your folder
dataset = datasets.ImageFolder(input_folder, transform=transform)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)  # Adjust batch_size as needed

# # Process the first batch of images
# first_batch = next(iter(data_loader))
# images, _ = first_batch  # Ignore labels, as they're not used
# images = images.to(device)
# # Generate fake images for the first batch
# conditions = torch.randn(images.size(0), condition_dim).to(device)  # Random conditions
# with torch.no_grad():
#     fake_images = generator(images, conditions)


class CNNDiscriminator(nn.Module):
    def _init_(self, img_size):
        super(CNNDiscriminator, self)._init_()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear((img_size // 8) * (img_size // 8) * 128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

criterion = nn.BCELoss()

discriminator = CNNDiscriminator(img_size).to(device)
generator.train() 
discriminator.train()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


batch_size=32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Adjust batch size as needed
num_epochs = 20
real_label = 1.0
fake_label = 0.0

print("number of batches:",len(data_loader))
def limited_data_loader(loader,max_batches):
    for i, data in enumerate(loader):
        if i >= max_batches:
            break
        yield data

# Directory to save models
save_dir = "./saved_models2"
os.makedirs(save_dir, exist_ok=True)

# Training loop with model saving
for epoch in range(num_epochs):
    g_loss_epoch = 0.0
    d_loss_epoch = 0.0

    # Use limited DataLoader
    for real_images, _ in limited_data_loader(data_loader, max_batches=280):
        # Move real images to device
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        ### Train Discriminator ###
        optimizer_D.zero_grad()

        # Real images
        real_labels = torch.full((batch_size, 1), real_label, device=device)
        output_real = discriminator(real_images)
        loss_real = criterion(output_real, real_labels)

        # Fake images
        conditions = torch.randn(batch_size, condition_dim, device=device)
        fake_images = generator(real_images, conditions)
        fake_labels = torch.full((batch_size, 1), fake_label, device=device)
        output_fake = discriminator(fake_images.detach())
        loss_fake = criterion(output_fake, fake_labels)

        # Total discriminator loss
        d_loss = loss_real + loss_fake
        d_loss.backward()
        optimizer_D.step()

        ### Train Generator ###
        optimizer_G.zero_grad()

        # Generate fake images again for generator loss
        output_fake_for_generator = discriminator(fake_images)
        g_loss = criterion(output_fake_for_generator, real_labels)  # Generator tries to fool discriminator
        g_loss.backward()
        optimizer_G.step()

        # Accumulate losses
        d_loss_epoch += d_loss.item()
        g_loss_epoch += g_loss.item()

    # Print losses for the epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}] | D Loss: {d_loss_epoch:.4f} | G Loss: {g_loss_epoch:.4f}")

    # Save models after each epoch
    torch.save(generator.state_dict(), os.path.join(save_dir, f"generator_epoch_{epoch+1}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, f"discriminator_epoch_{epoch+1}.pth"))
    print(f"Models saved for epoch {epoch + 1}")