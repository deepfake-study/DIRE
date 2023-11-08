import os
import glob
import torch
from torchvision.transforms import ToTensor
from PIL import Image  # Import the Image module from PIL
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

def calculate_snr(image1, image2):
    # Calculate the mean squared error (MSE)
    mse = np.mean((image1 - image2) ** 2)

    # Calculate the peak signal power (PSP)
    psp = np.max(image1) ** 2

    # Calculate the SNR
    snr = 10 * np.log10(psp / mse)

    return snr

# Define the image transformation pipeline
transform = transforms.Compose([
    # transforms.Resize((299, 299)),  # Resize the image to 299x299
    transforms.ToTensor()  # Convert the image to a PyTorch tensor
])

# Define the paths to the source folders
# source_folder1 = "/home/lorenzp/DeepFakeDetectors/DIRE/data/recons/train/imagenet/0_real"
# source_folder2 = "/home/lorenzp/DeepFakeDetectors/DIRE/data/recons/train/imagenet/1_pgd"

source_folder1 = "/home/lorenzp/DeepFakeDetectors/DIRE/data/recons/train/imagenet_4255/0_real"
source_folder2 = "/home/lorenzp/DeepFakeDetectors/DIRE/data/recons/train/imagenet_4255/1_pgd"

# Create custom dataset classes to load the images without labels
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_paths = glob.glob(os.path.join(folder_path, "*.png"))  # Get all image file paths
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_paths)

# Load the images from the first dataset using the custom dataset class
dataset1 = ImageDataset(source_folder1, transform=transform)

# Load the images from the second dataset using the custom dataset class
dataset2 = ImageDataset(source_folder2, transform=transform)

# Create data loaders for both datasets
dataloader1 = DataLoader(dataset1, batch_size=1, shuffle=False)
dataloader2 = DataLoader(dataset2, batch_size=1, shuffle=False)

# Initialize lists to store SSIM, SNR, and PSNR values
ssim_values = []
snr_values  = []
psnr_values = []

# Iterate through the images and calculate SSIM, SNR, and PSNR for each pair of images
for img1, img2 in zip(dataloader1, dataloader2):
    # Convert the images to NumPy arrays
    img1_np = img1.squeeze().permute(1, 2, 0).numpy()
    img2_np = img2.squeeze().permute(1, 2, 0).numpy()

    # Calculate SSIM
    ssim_values.append(ssim(img1_np, img2_np, multichannel=True))

    # Calculate SNR
    snr_values.append(calculate_snr(img1_np, img2_np))

    # Calculate PSNR
    psnr_values.append(psnr(img1_np, img2_np))

# Calculate the mean values for SSIM, SNR, and PSNR
mean_ssim = torch.mean(torch.tensor(ssim_values))
mean_snr = torch.mean(torch.tensor(snr_values))
mean_psnr = torch.mean(torch.tensor(psnr_values))

# Print the results
print("SSIM:", mean_ssim.item())
print("SNR:", mean_snr.item())
print("PSNR:", mean_psnr.item())
