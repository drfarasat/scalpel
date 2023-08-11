import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


img_dir = '/home/fmr/Downloads/scalpel/rescale/classes'
# Define the transformation to convert images to tensor
transform = transforms.Compose([transforms.ToTensor()])

# Load your dataset with the defined transformation
dataset = datasets.ImageFolder(root=img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

# Initialize lists to store the means and standard deviations for each channel
means = torch.zeros(3)
stds = torch.zeros(3)

# Iterate over all images in the dataset
for images, _ in dataloader:
    # For each channel, compute the mean and add it to the list
    for i in range(3):
        means[i] += images[:, i, :, :].mean()
        stds[i] += images[:, i, :, :].std()

# Divide by the number of images to get the mean and standard deviation for the entire dataset
means /= len(dataset)
stds /= len(dataset)

print("Means: ", means)
print("Stds: ", stds)
# result
#Means:  tensor([0.3250, 0.4593, 0.4189])
#Stds:  tensor([0.1759, 0.1573, 0.1695])