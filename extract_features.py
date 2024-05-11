import sys

import pandas as pd
import shutil
import os
from PIL import Image 
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights

# Define target image size and batch size
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 64

# Convert resized images to tensors and normalize pixel values
data_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def extract_features(model, dataloader):
    features = []
    labels = []
    # Use tqdm to create a progress bar for the loop
    for images, targets in tqdm(dataloader, desc='Extracting features', unit='batch'):
        with torch.no_grad():
            extracted_features = model(images)
            features.append(extracted_features)
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)
 
# Load training and test datasets (no augmentation)
train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms)
test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Import VGG16 model with pre-trained weights
weights = VGG16_Weights.IMAGENET1K_V1
vgg = vgg16(weights=weights) # Initialize VGG16 model with pre-trained weights
vgg.eval() # disables features like dropout and batch normalization updates
vgg.trainable = False # Freeze the parameters of the VGG16 model

# Extract features from training data
train_features, train_labels = extract_features(vgg, train_loader)

# Extract features from test data
test_features, test_labels = extract_features(vgg, test_loader)

# Save extracted training features and labels
torch.save(train_features, 'features/train_features.pth')
torch.save(train_labels, 'features/train_labels.pth')
torch.save(test_features, 'features/test_features.pth')
torch.save(test_labels, 'features/test_labels.pth')