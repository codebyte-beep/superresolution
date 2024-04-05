#!/usr/bin/env python
# coding: utf-8

# In[23]:


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

# Remove all the warnings
import warnings
warnings.filterwarnings('ignore')

# Set env CUDA_LAUNCH_BLOCKING=1
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Retina display
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

try:
    from einops import rearrange
except ImportError:
    get_ipython().run_line_magic('pip', 'install einops')
    from einops import rearrange


# In[24]:


if os.path.exists('dog.jpg'):
    print('dog.jpg exists')
else:
    get_ipython().system('wget https://segment-anything.com/assets/gallery/AdobeStock_94274587_welsh_corgi_pembroke_CD.jpg -O dog.jpg')


# In[25]:


# Read in a image from torchvision
img = torchvision.io.read_image("dog.jpg")
print(img.shape)


# In[26]:


from sklearn import preprocessing

scaler_img = preprocessing.MinMaxScaler().fit(img.reshape(-1, 1))
scaler_img


# In[27]:


img_scaled = scaler_img.transform(img.reshape(-1, 1)).reshape(img.shape)
img_scaled.shape

img_scaled = torch.tensor(img_scaled)


# In[28]:


img_scaled = img_scaled.to(device)
img_scaled


# In[29]:


crop = torchvision.transforms.functional.crop(img_scaled.cpu(), 600, 800, 300, 300)
crop.shape


# In[30]:


crop = crop.to(device)


# In[31]:


# Get the dimensions of the image tensor
num_channels, height, width = crop.shape
print(num_channels, height, width)


# In[32]:


def create_coordinate_map(img):
    """
    img: torch.Tensor of shape (num_channels, height, width)
    
    return: tuple of torch.Tensor of shape (height * width, 2) and torch.Tensor of shape (height * width, num_channels)
    """
    
    num_channels, height, width = img.shape
    
    # Create a 2D grid of (x,y) coordinates (h, w)
    # width values change faster than height values
    w_coords = torch.arange(width).repeat(height, 1)
    h_coords = torch.arange(height).repeat(width, 1).t()
    w_coords = w_coords.reshape(-1)
    h_coords = h_coords.reshape(-1)

    # Combine the x and y coordinates into a single tensor
    X = torch.stack([h_coords, w_coords], dim=1).float()

    # Move X to GPU if available
    X = X.to(device)

    # Reshape the image to (h * w, num_channels)
    Y = rearrange(img, 'c h w -> (h w) c').float()
    return X, Y


# In[33]:


class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return self.linear(x)


# In[34]:


net = LinearModel(2, 3)
net.to(device)


# In[35]:


def train(net, lr, X, Y, epochs, verbose=True):
    """
    net: torch.nn.Module
    lr: float
    X: torch.Tensor of shape (num_samples, 2)
    Y: torch.Tensor of shape (num_samples, 3)
    """

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = net(X)
        
        
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch} loss: {loss.item():.6f}")
    return loss.item()


# In[36]:


def plot_reconstructed_and_original_image(original_img, net, X, title=""):
    """
    net: torch.nn.Module
    X: torch.Tensor of shape (num_samples, 2)
    Y: torch.Tensor of shape (num_samples, 3)
    """
    num_channels, height, width = original_img.shape
    print(height, width, num_channels)
    net.eval()
    with torch.no_grad():
        outputs = net(X)
        print(outputs.shape)
        outputs = outputs.reshape(height, width, num_channels)
        #outputs = outputs.permute(1, 2, 0)
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax0.imshow(outputs.cpu())
    ax0.set_title("Reconstructed Image")
    

    ax1.imshow(original_img.cpu().permute(1, 2, 0))
    ax1.set_title("Original Image")
    
    for a in [ax0, ax1]:
        a.axis("off")


    fig.suptitle(title, y=0.9)
    plt.tight_layout()


# In[37]:


# create RFF features
def create_rff_features(X, num_features, sigma):
    from sklearn.kernel_approximation import RBFSampler
    rff = RBFSampler(n_components=num_features, gamma=1/(2 * sigma**2))
    X = X.cpu().numpy()
    X = rff.fit_transform(X)
    return torch.tensor(X, dtype=torch.float32).to(device)


# In[38]:


cropR = crop[0]
cropG = crop[1]
cropB = crop[2]


# In[39]:


# Mask the image with NaN values 
# for 10% data
# similarly we can do for 20% and 90% by just changing the value of prop
def mask_image(img, prop):
    img_copy = img.clone()
    mask = torch.rand(img.shape) < prop
    img_copy[mask] = float('nan')
    return img_copy, mask
masked_img = mask_image(cropR, 0.1)
cropR = masked_img[0]
print(masked_img[1])
cropG[masked_img[1]] = float('nan')
cropB[masked_img[1]] = float('nan')
combined_tensor = torch.stack((cropR, cropG, cropB), dim=0)
combined_tensor = combined_tensor.detach().float()
print(combined_tensor)
plt.imshow(rearrange(combined_tensor, 'c h w -> h w c').cpu().numpy())


# In[40]:


dog_X, dog_Y = create_coordinate_map(combined_tensor)
print(dog_X.shape)
print(dog_X)
print(dog_Y)
dog_X.dtype


# In[41]:


nan_indices = torch.isnan(dog_Y).any(dim=1)

# Filter dog_X and dog_Y based on nan_indices
filtered_dog_X = dog_X[~nan_indices]
filtered_dog_Y = dog_Y[~nan_indices]
# print(dog_Y[1][2])
# print(nan_indices)
# print(filtered_dog_X)
# print(filtered_dog_Y[:100, :])
# print(filtered_dog_Y.shape)
# print(filtered_dog_X)
filtered_dog_X.shape


# In[42]:


# MinMaxScaler from -1 to 1
scaler_X = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(filtered_dog_X.cpu())

# Scale the X coordinates
dog_X_scaled = scaler_X.transform(filtered_dog_X.cpu())

# Move the scaled X coordinates to the GPU
dog_X_scaled = torch.tensor(dog_X_scaled).to(device)

# Set to dtype float32
dog_X_scaled = dog_X_scaled.float()




scalerr_X = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(dog_X.cpu())

# Scale the X coordinates
dogg_X_scaled = scalerr_X.transform(dog_X.cpu())

# Move the scaled X coordinates to the GPU
dogg_X_scaled = torch.tensor(dogg_X_scaled).to(device)

# Set to dtype float32
dogg_X_scaled = dogg_X_scaled.float()


# In[43]:


X_rff = create_rff_features(dog_X_scaled, 3750, 0.008)
X_rff_normal = create_rff_features(dogg_X_scaled, 3750, 0.008)
print(X_rff)
print(X_rff.shape)
print(X_rff.shape[1])


# In[ ]:


net = LinearModel(X_rff.shape[1], 3)
net.to(device)

train(net, 0.01, X_rff, filtered_dog_Y, 2500)


# In[ ]:


print(X_rff)
plot_reconstructed_and_original_image(combined_tensor, net, X_rff_normal, title="Reconstructed Image with RFF Features")
print(net)

