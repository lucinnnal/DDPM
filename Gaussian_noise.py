import os
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

x = torch.randn(3, 64, 64) 
T = 1000 
betas = torch.linspace(0.00001,0.02, T) # noise scheduling that increases linearly

# Add noise for timesteps 1000
for t in range(T):
    beta = betas[t]
    # Create a gaussian noise that has a same shape with x
    eps = torch.randn_like(x)

    # Reparamterize
    x = x * torch.sqrt(1-beta) + eps * torch.sqrt(beta)

# Image load
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "flower.png")
image = plt.imread(file_path)
print(f"raw image shape: {image.shape}")

# Transforms
preprocess = transforms.Compose([
    transforms.ToTensor() # PIL or numby form image to Tensor(scaling to 0.0 ~ 1.0, dimension change from (H,W,C) to (C,H,W))
])
x = preprocess(image)
print(f"preprocessed image: {x.shape}")

# Duplicate Original Image Tensor
original_image = x.clone()

# To Original Image
def reverse_to_img(x):
    x = x * 255
    x = x.clamp(0, 255)
    x = x.to(torch.uint8)
    to_pil = transforms.ToPILImage()
    return to_pil(x)

# Diffusion (Adding noise) -> Show raw image per 100 step of Adding noise
# T개 Sampling
T = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, 1000)
imgs = [] # add images per 100 step

for t in range(T):
    if t % 100 == 0:
        img = reverse_to_img(x)
        imgs.append(img)
    
    beta = betas[t]
    eps = torch.rand_like(x)
    x = x * torch.sqrt(1 - beta) + eps * torch.sqrt(beta)

# Show 10 images (Added noise per 100 time step)
"""
plt.figure(figsize=(15,6))
for i,img in enumerate(imgs[:10]):
    plt.subplot(2,5,i+1)
    plt.imshow(img)
    plt.title(f"Noise: {i*100}")
    plt.axis('off')

plt.show()
"""

# Sample Just Last Timestep Noise from original Image
# torch.cumprod(x, dim=0) -> 0 dimension을 따라 누적곱을 수행
""""
T = 1000
beta_start = 0.001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, 1000)
"""
def add_noise(x_0, t, betas):
    T = len(betas)
    assert t >=1 and t <= T # check whether t is between 1~T (error occurs when t does not meet the condition)

    alphas = 1 - betas

    a_t = torch.cumprod(alphas[:t], dim = 0)[-1]
    eps = torch.randn_like(x_0)

    x_t = torch.sqrt(a_t) * x_0 + torch.sqrt(1 - a_t) * eps

    return x_t

# t = 100 sampling from x_0
t = 200
x_t = add_noise(original_image, t, betas)

# Reverse to IMG
img = reverse_to_img(x_t)
plt.imshow(img)
plt.title(f"Noise {t}")
plt.axis('off')
plt.show()