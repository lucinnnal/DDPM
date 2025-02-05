
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