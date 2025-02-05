# Library import
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import Diffuser Class and UNet class
from Diffuser import Diffuser
from UNet import UNet

# Hyperparameters
image_size = 28
batch_size = 128
num_timesteps = 1000
epochs = 10
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Show the images that the diffusion model made
def show_images(images, rows=2, cols=10): # total 20 images
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap='gray')
            plt.axis('off')
            i += 1
    plt.show()

# ETC settings including dataset
# Preprocessing
preprocess = transforms.Compose([
    transforms.ToTensor() # convert the image into tensor (normalize into 0~1)
])

# Dataset & Dataloader
dataset = torchvision.datasets.MNIST(
    root = './data',
    transform = preprocess,
    train = True,
    download = True
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)

# Diffuser & Model instance 
diffuser = Diffuser()
model = UNet()
model.to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# Training
losses = []
for epoch in range(epochs):
    total_loss = 0
    cnt = 0 # Iteration

    # Sample images for every epoch
    imgs = diffuser.sample(model)
    show_images(imgs)


    for imgs, labels in tqdm(dataloader):
        # zero_grad -ing all the gradients of the parameters of the model that has been calculated in the last iteration
        optimizer.zero_grad()
        # load imgs to device
        x = imgs.to(device)
        # Random sampling timesteps for each x s in batch
        ts = torch.randint(1, num_timesteps + 1, (len(imgs),), device = device) # -> (N,)

        # Forward
        x_t, noise = diffuser.add_noise(x, ts) # noise adding (from 0 to t)
        pred_noise = model(x_t, ts) # noise calculation with UNet
        loss = F.mse_loss(pred_noise, noise)

        # Backward
        loss.backward()

        # Parameter update
        optimizer.step()

        total_loss += loss.item()
        cnt += 1
    
    # Epoch avg loss
    avg_loss = total_loss / cnt
    losses.append(avg_loss)
    print(f"Epoch: {epoch}, Average loss: {avg_loss}")

# Loss Graph
plt.title("Epoch Loss")
plt.plot(losses)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

# Sampling After Training
imgs = diffuser.sample(model)
show_images(imgs)