import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm

class Diffuser(nn.Module):
    def __init__(self, num_timesteps = 1000, beta_start = 0.0001, beta_end = 0.02, device = 'cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device = device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim = 0)

    # Adding noise (원본 데이터와 샘플한 timestep을 넘겨 주기) -> Forward Process
    # N : Batch_size
    # x_0 shape : (N,C,H,W)
    # 각 x마다 다른 timestep이 추출됨 (N,)의 크기 
    def add_noise(self, x_0, t):
        # Check if all timesteps are valid !
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all() # Should be sampled one of 1~T (.all() method is to check all elements in the array)
        t_idx = t - 1
        alpha_bar = self.alpha_bars[t_idx]

        # Noise -> for reparameterize epsilon
        noise = torch.randn_like(x_0, device = self.device)

        # Noise Adding
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N,1,1,1)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise # x_t -> (N,C,H,W)

        return x_t, noise # Returns noise added pic and noise (0 -> t)  

    # Denoising (At certain timestep)
    def denoise(self, model, x, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        # x_t -> (N,C,H,W), t -> (N,)
        t_idx = t - 1
        alphas = self.alphas[t_idx] # (N,)
        alpha_bars = self.alpha_bars[t_idx] # (N,)
        alpha_bars_prev = self.alpha_bars[t_idx - 1] #(N,)
        
        # Dimension Expand
        N = alphas.size(0) # -> Batch_size
        alphas = alphas.view(N,1,1,1)
        alpha_bars = alpha_bars.view(N,1,1,1)
        alpha_bars_prev = alpha_bars_prev.view(N,1,1,1)

        # Model(UNet) output -> predicting noise that has been add from x0 to xt
        # Denoising module ouput -> x_t-1 from x_t (x_t-1 distribution's average is the output and using a reparameterize method to sample x_t-1)

        # Switch model to eval mode (UNet -> the neural net used in diffusion)
        model.eval()
        with torch.no_grad():
            eps = model(x, t)
        model.train()

        # Noise for Reparameterize method
        noise = torch.randn_like(x, device = self.device)
        noise[t == 1] = 0

        # t = 1에서 x_0을 뽑아낼 떄에는 noise가 0
        # Parameters for x_t-1 distribution 
        sigma = torch.sqrt((1 - alphas) * (1 - alpha_bars_prev) / (1 - alpha_bars))
        mu = (x - ((1 - alphas) / torch.sqrt(1 - alpha_bars)) * eps) / torch.sqrt(alphas)

        return mu + sigma * noise # (N,C,H,W)

    # Tensor to Image(PIL)
    def reverse_to_img(self, x):
        x = x * 255
        x = x.clamp(0,255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()

        return to_pil(x)
    
    # Sample -> Generating image from Random Gaussian Noise and the G.N input size will be (N,C,H,W)
    def sample(self, model, x_shape = (20, 1, 28, 28)):
        batch_size =  x_shape[0] # Default batch_size is 20
        # The noise in T timestep
        x = torch.randn(x_shape, device = self.device)
    
        T = self.num_timesteps

        # Sampling t-1 from T iteratively and finally get x_0s (batch)
        for i in tqdm(range(T, 0, -1)):
            t = torch.tensor([i] * batch_size, device = self.device, dtype = torch.long) # (N,)
            x = self.denoise(model, x, t) # (N,C,H,W)
        
        # Reverse from tensor to img
        imgs = [self.reverse_to_img(img) for img in x]

        return imgs