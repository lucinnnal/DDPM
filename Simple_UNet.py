# Convblock of UNet
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.Convs = nn.Sequential( # multiple layers at once!! -> 직렬적으로 연결하여 차례대로 통과
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size = 3, padding =1 ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.Convs(x)

# UNet
class UNet(nn.Module):
    def __init__(self, in_ch = 1):

        super().__init__()
        self.down1 = ConvBlock(in_ch, 64)
        self.down2 = ConvBlock(64, 128)
        self.bot1 = ConvBlock(128, 256)
        self.up2 = ConvBlock(128+256, 128)
        self.up1 = ConvBlock(128+64, 64)
        self.out = nn.Conv2d(64, in_ch, kernel_size = 1) # channel decresing convolution

        # MaxPool & Upsampling
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear') # size * 2 : 쌍성형보간법을 이용해 사진 크기 자체를 2배로 늘리기
    
    def forward(self, x):

        x1 = self.down1(x)
        x = self.maxpool(x1)
        x2 = self.down2(x)
        x = self.maxpool(x2)
        x = self.bot1(x)
        x = self.upsample(x)
        x = torch.cat([x2, x], dim = 1) # skip connection
        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x1, x], dim = 1) # skip connection
        x = self.up1(x)
        x = self.out(x)

        return x
    
# pos_encoding 




