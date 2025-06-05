# Step 7: Define the Pix2Pix Model (Using the Provided Code)
import torch
import torch.nn as nn  # For defining neural networks
import torch.nn.functional as F  # For activation functions (if needed)

class DownsamplingBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=4, stride=2, padding=1, negative_slope=0.2, use_norm=True):
        super(DownsamplingBlock, self).__init__()
        block = [nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=not use_norm)]
        if use_norm:
            block.append(nn.BatchNorm2d(c_out))
        block.append(nn.LeakyReLU(negative_slope))
        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return self.conv_block(x)

class UpsamplingBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=4, stride=2, padding=1, use_dropout=False, use_upsampling=False, mode='nearest'):
        super(UpsamplingBlock, self).__init__()
        block = []
        if use_upsampling:
            block.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode),
                nn.Conv2d(c_in, c_out, 3, 1, padding, bias=False)
            ))
        else:
            block.append(nn.ConvTranspose2d(c_in, c_out, kernel_size, stride, padding, bias=False))
        block.append(nn.BatchNorm2d(c_out))
        if use_dropout:
            block.append(nn.Dropout(0.5))
        block.append(nn.ReLU())
        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return self.conv_block(x)

class UnetEncoder(nn.Module):
    def __init__(self, c_in=3, c_out=512):
        super(UnetEncoder, self).__init__()
        self.enc1 = DownsamplingBlock(c_in, 64, use_norm=False)
        self.enc2 = DownsamplingBlock(64, 128)
        self.enc3 = DownsamplingBlock(128, 256)
        self.enc4 = DownsamplingBlock(256, 512)
        self.enc5 = DownsamplingBlock(512, 512)
        self.enc6 = DownsamplingBlock(512, 512)
        self.enc7 = DownsamplingBlock(512, 512)
        self.enc8 = DownsamplingBlock(512, c_out)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        return [x8, x7, x6, x5, x4, x3, x2, x1]

class UnetDecoder(nn.Module):
    def __init__(self, c_in=512, c_out=64, use_upsampling=False, mode='nearest'):
        super(UnetDecoder, self).__init__()
        self.dec1 = UpsamplingBlock(c_in, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode)
        self.dec2 = UpsamplingBlock(1024, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode)
        self.dec3 = UpsamplingBlock(1024, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode)
        self.dec4 = UpsamplingBlock(1024, 512, use_upsampling=use_upsampling, mode=mode)
        self.dec5 = UpsamplingBlock(1024, 256, use_upsampling=use_upsampling, mode=mode)
        self.dec6 = UpsamplingBlock(512, 128, use_upsampling=use_upsampling, mode=mode)
        self.dec7 = UpsamplingBlock(256, 64, use_upsampling=use_upsampling, mode=mode)
        self.dec8 = UpsamplingBlock(128, c_out, use_upsampling=use_upsampling, mode=mode)

    def forward(self, x):
        x9 = torch.cat([x[1], self.dec1(x[0])], 1)
        x10 = torch.cat([x[2], self.dec2(x9)], 1)
        x11 = torch.cat([x[3], self.dec3(x10)], 1)
        x12 = torch.cat([x[4], self.dec4(x11)], 1)
        x13 = torch.cat([x[5], self.dec5(x12)], 1)
        x14 = torch.cat([x[6], self.dec6(x13)], 1)
        x15 = torch.cat([x[7], self.dec7(x14)], 1)
        out = self.dec8(x15)
        return out

class UnetGenerator(nn.Module):
    def __init__(self, c_in=3, c_out=3, use_upsampling=False, mode='nearest'):
        super(UnetGenerator, self).__init__()
        self.encoder = UnetEncoder(c_in=c_in)
        self.decoder = UnetDecoder(use_upsampling=use_upsampling, mode=mode)
        self.head = nn.Sequential(
            nn.Conv2d(64, c_out, 3, 1, padding=1, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        outE = self.encoder(x)
        outD = self.decoder(outE)
        out = self.head(outD)
        return out

class PatchDiscriminator(nn.Module):
    def __init__(self, c_in=3, c_hid=64, n_layers=3):
        super(PatchDiscriminator, self).__init__()
        model = [DownsamplingBlock(c_in, c_hid, use_norm=False)]
        n_p, n_c = 1, 1
        for n in range(1, n_layers):
            n_p, n_c = n_c, min(2**n, 8)
            model.append(DownsamplingBlock(c_hid*n_p, c_hid*n_c))
        n_p, n_c = n_c, min(2**n_layers, 8)
        model.append(DownsamplingBlock(c_hid*n_p, c_hid*n_c, stride=1))
        model.append(nn.Conv2d(c_hid*n_c, 1, 4, 1, padding=1, bias=True))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)