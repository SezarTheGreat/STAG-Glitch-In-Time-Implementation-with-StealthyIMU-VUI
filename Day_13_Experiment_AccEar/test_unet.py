import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        
        self.enc1 = nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1)
        
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128)
        )
        
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256)
        )
        
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(512)
        )
        
        self.bottleneck = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True) 
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        
        d4 = self.dec4(b)
        d4_cat = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(d4_cat)
        d3_cat = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3_cat)
        d2_cat = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2_cat)
        d1_cat = torch.cat([d1, e1], dim=1)
        
        out = self.final_up(d1_cat)
        return out

checkpoint_path = r"C:\Users\jyoti\OneDrive\Desktop\STAG Implementation with StealthyIMU VUI\Day_13_Experiment_AccEar\checkpoints\accear_cgan_best_model.pt"

try:
    data = torch.load(checkpoint_path, map_location='cpu')
    model = UNetGenerator()
    model.load_state_dict(data['generator_state_dict'])
    print("UNetGenerator loaded successfully!")
    
    # Test forward pass
    dummy_input = torch.randn(1, 2, 256, 256)
    out = model(dummy_input)
    print(f"Forward pass successful, output shape: {out.shape}")
except Exception as e:
    print(f"Error: {e}")
