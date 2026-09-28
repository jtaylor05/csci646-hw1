import torch.nn as nn

class ResUnit(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 n: int, 
                 *,
                 kernel = (3, 3), 
                 stride = (1, 1), 
                 padding = (1, 1),
                 dropout = 0.2):
        super(ResUnit, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel, 1 if in_channels == out_channels else 2, padding, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        for i in range(n-1):
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(out_channels, out_channels, kernel, stride, padding, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.Dropout(dropout))
        
        self.unit = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.unit(x)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip: bool = True):
        super(ResBlock, self).__init__()
        
        self.skip = skip
        
        self.unit1 = ResUnit(in_channels, out_channels, 2)
        self.unit2 = ResUnit(out_channels, out_channels, 2)
        
        if in_channels == out_channels:
            def wrapper(x): return x
            self.resize = wrapper
        else:
            self.resize = nn.Conv2d(in_channels, out_channels, 1, 2)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        a = self.unit1(x)
        skip1 = a
        if self.skip:
            skip1 = self.relu(self.resize(x) + a)
        else:
            skip1 = self.relu(a)
        b = self.unit2(skip1)
        if self.skip:
            return self.relu(skip1 + b)
        return self.relu(b)
        
        
    
class ResNet_18(nn.Module):
    def __init__(self, skip: bool = True):
        super(ResNet_18, self).__init__()
        
        self.resnet = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),   # PRE-BLOCK (32, 32)
            ResBlock(64, 64, skip),   # BLOCK 1 (32, 32)
            ResBlock(64, 128, skip),  # BLOCK 2 (16,16)
            ResBlock(128, 256, skip), # BLOCK 3 (8, 8)
            ResBlock(256, 512, skip), # BLOCK 4 (4, 4)
            nn.AvgPool2d(kernel_size=4, stride=1), # POOL (1, 1)
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=10)
        )
    
    def forward(self, x):
        return self.resnet(x)
        