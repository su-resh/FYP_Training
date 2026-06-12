import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNetWithClassification(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        if pretrained:
            resnet = resnet34(weights='IMAGENET1K_V1')
        else:
            resnet = resnet34(weights=None)
        
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.up5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec5 = DoubleConv(512, 256)
        
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        return e1, e2, e3, e4, e5
    
    def resize_to(self, x, target):
        if x.shape[2:] != target.shape[2:]:
            return F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        return x
    
    def decode(self, e1, e2, e3, e4, e5):
        d5 = self.up5(e5)
        e4_resized = self.resize_to(e4, d5)
        d5 = torch.cat([d5, e4_resized], dim=1)
        d5 = self.dec5(d5)
        
        d4 = self.up4(d5)
        e3_resized = self.resize_to(e3, d4)
        d4 = torch.cat([d4, e3_resized], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e2_resized = self.resize_to(e2, d3)
        d3 = torch.cat([d3, e2_resized], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e1_resized = self.resize_to(e1, d2)
        d2 = torch.cat([d2, e1_resized], dim=1)
        d2 = self.dec2(d2)
        
        out = self.final(d2)
        
        return out
    
    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encode(x)
        mask = self.decode(e1, e2, e3, e4, e5)
        classification = self.classifier(e5)
        return mask, classification


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = UNetWithClassification(num_classes=2)
    print(f"Model parameters: {count_parameters(model):,}")
    
    x = torch.randn(1, 3, 384, 384)
    mask, classification = model(x)
    print(f"Mask shape: {mask.shape}")
    print(f"Classification shape: {classification.shape}")