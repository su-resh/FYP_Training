import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 16)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 16), channels),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x).unsqueeze(-1).unsqueeze(-1)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa)
        x = x * sa
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))
        x5 = F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.relu(self.bn_out(self.conv_out(x)))
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_cbam=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        return x


class UNetWithClassification(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet50(weights=weights)

        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        e5_channels = 2048
        self.aspp = ASPP(e5_channels, e5_channels // 4)

        aspp_channels = e5_channels // 4
        e4_channels = 1024
        e3_channels = 512
        e2_channels = 256
        e1_channels = 64

        self.up5 = nn.ConvTranspose2d(aspp_channels, aspp_channels, 2, stride=2)
        self.dec5 = DoubleConv(e4_channels + aspp_channels, aspp_channels)

        self.up4 = nn.ConvTranspose2d(aspp_channels, aspp_channels, 2, stride=2)
        self.dec4 = DoubleConv(e3_channels + aspp_channels, aspp_channels // 2)

        self.up3 = nn.ConvTranspose2d(aspp_channels // 2, aspp_channels // 4, 2, stride=2)
        self.dec3 = DoubleConv(e2_channels + aspp_channels // 4, aspp_channels // 4)

        self.up2 = nn.ConvTranspose2d(aspp_channels // 4, aspp_channels // 8, 2, stride=2)
        self.dec2 = DoubleConv(e1_channels + aspp_channels // 8, aspp_channels // 8)

        self.seg_head = nn.Sequential(
            nn.Conv2d(aspp_channels // 8, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(e5_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
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
        e5 = self.aspp(e5)
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
        out = self.seg_head(d2)
        return out

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encode(x)
        mask = self.decode(e1, e2, e3, e4, e5)
        classification = self.classifier(e5)
        return mask, classification


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class ComboSegLoss(nn.Module):
    def __init__(self, focal_weight=0.3, dice_weight=0.5, bce_weight=0.2):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        return (self.focal_weight * self.focal(pred, target) +
                self.dice_weight * self.dice(pred, target) +
                self.bce_weight * self.bce(pred, target))


class LabelSmoothCELoss(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        n_classes = pred.size(1)
        with torch.no_grad():
            smooth_target = torch.full_like(pred, self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        log_probs = F.log_softmax(pred, dim=1)
        loss = -(smooth_target * log_probs).sum(dim=1)
        if self.weight is not None:
            loss = loss * self.weight.to(pred.device).gather(0, target)
        return loss.mean()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


if __name__ == '__main__':
    model = UNetWithClassification(num_classes=2)
    print(f"Model parameters: {count_parameters(model):,}")

    x = torch.randn(1, 3, 512, 512)
    mask, classification = model(x)
    print(f"Mask shape: {mask.shape}")
    print(f"Classification shape: {classification.shape}")
