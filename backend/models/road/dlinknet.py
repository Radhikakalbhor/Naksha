import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


# Dilated Convolutional Block
class Dblock(nn.Module):
    """
    Dblock sử dụng nhiều convolution với các dilation rates khác nhau.
    Điều này giúp mở rộng receptive field mà không làm giảm độ phân giải.
    """
    def __init__(self, in_channels):
        super(Dblock, self).__init__()

        # 4 lớp convolution với dilation tăng dần: (1, 2, 4, 8)
        self.dilate1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=8, padding=8)

    def forward(self, x):
        """
        Đầu vào: Feature map từ Encoder (ResNet-34)
        Trả về: Feature map sau khi qua Dilated Convolutions
        """
        d1 = F.relu(self.dilate1(x))
        d2 = F.relu(self.dilate2(d1))
        d3 = F.relu(self.dilate3(d2))
        d4 = F.relu(self.dilate4(d3))

        return x + d1 + d2 + d3 + d4  # Skip connection giúp tránh mất thông tin
    


# Decoder Block
class DecoderBlock(nn.Module):
    """
    DecoderBlock gồm 2 Convolution layers + 1 UpSampling để phục hồi kích thước ảnh.
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Hai lớp convolution để học lại thông tin sau khi upsampling
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Upsampling bằng Transposed Convolution (Deconvolution)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.upsample(x)  # Up-sampling kích thước ảnh
        return x


# D-LinkNet Model
class DLinkNet(nn.Module):
    def __init__(self, num_classes=1):
        """
        Kiến trúc chính của D-LinkNet:
        - Encoder: ResNet-34 (Pretrained)
        - Bottleneck: Dilated Convolution Block
        - Decoder: UpSampling + Skip Connections
        """
        super(DLinkNet, self).__init__()

        # 🔹 **Encoder: Sử dụng ResNet-34 pretrained**
        resnet = models.resnet34(pretrained=True)

        # Lấy phần feature extractor từ ResNet-34 (bỏ đi fully connected layers)
        self.first_conv = nn.Sequential(
            resnet.conv1,  # (64, H/2, W/2)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool # (64, H/4, W/4)
        )

        # 4 Block của ResNet-34 (Dùng làm Encoder)
        self.encoder1 = resnet.layer1  # (64, H/4, W/4)
        self.encoder2 = resnet.layer2  # (128, H/8, W/8)
        self.encoder3 = resnet.layer3  # (256, H/16, W/16)
        self.encoder4 = resnet.layer4  # (512, H/32, W/32)

        # 🔹 **Bottleneck: Dilated Convolution Block**
        self.dblock = Dblock(512)  # (512, H/32, W/32)

        # 🔹 **Decoder với Skip Connections**
        self.decoder4 = DecoderBlock(512, 256)  # (256, H/16, W/16)
        self.decoder3 = DecoderBlock(256, 128)  # (128, H/8, W/8)
        self.decoder2 = DecoderBlock(128, 64)   # (64, H/4, W/4)
        self.decoder1 = DecoderBlock(64, 32)    # (32, H/2, W/2)
        # self.decoder0 = DecoderBlock(32, 16)    # (32, H, W)

        # 🔹 **Final Convolution để tạo segmentation mask**
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Đầu vào: Ảnh có shape (Batch, 3, H, W)
        Đầu ra: Segmentation mask có shape (Batch, num_classes, H, W)
        """

        # **Encoder**
        x1 = self.first_conv(x)    # (Batch, 64, 256, 256)
        x2 = self.encoder1(x1)     # (Batch, 64, 256, 256)
        x3 = self.encoder2(x2)     # (Batch, 128, 128, 128)
        x4 = self.encoder3(x3)     # (Batch, 256, 64, 64)
        x5 = self.encoder4(x4)     # (Batch, 512, 32, 32)

        # **Dilated Convolution Block**
        x5 = self.dblock(x5)

        # **Decoder với Skip Connections**
        d4 = self.decoder4(x5) + x4  # (Batch, 256, 64, 64)
        d3 = self.decoder3(d4) + x3  # (Batch, 128, 128, 128)
        d2 = self.decoder2(d3) + x2  # (Batch, 64, 256, 256)
        d1 = self.decoder1(d2) 
        # d0 = self.decoder0(d1)

        # **Final Output**
        output = self.final_conv(d1)  # (Batch, num_classes, H, W)
        output = F.interpolate(output, size=(1024, 1024), mode='bilinear', align_corners=False)  # Upscale về 1024x1024
        output = torch.sigmoid(output)  # Vì đây là bài toán segmentation binary

        return output