import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBackbone(nn.Module):
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Basic block 1
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        # Basic block 2
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        # Basic block 3
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)

    def forward(self, x):
        # Initial convolution + BN + ReLU + MaxPool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Basic block 1
        x1 = self.conv2(x)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)

        # Basic block 2
        x2 = self.conv3(x1)
        x2 = self.bn3(x2)
        x2 = self.relu(x2)

        # Basic block 3
        x3 = self.conv4(x2)
        x3 = self.bn4(x3)
        x3 = self.relu(x3)

        return x1, x2, x3

class FPNCATSimple(nn.Module):
    def __init__(self, num_objects=5, num_classes=5):
        super(FPNCATSimple, self).__init__()

        # Using the SimpleBackbone
        self.backbone = SimpleBackbone()

        # Lateral layers
        self.lat_layer1 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.top_layer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.top_layer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.top_layer3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Final layers for object detection (based on concatenated FPN maps)
        #self.fc_bbox = nn.Linear(3 * 256 * 32 * 32, num_objects * 4)
        #self.fc_class = nn.Linear(3 * 256 * 32 * 32, num_objects * num_classes)
        # Final layers for object detection (based on concatenated FPN maps)
        #self.fc_bbox = nn.Linear(3 * 256 * 256 * 256, num_objects * 4)
        #self.fc_class = nn.Linear(3 * 256 * 256 * 256, num_objects * num_classes)
        input_dim = 768 * 64 * 64  # from the printed shape of combined tensor
        self.fc_bbox = nn.Linear(input_dim, num_objects * 4)
        self.fc_class = nn.Linear(input_dim, num_objects * num_classes)



    def forward(self, x):
        # Use the SimpleBackbone
        c1, c2, c3 = self.backbone(x)

        # Lateral connections
        p3 = self.lat_layer3(c3)
        p2 = self.lat_layer2(c2) + F.interpolate(p3, scale_factor=2)
        p1 = self.lat_layer1(c1) + F.interpolate(p2, scale_factor=2)

        # Top-down pathway
        p1 = self.top_layer1(p1)
        p2 = self.top_layer2(p2)
        p3 = self.top_layer3(p3)

        # Concatenate the feature maps
        combined = torch.cat([
            p1, 
            F.interpolate(p2, size=p1.shape[-2:], mode='bilinear'), 
            F.interpolate(p3, size=p1.shape[-2:], mode='bilinear')
        ], dim=1)

        
        # Flatten for FC layers
        combined_flat = combined.view(combined.size(0), -1)
        #print("Shape of combined tensor:", combined.shape)

        # Detection heads
        bbox = self.fc_bbox(combined_flat)
        cls = self.fc_class(combined_flat)
        
        return bbox, cls
