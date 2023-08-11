import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_channels, inner_channels):
        super(SelfAttention, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, inner_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(inner_channels, in_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, _, _ = x.size()
        y = self.global_avgpool(x).view(B, C)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(B, C, 1, 1)
        return x * y

class AttentionObjectDetector(nn.Module):
    def __init__(self, num_objects=5, num_classes=5, input_size=(512, 512)):
        super(AttentionObjectDetector, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Self Attention Layer
        self.attention = SelfAttention(128, 64) 
        
        # Compute the output size after convolutional layers
        conv_output_size = self.compute_conv_output_size(input_size)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_bbox = nn.Linear(256, num_objects * 4)
        self.fc_class = nn.Linear(256, num_objects * num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x) # Apply attention here
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        bbox = self.fc_bbox(x)
        cls = self.fc_class(x)
        return bbox, cls

    def compute_conv_output_size(self, input_size):
        # Temporarily set all layers to evaluation mode
        self.features.eval()
        
        with torch.no_grad():
            x = torch.zeros(1, 3, *input_size)
            x = self.features(x)
        
        # Return the product of dimensions (flattened size)
        return x.view(-1).shape[0]
