import torch.nn as nn

class SimpleObjectDetectorParam(nn.Module):
    def __init__(self, input_width, input_height, num_classes=5, max_objects=5):
        super(SimpleObjectDetector, self).__init__()
        
        self.max_objects = max_objects
        self.num_classes = num_classes
        
        # Define the number of pooling layers and the reduction factor for each pooling operation
        self.num_pooling_layers = 4
        self.pooling_reduction_factor = 2  # because we're using MaxPool2d with kernel_size=2
        
        # Calculate the spatial dimensions after the convolutional layers
        self.final_width = input_width // (self.pooling_reduction_factor ** self.num_pooling_layers)
        self.final_height = input_height // (self.pooling_reduction_factor ** self.num_pooling_layers)
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Fully connected layers now use the parameterized dimensions
        flattened_size = 128 * self.final_width * self.final_height
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes * self.max_objects)
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4 * self.max_objects)
        )
        
    def forward(self, x):
        x = self.features(x)
        
        class_output = self.classifier(x)
        class_output = class_output.view(-1, self.max_objects, self.num_classes)  # reshape to batch_size x max_objects x num_classes
        
        bbox_output = self.regressor(x)
        bbox_output = bbox_output.view(-1, self.max_objects, 4)  # reshape to batch_size x max_objects x 4
        
        return bbox_output, class_output

import torch
import torch.nn as nn

class SimpleObjectDetector(nn.Module):
    def __init__(self, num_objects=5, num_classes=5):
        super(SimpleObjectDetector, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 64 * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.fc_bbox = nn.Linear(512, num_objects * 4)  # Bounding box regression
        self.fc_class = nn.Linear(512, num_objects * num_classes)  # Class labels

        # Fully connected layers
#        self.fc_bbox = nn.Linear(64 * 500 * 375, num_objects * 4)  # Bounding box regression
 #       self.fc_class = nn.Linear(64 * 500 * 375, num_objects * num_classes)  # Class labels

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        bbox = self.fc_bbox(x)
        cls = self.fc_class(x)
        return bbox, cls


class SimpleObjectDetectorDropout(nn.Module):
    def __init__(self, num_objects=5, num_classes=5):
        super(SimpleObjectDetectorDropout, self).__init__()
        
       # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),  # Added dropout layer
            
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),  # Added dropout layer
            
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),  # Added dropout layer
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 64 * 64, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),  # Added dropout layer before the next FC layer
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Added dropout layer before the output layers
        )
        
        self.fc_bbox = nn.Linear(512, num_objects * 4)  # Bounding box regression
        self.fc_class = nn.Linear(512, num_objects * num_classes)  # Class labels

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        bbox = self.fc_bbox(x)
        cls = self.fc_class(x)
        return bbox, cls

# reduced parameters size
class SimpleObjectDetectorRed(nn.Module):
    def __init__(self, num_objects=5, num_classes=5):
        super(SimpleObjectDetectorRed, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),  # Additional Conv layer
            nn.MaxPool2d(2, 2)  # Additional Pooling layer
        )
        
        # Compute the size for the first Linear layer
        fc_input_size = 128 * (512 // 16) * (512 // 16)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 512),  # Adjusted the output dimension
            nn.ReLU(),
            nn.Linear(512, 256),  # Reduced this as well
            nn.ReLU()
        )
        self.fc_bbox = nn.Linear(256, num_objects * 4)
        self.fc_class = nn.Linear(256, num_objects * num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        bbox = self.fc_bbox(x)
        cls = self.fc_class(x)
        return bbox, cls

class SimpleObjectDetectorRedInput(nn.Module):
    def __init__(self, num_objects=5, num_classes=5, input_size=(512, 512)):
        super(SimpleObjectDetectorRedInput, self).__init__()
        
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


import torch.nn as nn

class SimpleObjectDetectorBN(nn.Module):
    def __init__(self, num_objects=5, num_classes=5):
        super(SimpleObjectDetectorBN, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # Adjusted size for FC layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.fc_bbox = nn.Linear(256, num_objects * 4)  # Bounding box regression
        self.fc_class = nn.Linear(256, num_objects * num_classes)  # Class labels

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        bbox = self.fc_bbox(x)
        cls = self.fc_class(x)
        return bbox, cls


import torch.nn as nn
import torchvision.models as models


class SimpleObjectDetectorWithResnet(nn.Module):
    def __init__(self, num_objects=5, num_classes=5):
        super(SimpleObjectDetectorWithResnet, self).__init__()
        
        # Load pre-trained ResNet-18 and remove the fully connected layer
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Flattened features from ResNet-18 will be of size: (batch_size, 512, 16, 16)
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.5), # Added dropout for regularization
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5)  # Added dropout for regularization
        )
        
        self.fc_bbox = nn.Linear(512, num_objects * 4)  # Bounding box regression
        self.fc_class = nn.Linear(512, num_objects * num_classes)  # Class labels
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        bbox = self.fc_bbox(x)
        cls = self.fc_class(x)
        return bbox, cls

import torch.nn as nn
import torchvision.models as models

class FPN(nn.Module):
    def __init__(self, num_objects=5, num_classes=5):
        super(FPN, self).__init__()

        # Load pre-trained ResNet18 and remove avg pool and FC
        backbone = models.resnet18(pretrained=True)
        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Lateral layers
        self.lat_layer1 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer4 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.top_layer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.top_layer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.top_layer3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Final layers for object detection (based on finest resolution FPN map)
        #self.fc_bbox = nn.Linear(256 * 32 * 32, num_objects * 4)
        #self.fc_class = nn.Linear(256 * 32 * 32, num_objects * num_classes)
        # Adjust these dimensions for a 512x512 input:
        self.fc_bbox = nn.Linear(256 * 128 * 128, num_objects * 4)
        self.fc_class = nn.Linear(256 * 128 * 128, num_objects * num_classes)


    def forward(self, x):
        # Bottom-up pathway
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # Lateral connections
        p4 = self.lat_layer4(c4)
        p3 = self.lat_layer3(c3) + F.interpolate(p4, scale_factor=2)
        p2 = self.lat_layer2(c2) + F.interpolate(p3, scale_factor=2)
        p1 = self.lat_layer1(c1) + F.interpolate(p2, scale_factor=2)

        # Top-down pathway
        p1 = self.top_layer1(p1)
        p2 = self.top_layer2(p2)
        p3 = self.top_layer3(p3)

        # Flatten for FC layers
        p1_flat = p1.view(p1.size(0), -1)

        # Detection heads
        bbox = self.fc_bbox(p1_flat)
        cls = self.fc_class(p1_flat)
        
        return bbox, cls



import torch.nn as nn
import torch.nn.functional as F

class GrayScaleObjectDetector(nn.Module):
    def __init__(self, num_objects=5, num_classes=5, input_size=(1024, 1024)):
        super(GrayScaleObjectDetector, self).__init__()
        
        # Convert the image to grayscale: 3 -> 1 channel
        self.to_gray = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1)
        )
        
        # The main feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            #nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            #nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            #nn.MaxPool2d(2, 2)
        )
        
        # Compute the output size after convolutional layers
        conv_output_size = self.compute_conv_output_size(input_size)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_bbox = nn.Linear(128, num_objects * 4)
        self.fc_class = nn.Linear(128, num_objects * num_classes)

    def forward(self, x):
        x = self.to_gray(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        bbox = self.fc_bbox(x)
        cls = self.fc_class(x)
        return bbox, cls

    def compute_conv_output_size(self, input_size):
        self.features.eval()
            
        with torch.no_grad():
            x = torch.zeros(1, 3, *input_size)
            x = self.to_gray(x)  # Convert to grayscale first
            x = self.features(x)
            
        return x.view(-1).shape[0]
    

class FPNCAT(nn.Module):
    def __init__(self, num_objects=5, num_classes=5):
        super(FPNCAT, self).__init__()

        # Load pre-trained ResNet18 and remove avg pool and FC
        backbone = models.resnet18(pretrained=True)
        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Lateral layers
        self.lat_layer1 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer4 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.top_layer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.top_layer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.top_layer3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Final layers for object detection (based on concatenated FPN maps)
        self.fc_bbox = nn.Linear(3 * 256 * 96 * 96, num_objects * 4)
        self.fc_class = nn.Linear(3 * 256 * 96 * 96, num_objects * num_classes)

    def forward(self, x):
        # Bottom-up pathway
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # Lateral connections
        p4 = self.lat_layer4(c4)
        p3 = self.lat_layer3(c3) + F.interpolate(p4, scale_factor=2)
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

        # Detection heads
        bbox = self.fc_bbox(combined_flat)
        cls = self.fc_class(combined_flat)
        
        return bbox, cls

