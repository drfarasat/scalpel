import torch
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from network import FPN

from network_fpn import FPNCATSimple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FPNCATSimple().to(device)
model = FPN().to(device)
    #model = FPNCATSimple().to(device)
    #model.load_state_dict(torch.load("path_to_saved_model.pth", map_location=device))
checkpoint_path = "savedweights/fpncatRes18/model_weights_epoch_3000.pt"
#checkpoint_path ="savedweights/simpleBfpncat/model_weights_epoch_4000.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
   

model.eval()


mean = [0.3250, 0.4593, 0.4189]
std = [0.1759, 0.1573, 0.1695]


data_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])
test_img_dir = "test_images"
test_imgs = [os.path.join(test_img_dir, img) for img in os.listdir(test_img_dir) if img.endswith('.png')]

# 3. Perform inference and visualize results
for img_path in test_imgs:
    image = Image.open(img_path).convert("RGB")
    #input_tensor = data_transforms(image).unsqueeze(0).to(device)
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_bboxes, pred_classes = model(input_tensor)
        

        
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    # Convert predictions to proper shape
    pred_bboxes = pred_bboxes.view(-1, 5, 4).cpu().numpy()
    pred_classes = torch.argmax(pred_classes.view(-1, 5, 5)[:5], dim=-1).cpu().numpy()


    for box, cls in zip(pred_bboxes[0], pred_classes[0]):
        # Assuming boxes are (x,y,w,h)
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, str(cls.item()), color='white', bbox=dict(facecolor='red', alpha=0.5))
    
    plt.show()
