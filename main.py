import torch
import torch.nn as nn
import os
import cv2
import numpy as np

class UNet(nn.Module):
  def __init__(self):
    super(UNet,self).__init__()

    self.downsample = nn.Sequential(

        nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.01, inplace = True),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.01, inplace = True),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.01, inplace = True),
        nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.01, inplace = True),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=0.01, inplace = True),
        nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=0.01, inplace = True),
        nn.MaxPool2d(2),

    )

    self.upsample = nn.Sequential(

        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True),
        nn.Conv2d(256, 128, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.01, inplace = True),
        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True),
        nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.01, inplace = True),
        nn.MaxPool2d(2),

        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True),
        nn.Conv2d(128, 64, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.01, inplace= True),
        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope = 0.01, inplace = True),
        nn.MaxPool2d(2),

        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True),
        nn.Conv2d(64, 3, kernel_size = 3, padding = 1),
        nn.ReLU()
    )

  def forward(self,x):
    ds = self.downsample(x)
    output = self.upsample(ds)

    return output
  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = UNet()
model_path = 'models/15_unet_ploss_vgg19.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

input_path = "test/low"
output_path = "test/high"

if not os.path.exists(output_path):
    os.makedirs(output_path)

all_files = os.listdir(input_path)

for image_name in all_files:
    image_path = os.path.join(input_path, image_name)
    image = cv2.imread(image_path)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_size = (128, 128)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted = model(image).squeeze(0).cpu().numpy()
    
    predicted = predicted.transpose(1, 2, 0)
    predicted = (predicted * 255).astype(np.uint8)

    output_image_path = os.path.join(output_path, image_name)
    cv2.imwrite(output_image_path, cv2.cvtColor(predicted, cv2.COLOR_RGB2BGR))

