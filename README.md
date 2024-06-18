
# Image Denoising Using UNets and Perceptual Loss

The project aims at proposing a model that does automatic image denoising using a UNet architecture and Perceptual loss.

Note : The model was trained on kaggle P100 GPU. There in some places in model.ipynb file you might find kaggle/working as directory.
```
Image-Denoising/
├── requirements.txt
├── README.md
├── test/
│   ├── high/
│   └── low/
├── models/
│   ├── 10_unet_ploss_vgg19.pth
│   ├── 15_unet_ploss_vgg19.pth (renamed 15epoch)
│   ├── 20_unet_ploss_vgg19.pth
│   ├── 25_unet_ploss_vgg19.pth
├── model.ipynb
├── 25_unet_ploss_vgg19.pth
├── main.py
└── running_on_test_data/
     ├── 15_unet_ploss_vgg19.pth
     ├── experiments.ipynb
     ├── testData_prep.ipynb
     ├── test_psnr.ipynb
     └── images/
          ├── ground_truth/
          ├── high/
          └── low/

```



## Installation

To deploy this project run

```bash
git clone https://github.com/Ayush-Singh677/Image-Denoising.git

cd Image-Denoising

pip install -r requirements.txt
```

