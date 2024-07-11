
# Image Denoising Using UNets and Perceptual Loss

The project aims at proposing a model that does automatic image denoising using a UNet architecture and Perceptual loss.

Note : The model was trained on kaggle P100 GPU. There in some places in model.ipynb file you might find kaggle/working as directory.
```
Image-Denoising/
├── denoising-skipconnections (1).ipynb (model with skip connections, couldn't train due to lack of computation)
├── model.ipynb (without skip connections, trained)
├── requirements.txt
├── README.md
├── test/
│   ├── high/
│   └── low/
├── models/
│   ├── 10_unet_ploss_vgg19.pth
│   ├── 15_unet_ploss_vgg19.pth
│   ├── 20_unet_ploss_vgg19.pth
│   ├── 25_unet_ploss_vgg19.pth
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
```
Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install the required dependencies:
```
pip install -r requirements.txt
```
### Usage
Insert images images that you want to denoise in ```/test/low```
Run the following command
```bash
python3 main.py
```
Your denoised images will be saved to ```/test/low```


