# ELA_Code: Efficient Localization Attention for Deep Convolutional Neural Networks

Welcome to the official code repository for the paper **[ELA: Efficient Local Attention for Deep Convolutional Neural Networks](https://arxiv.org/abs/2403.01123)**. This project implements the Efficient Localization Attention (ELA) module, a lightweight spatial attention mechanism designed for convolutional neural networks (CNNs). ELA avoids channel reduction and uses 1D convolutions with Group Normalization to capture long-range dependencies, achieving superior performance with low computational cost across multiple vision tasks.

This repository provides implementations for:
- **Image Classification**: MobileNetV2 on MS COCO.
- **Object Detection**: YOLOF and YOLOX (primarily YOLOX-Nano) on MS COCO and Pascal VOC 2007.
- **Semantic Segmentation**: DeepLabV3 on Pascal VOC 2012.

The code supports training, evaluation, and testing of ELA and other attention modules, with modular designs for easy experimentation.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
  - [YOLOX (Object Detection)](#yolox-object-detection)
  - [YOLOF (Object Detection)](#yolof-object-detection)
  - [MbV2 (Image Classification)](#mbv2-image-classification)
  - [Semantic (Semantic Segmentation)](#semantic-semantic-segmentation)
- [Attention Modules](#attention-modules)
- [Citation](#citation)
- [Contact](#contact)

## Project Structure

ELA_Code/
├── MbV2/                    # MobileNetV2 for image classification (MS COCO)
│   ├── models/              # Model definitions with ELA and other attention modules
│   ├── train.py            # Training script
│   ├── test.py             # Testing script
│   └── weights/            # Directory for pretrained weights (user-provided)
├── YOLOF/                   # YOLOF for object detection (MS COCO)
│   ├── train.py            # Training script
│   ├── predict.py          # Prediction script
│   └── weights/            # Directory for pretrained weights (user-provided)
├── YOLOX/                   # YOLOX-Nano for object detection (Pascal VOC 2007)
│   ├── train.py            # Training script
│   ├── predict.py          # Prediction script
│   └── weights/            # Directory for pretrained weights (user-provided)
├── Semantic/                # DeepLabV3 for semantic segmentation (Pascal VOC 2012)
│   ├── models/             # Model definitions with ELA and other attention modules
│   ├── train.py            # Training script
│   ├── test.py             # Testing script
│   └── weights/            # Directory for pretrained weights (user-provided)
├── README.md                # This file
└── requirements.txt         # Required Python packages

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Xuwei86/ELA_Code.git
   cd ELA_Code

Set up a Python environment:
We recommend Python 3.8+ and PyTorch 1.8+.

Create a virtual environment (optional):
bash

python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

Install dependencies:
Install required packages listed in requirements.txt:
bash

pip install -r requirements.txt

Typical dependencies include:
plaintext

torch>=1.8.0
torchvision>=0.9.0
numpy
opencv-python
pycocotools
tqdm

Verify installation:
Ensure PyTorch is correctly installed with GPU support (if applicable):
bash

python -c "import torch; print(torch.cuda.is_available())"

Datasets
This project does not provide datasets. Users must download the following publicly available datasets and organize them as follows:
MS COCO (2017):
Used for MbV2 (classification) and YOLOF (detection).

Download from COCO dataset.

Expected structure:

coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    ├── instances_val2017.json

Pascal VOC 2007:
Used for YOLOX-Nano (detection).

Download from VOC 2007.

Expected structure:

VOC2007/
├── JPEGImages/
├── Annotations/
└── ImageSets/
    ├── Main/
        ├── train.txt
        ├── val.txt

Pascal VOC 2012:
Used for DeepLabV3 (segmentation).

Download from VOC 2012.

Expected structure:

VOC2012/
├── JPEGImages/
├── SegmentationClass/
└── ImageSets/
    ├── Segmentation/
        ├── train.txt
        ├── val.txt

Update the dataset paths in the configuration files or scripts (e.g., train.py) for each task.
Usage
YOLOX (Object Detection)
Task: Object detection on Pascal VOC 2007 using YOLOX-Nano.

Training:
Configure the dataset path in YOLOX/train.py or a config file.

Run the training script:
bash

cd YOLOX
python train.py

Trained weights will be saved in YOLOX/weights/.

Prediction:
Place a pretrained weight file in YOLOX/weights/ (e.g., yolox_nano.pth).

Configure the prediction settings (e.g., input image path) in YOLOX/predict.py.

Run the prediction script:
bash

python predict.py

Results (e.g., bounding boxes) will be saved or visualized as specified.

YOLOF (Object Detection)
Task: Object detection on MS COCO using YOLOF.

Usage: Similar to YOLOX.
Training: cd YOLOF; python train.py

Prediction: Place weights in YOLOF/weights/, then run python predict.py.

Note: Refer to YOLOF/README.md (if available) for specific configurations or debug the scripts directly.

MbV2 (Image Classification)
Task: Image classification on MS COCO using MobileNetV2.

Training:
Configure the dataset path in MbV2/train.py.

Run the training script:
bash

cd MbV2
python train.py

Weights will be saved in MbV2/weights/.

Testing/Customization:
Test with pretrained weights: python test.py.

Customize attention modules: Modify MbV2/models/attention/ to include your own attention mechanisms (e.g., CA, SE, or custom modules).

Multiple attention variants (e.g., ELA, CA, SE) are provided in MbV2/models/attention/.

Semantic (Semantic Segmentation)
Task: Semantic segmentation on Pascal VOC 2012 using DeepLabV3.

Training:
Configure the dataset path in Semantic/train.py.

Run the training script:
bash

cd Semantic
python train.py

Weights will be saved in Semantic/weights/.

Testing/Customization:
Test with pretrained weights: python test.py.

Customize attention modules: Modify Semantic/models/attention/ to test ELA or other attention mechanisms.

Multiple attention variants are provided in Semantic/models/attention/.

Attention Modules
The repository includes implementations of ELA and several state-of-the-art attention modules for comparison, located in:
MbV2/models/attention/

Semantic/models/attention/

Available modules include:
ELA: Our proposed Efficient Localization Attention.

CA: Coordinate Attention \cite{hou2021coordinate}.

SE: Squeeze-and-Excitation \cite{senet}.

CBAM, ECA, SA, Sea, GC, Strip, Amca (see paper for details).

Users can easily integrate custom attention modules by modifying the model definitions in models/attention/.
Citation
If you find this code or the ELA module useful, please cite our paper:
bibtex

@article{xu2024ela,
  title={ELA: Efficient Local Attention for Deep Convolutional Neural Networks},
  author={Xu, Wei and others},
  journal={arXiv preprint arXiv:2403.01123},
  year={2024}
}


