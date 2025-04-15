# SiGAN

[![SIGAN](https://cchsu.info/files/sigan.jpg "SIGAN")](https://cchsu.info/files/sigan.jpg "SIGAN")

The implementation of paper Chih-Chung Hsu, Chia-Wen Lin, Weng-Tai Su, Gene Cheung, SiGAN: Siamese Generative Adversarial Network for Identity-Preserving Face Hallucination, published in IEEE Transactions on Image Processing (TIP) 2019.
Please cite if you use our code on your research.


We modify the code forked from https://github.com/david-gpu/srez to implement pairwise learning architecture for face hallucination.

# Reqirements

Tensorflow 1.08-1.12 - Does not support tensorflow 2.0 yet.

# Pretrained Model

The trained model for super-resolve 32x32 to 128x128 image can be downloaded from
https://drive.google.com/file/d/1qvWqsRfP2hZrZzXOG4NmZRuxb7fHkFAe/view?usp=sharing

# How to use

## Setup

1. Create a Python 3.6 conda environment

```bash
conda create -n sigan python=3.6
conda activate sigan
```

2. Install requirements

```bash
pip install -r requirements.txt
conda install jupyter
conda install -c conda-forge cudatoolkit=9.0 cudnn=7
```

3. Launch Jupyter notebook

```bash
jupyter-notebook
```

## Training

Open and run `srez_train_sia.ipynb` in Jupyter

Requires the CASIA-WebFace dataset.

## Testing your own image with trained model

Under the jupyter notebook, you can run the following notebook to see the result.
>test_sia.ipynb

Or directly run
>python SRDemo.py

to produce the super-resolved images sized of 128x128 from LR inputs 32x32.

# Dataset

Our dataset is based on "CASIA-WebFaces".

# Citation
    @ARTICLE{8751141,
    author={C. {Hsu} and C. {Lin} and W. {Su} and G. {Cheung}},
    journal={IEEE Transactions on Image Processing},
    title={SiGAN: Siamese Generative Adversarial Network for Identity-Preserving Face Hallucination},
    year={2019},
    volume={28},
    number={12},
    pages={6225-6236},
    keywords={face recognition;image reconstruction;image representation;image resolution;iterative methods;learning (artificial intelligence);SiGAN;Siamese generative adversarial network;identity-preserving face hallucination;generative adversarial networks;high-quality high-resolution;identity preservation;identical generators;reconstruction error;identity label information;loss function;generator pair;face reconstruction;identity recognition;objective face verification performance;visual-quality reconstruction;unseen identities;face hallucination GAN;Siamese GAN;Face;Image reconstruction;Face recognition;Training;Generators;Image resolution;Generative adversarial networks;Face hallucination;convolutional neural networks;generative adversarial networks;super-resolution;generative model},
    doi={10.1109/TIP.2019.2924554},
    ISSN={},
    month={Dec},}
