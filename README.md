<div align="center">

# Data-driven Limited Area Mesoscale Prediction for Taiwan (DLAMP.tw)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB)](https://www.python.org/downloads/release/python-310/)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-purple?logo=PyTorch&link=https%3A%2F%2Fpypi.org%2Fproject%2Ftorch%2F)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch--lightning-2.2.4-blue?link=https%3A%2F%2Flightning.ai%2Fdocs%2Fpytorch%2F2.0.3%2Flevels%2Fcore_skills.html)
</div>

This repository contains the codebase of `DLAMP.tw`, a pure data-driven regional forecasting model. This model's backbone design is based on [Pangu-weather](https://github.com/198808xc/Pangu-Weather). We make a varient of it by adding DDPM diffusion process to strengthen the ability of predicting stochastic convections.

![](./assets/demo/typhoons.gif)
Demo for Typhoon Muifa (2022). From left to right are: (a). ground truth, (b). prediction of Swin-Transformer (Pangu-weather) model and (c). Swin-Transformer predicts the mean field plus DDPM predicts the convections.

# Build Environment
activate a virtual envirionment, here use conda env
```
conda create --name [env name] python=3.11 -y
conda activate [env name]
```
install NVIDIA modulus package
```
git clone git@github.com:NVIDIA/modulus.git && cd modulus
make install
```
other packages
```
pip3 install -r requirements.txt
pip3 install hydra-core --upgrade
```
install onnxruntime according to your CUDA version, please check [onnxruntime_official](https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime) for more details.
```
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

# Quick Start
step 1. set hyperparameters
1. `config/**/*.yaml`  
   Note: all the configurations are wrapped by `@hydra.main` during training.
2. `src/const.py`

step 2. start training
```bash
python train.py
```

step 3. start inference
```bash
python predict.py
```

# Model Zoo
|Model Name|Total Params|Shape|Backbone|Reference|
|:----:|:----:|:----:|:----:|:---:|
|Swin-Transformer|29M|(B, 23, 224, 224)|-|[Pangu-weather](https://arxiv.org/abs/2211.02556)|
|Diffusion Model (DDPM)|42M|(B, 1, 224, 224)|ResUNet|[DDPM](https://arxiv.org/abs/2006.11239)|

# Acknowledgement
This project is sponsored by the [Taiwan's Centeral Weather Administration](https://www.cwa.gov.tw/V8/C/) and the [Department of Atmospheric Sciences, NTU](https://www.as.ntu.edu.tw). Without the support of the governmental and educational institutions, this project is not possible. Also, big thanks to the co-author [Tracy](https://github.com/tracylo1221). She helps me make many plots and figures.