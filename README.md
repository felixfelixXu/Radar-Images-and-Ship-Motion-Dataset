# Ship-motion-and-Radar-images-dataset
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15735441.svg)](https://doi.org/10.5281/zenodo.15735441)
# **///////////////Regarding the code files//////////////**
We uploaded our complete code to the warehouse, and the file name was "Code".
# **Wave Parameter Inversion Model: Code and Dataset**

This repository contains the code and dataset used for training and evaluating the wave parameter inversion model based on radar images and ship motion data. The model is implemented in Python using PyTorch and is designed to predict significant wave height and characteristic wave period from radar imagery and ship motion data.

## **Required Environment**

Before running the code, ensure that the following software and packages are installed:

### **1. Python Version**
The code is compatible with Python 3.8 and above.

### **2. Required Libraries**

To set up the required Python environment, you can use the following `requirements.txt` file or manually install the dependencies:

```plaintext
torch==1.13.0
torchvision==0.14.0
pandas==1.5.0
numpy==1.23.0
scikit-learn==1.0.2
matplotlib==3.4.3
Pillow==8.3.2
scipy==1.7.1
glob2==0.7
