# Ship-motion-and-Radar-images-dataset
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15735441.svg)](https://doi.org/10.5281/zenodo.15735441)
# **///////////////Regarding the code files//////////////**
We uploaded our complete code to the repository, and the file name was "Code".
# Wave Parameter Inversion Model: Code and Dataset

This repository contains the code and dataset used to train and evaluate a wave-parameter inversion model that fuses shipborne X-band radar imagery and ship motion time series. The implementation is in Python (PyTorch). The model predicts two targets: **Characteristic Wave Period** and **Significant Wave Height**.

---

## Required Environment

- **Python**: compatible with Python 3.8+  
- **Key Python packages**: PyTorch, torchvision, pandas, numpy, scikit-learn, matplotlib, Pillow, scipy, glob2 (see `requirements.txt` for exact versions used in experiments)  
- **Hardware**: recommended GPU with CUDA support (e.g., NVIDIA GeForce RTX 3060). CPU-only execution is supported but significantly slower.  
- **Optional**: `torchinfo` for printing a model summary.

---

## Dataset (expected layout)

Place the dataset files under the data directories referenced in `Config` (or adjust `Config` accordingly). The code expects:

all_pitch.csv # ship motion time series (one or more files)

wave.csv # ground-truth wave parameters (columns: characteristic period, significant wave height)

cropped_radar_images/ # directory of radar images, named like radar_image_<id>_cropped.png

markdown
复制代码

Each sample combines one radar image, one motion sequence of length `Config.sequence_length`, and the corresponding wave parameter label.

---

## Data preprocessing (what the code does)

- **Motion data**: segmented into fixed-length sequences (length defined by `Config.sequence_length`) and standardized using dataset-level mean/std.  
- **Wave labels**: normalized (z-score) using dataset-level mean/std for each target.  
- **Radar images**: resized to `224×224`, converted to single channel, then normalized.  
- The `WaveDataProcessor` class performs loading, normalization, file matching, and dataset splits.

---

## Usage (high-level steps)

1. **Configure paths and hyperparameters**  
   Edit the `Config` class to point to your dataset and to set `batch_size`, `epochs`, `lr`, `sequence_length`, loss weights, device selection, etc.

2. **Prepare data objects**  
   ```python
   processor = WaveDataProcessor()
   motion_data, wave_data = processor.load_data()
   train_dataset, val_dataset, test_dataset = processor.create_datasets(motion_data, wave_data)
Create data loaders

python
复制代码
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False, num_workers=2)
Instantiate model and trainer

python
复制代码
model = WaveParameterModel(motion_data_dim=Config.sequence_length)
trainer = Trainer(model, config, processor)
Train

python
复制代码
trainer.train(train_loader, val_loader, test_loader)
Evaluate / Test
During/after training the best model is saved as best_wave_model.pth. To evaluate on the test set the script calls:

python
复制代码
test_metrics = trainer.test(test_loader)
Load saved model

python
复制代码
model.load_state_dict(torch.load("best_wave_model.pth"))
