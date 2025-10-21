# Ship-motion-and-Radar-images-dataset
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15735441.svg)](https://doi.org/10.5281/zenodo.15735441)
# **///////////////Regarding the code files//////////////**
We uploaded our complete code to the repository, and the file name was "Code".
# Wave Parameter Inversion Model: Code and Dataset

This repository contains the code and dataset used to train and evaluate a wave-parameter inversion model that fuses shipborne X-band radar imagery and ship motion time series. The implementation is in Python. The model predicts two targets: **Characteristic Wave Period** and **Significant Wave Height**.

---

## Required Environment

- **Python**: compatible with Python 3.8+  
- **Key Python packages**: PyTorch, torchvision, pandas, numpy, scikit-learn, matplotlib, Pillow, scipy, glob2 
- **Hardware**: recommended GPU with CUDA support. CPU-only execution is supported but significantly slower.  
- **Optional**: `torchinfo` for printing a model summary.

---

## Dataset (expected layout)

Place the dataset files under the data directories referenced in `Config` (or adjust `Config` accordingly). The code expects:

all_pitch.csv # ship motion time series

wave.csv # ground-truth wave parameters (columns: characteristic period, significant wave height)

cropped_radar_images/ # directory of radar images, named radar_image_<id>_cropped.png

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
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False, num_workers=2)
Instantiate model and trainer

model = WaveParameterModel(motion_data_dim=Config.sequence_length)
trainer = Trainer(model, config, processor)
Train

trainer.train(train_loader, val_loader, test_loader)
Evaluate / Test
During/after training the best model is saved as best_wave_model.pth. To evaluate on the test set the script calls:

test_metrics = trainer.test(test_loader)
Load saved model

model.load_state_dict(torch.load("best_wave_model.pth"))
Training details (as implemented)

Optimizer: torch.optim.AdamW with learning rate Config.lr (default 1e-4).

Scheduler: ReduceLROnPlateau (monitoring validation loss, factor=0.5, patience=5).

Loss: weighted L1 loss for two outputs; per-target weights defined in Config (tp_loss_weight, sh_loss_weight).

Gradient clipping: global norm clipped to 1.0.

Batch size / epochs: controlled via Config. Early stopping with patience (20 by default) is applied based on validation loss.

Random seeds: torch.manual_seed(42) and np.random.seed(42) are set for reproducibility.

Outputs and saved artifacts

best_wave_model.pth — saved model state dict (best validation loss).

training_log.txt — per-epoch training / validation loss and selected metrics.

wave_model_test_results.xlsx / .csv — detailed per-sample test results (true, pred, error, relative error).

wave_model_training_history.xlsx or .csv — training history including validation metrics per epoch.

loss_curve.png — training and validation loss curve.

prediction_scatter.png — scatter plots of predicted vs. true values for both targets.

Evaluation metrics produced

Per-target and overall metrics are computed (after denormalization) and include:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

MSE (Mean Squared Error)

R² (Coefficient of Determination)

MRE (Mean Relative Error)

These metrics are printed to console and saved in the history/test result files.

Implementation notes

The model is a dual-branch fusion network: a CNN-based branch for radar imagery and a BiLSTM-based branch for ship motion time series, with channel/spatial/temporal attention modules and a progressive cross-modal fusion module.

The code is modular: you can swap encoders, adjust attention modules, or change fusion strategies by editing the corresponding classes.

Data splits are deterministic (index-based) to ensure reproducibility of training/validation/test partitions.
