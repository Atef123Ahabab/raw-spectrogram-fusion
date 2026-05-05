# Feature Fusion Project: Raw Signal + Spectrogram Classification

This project implements a two-stream neural network that combines raw time-domain signals and spectrograms for improved classification performance.

## Project Structure

The code is organized into 17 separate Python files, each representing a cell you can copy-paste into Google Colab:

### Setup and Preparation (Cells 1-7)
1. **01_imports_and_setup.py** - Import libraries and set random seeds
2. **02_dataset_class.py** - Dataset class and data preparation functions
3. **03_model_architectures.py** - All three model architectures
4. **04_training_functions.py** - Training and evaluation functions
5. **05_visualization_functions.py** - Plotting and analysis functions
6. **06_data_preparation.py** - Load and prepare datasets
7. **07_hyperparameters.py** - Training configuration

### Model Training (Cells 8-10)
8. **08_train_raw_only.py** - Train the RawOnlyModel
9. **09_train_spec_only.py** - Train the SpecOnlyModel
10. **10_train_fusion.py** - Train the TwoStreamFusionModel

### Evaluation and Analysis (Cells 11-17)
11. **11_plot_training_curves.py** - Plot training/validation curves
12. **12_evaluate_all_models.py** - Test set evaluation
13. **13_performance_comparison.py** - Compare all models
14. **14_confusion_matrices.py** - Generate confusion matrices
15. **15_per_class_analysis.py** - Detailed per-class metrics
16. **16_class_improvement_analysis.py** - Analyze which classes benefit from fusion
17. **17_discussion_summary.py** - Generate discussion points for report

## How to Use

### Option 1: Use with Your Assignment 2 Data

1. Open **02_dataset_class.py**
2. Replace the `prepare_data_from_assignment2()` function with your actual data loading code
3. Ensure it returns:
   - `train_raw, train_spec, train_labels`
   - `val_raw, val_spec, val_labels`
   - `test_raw, test_spec, test_labels`
   - `num_classes`

### Option 2: Use as-is with Synthetic Data

The code includes a placeholder that generates synthetic data. This is useful for:
- Testing the pipeline
- Understanding the expected data format
- Verifying everything works before using real data

## Model Architectures

### 1. RawBranch1D
- **Input**: Raw signal (B, T)
- **Architecture**: 4 Conv1D layers with batch norm
- **Output**: Embedding vector (B, emb_dim)

### 2. SpecBranch2D
- **Input**: Spectrogram (B, 1, F, T_spec)
- **Architecture**: 4 Conv2D layers with batch norm and pooling
- **Output**: Embedding vector (B, emb_dim)

### 3. FusionHead
- **Input**: Concatenated embeddings (B, 2*emb_dim)
- **Architecture**: 3 fully connected layers with dropout
- **Output**: Class logits (B, num_classes)

## Three Models Trained

1. **RawOnlyModel**: Uses only raw signals
2. **SpecOnlyModel**: Uses only spectrograms
3. **TwoStreamFusionModel**: Combines both (late fusion)

## Expected Outputs

The code will generate:

### Performance Metrics
- Accuracy and F1-score for all three models
- Comparison table and bar plots
- Improvement analysis

### Visualizations
- Training/validation loss curves
- Training/validation accuracy curves
- Confusion matrices (one for each model)
- Per-class F1-score comparison
- Class-level improvement analysis

### Discussion Points
- Whether fusion improves performance
- Which classes benefit most
- Explanations and observations

## Customization

### Adjust Hyperparameters (Cell 7)
```python
emb_dim = 128          # Embedding dimension
num_epochs = 30        # Training epochs
learning_rate = 0.001  # Learning rate
batch_size = 32        # Batch size (in cell 6)
```

### Modify Architectures (Cell 3)
You can adjust:
- Number of convolutional layers
- Filter sizes and numbers
- Dropout rates
- Embedding dimensions

### Change Class Names (Cell 6)
```python
class_names = ['Class 0', 'Class 1', ...]  # Replace with your actual class names
```

## Requirements

The code requires:
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- scipy (for spectrogram generation)
- tqdm

Install in Colab with:
```python
!pip install torch torchvision matplotlib seaborn scikit-learn scipy tqdm
```

## Tips for Google Colab

1. **Run cells in order** - Each cell may depend on previous ones
2. **Save your models** - Add model saving code if you want to preserve trained models
3. **Use GPU** - Go to Runtime → Change runtime type → GPU for faster training
4. **Adjust num_workers** - Set to 0 in DataLoader for Colab compatibility (already done)

## Expected Workflow

1. Copy cells 1-7 → Set up and prepare data
2. Copy cells 8-10 → Train all three models (will take time)
3. Copy cells 11-17 → Analyze and visualize results
4. Use the outputs for your project report

## Notes

- All models use the **same train/val/test splits** for fair comparison
- All models use the **same hyperparameters** (learning rate, epochs, etc.)
- The fusion model implements **late fusion** (fusion after feature extraction)
- Random seeds are set for **reproducibility**

## For Your Report

Include:
- Performance comparison table (from cell 13)
- All confusion matrices (from cell 14)
- Training curves (from cell 11)
- Per-class analysis (from cells 15-16)
- Discussion summary (from cell 17)

Good luck with your project! 🚀
