# Intel Image Scene Classification: Traditional ML vs Deep Learning

A comparative study across 21 experiments - Random Forest, CNN, and EfficientNetB0 Transfer Learning - on the Intel Image Classification dataset (6 classes, 25k images).

## Results Summary

| Model | Best Experiment | Test Accuracy | Mean AUC |
|---|---|---|---|
| Random Forest | RF Exp 7 – PCA + Depth + Balanced | 71.03% | 0.93 |
| CNN | CNN Exp 7 – InceptionV3 Frozen | 92.17% | 0.99 |
| Transfer Learning | TL Exp 7 – Full Fine-Tuning | 89.80% | 0.99 |

## Project Structure
```
├── Intel_Image_Classification_Full_Comparison.ipynb   # Main notebook (21 experiments)
└── README.md             
```

## Setup
```bash
pip install kagglehub scikit-image scikit-learn tensorflow opencv-python
```

## How to Run

1. Open the notebook in Google Colab
2. Set runtime to **GPU (T4 recommended)**
3. Run all cells top to bottom - no manual setup required

The dataset downloads automatically in Cell 1. All 21 experiments run sequentially.

## Data Split

| Split | Source | Size |
|---|---|---|
| Train | 80% of `seg_train` | ~11,227 images |
| Validation | 20% of `seg_train` | ~2,807 images |
| Test | `seg_test` (held-out) | 3,000 images |

`seg_test` is never used during training or hyperparameter tuning.

## Key Findings

- **InceptionV3 frozen features** outperformed all other models including EfficientNetB0
- **EfficientNetB0 partial freezing failed** due to BatchNorm miscalibration - only full fine-tuning (89.80%) produced competitive results
- **Random Forest** with HOG + colour histogram features reached 71.03%; the 2.5-point spread across 7 experiments confirms features, not hyperparameters, were the bottleneck
- **Glacier ↔ Mountain** was the dominant confusion pair across every model and every experiment

## Dependencies

- Python 3.10+
- TensorFlow 2.x
- scikit-learn
- scikit-image (for HOG)
- OpenCV
- kagglehub
- pandas, numpy, matplotlib, seaborn
