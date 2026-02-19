# Automated Aerial Scene Classification
A comparative study of classical machine learning and deep learning approaches for urban planning, environmental monitoring, and disaster response - applied to the Intel Image Classification dataset.



## Project Overview
This project investigates the extent to which machine learning models can reliably classify natural and urban scene images into six categories: **buildings, forest, glacier, mountain, sea, and street**. The motivation is grounded in real-world demand for automated aerial and satellite image interpretation systems that can support urban planners, environmental agencies, and disaster response teams at a scale that manual analysis cannot match.

The pipeline systematically compares four model families across 21 experiments:
- Random Forest (7 experiments) - classical ML with hand-crafted features
- Baseline CNN (7 experiments) - custom deep learning architecture
- EfficientNetB0 (7 experiments) - transfer learning from ImageNet



## Dataset
**Intel Image Classification**  
Source: [Kaggle — puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

| Property | Value |
|---|---|
| Total images | ~17,000 |
| Number of classes | 6 |
| Image size (original) | 150 x 150 px |
| Image size (used) | 128 x 128 px |
| Class balance | Well balanced (~2,300 per class) |

**Classes:** buildings, forest, glacier, mountain, sea, street

The dataset is downloaded automatically at runtime via `kagglehub` — no manual download required.





---

## Notebook Structure
| Section | Content |
|---|---|
| 0 | Environment setup, imports, random seeds |
| 1 | Data loading via kagglehub, class distribution |
| 2 | Preprocessing, tf.data pipeline, augmentation visualisation |
| 3 | Feature extraction for classical ML (colour histograms, LBP, HOG) |
| 4 | Evaluation utilities and logging functions |
| 5 | Random Forest — 7 experiments |
| 6 | Deep learning utilities, CNN architecture, training functions |
| 8 | Baseline CNN — 7 experiments |
| 9 | Transfer learning with EfficientNetB0 — 7 experiments |
| 10 | Cross-model evaluation, ROC curves, confusion matrices |
| 11 | Conclusion and critical analysis |

---

## Methods

### Preprocessing
- Images resized to 128 x 128 pixels
- Pixel values normalised to [0, 1]
- Stratified 70 / 15 / 15 train / validation / test split

### Data Augmentation (training only)
Applied via the `tf.data` API with `seed=42` for reproducibility:
- Random horizontal and vertical flips
- Random rotation (±15%)
- Random zoom (±10%)
- Random brightness and contrast variation (±8%)
- Output clipped to [0, 1] after augmentation

### Classical ML Feature Extraction
- **Colour histograms** — 64-bin histograms across RGB and HSV channels
- **LBP (Local Binary Pattern)** — uniform texture descriptor with radius 3, 24 points
- **HOG (Histogram of Oriented Gradients)** — shape and edge features

Features are cached to disk after first extraction. Subsequent runs load from cache instantly.

### Classical ML Models
| Model | Library | Hyperparameters varied |
|---|---|---|
| Random Forest | Scikit-learn | n_estimators, max_depth, min_samples_split, max_features, class_weight |


### Deep Learning Models
| Model | Framework | Description |
|---|---|---|
| Baseline CNN | TensorFlow / Keras | 3 Conv blocks + GlobalAveragePooling + Dense head |
| EfficientNetB0 | TensorFlow / Keras | ImageNet pretrained, fine-tuned top N layers |

All deep learning experiments use `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, and `CSVLogger`.

---

## Evaluation Metrics
- Accuracy
- Weighted Precision, Recall, F1-score (via Scikit-learn `classification_report`)
- ROC-AUC (one-vs-rest, weighted average)
- Confusion matrix per model
- Learning curves (training vs validation accuracy and loss)
- Feature importance (Random Forest)

---

## Reproducibility
- All random seeds fixed: Python, NumPy, TensorFlow, `PYTHONHASHSEED`
- Augmentation pipeline uses `seed=42`
- Feature extraction results cached to disk
- Model checkpoints saved after each epoch
- All 28 experiments logged to CSV with timestamps and full hyperparameter configurations
- Notebook runs top-to-bottom without modification in Google Colab

---

## Requirements
```
pip install kagglehub scikit-image scikit-learn matplotlib seaborn tqdm opencv-python-headless
```

---

## How to Run
1. Open the notebook in Google Colab
2. Run all cells from top to bottom
3. The `kagglehub.dataset_download` cell authenticates automatically if you are logged into Kaggle in your browser
4. All outputs are saved to `intel_scene_classification/` within the Colab session
5. To persist outputs beyond the session, mount Google Drive and copy the directory there

---

## Acknowledgements
- Dataset: Intel Image Classification, published on Kaggle by Puneet Bansal
- Pretrained weights: EfficientNetB0 trained on ImageNet via `tf.keras.applications`
- Libraries: TensorFlow, Keras, Scikit-learn, Scikit-image, OpenCV, NumPy, Pandas, Matplotlib, Seaborn
