# Storm Damage Classification using ResNet50

## Overview
This project applies **deep learning techniques** to classify **storm damage** using **satellite imagery**. It utilizes **ResNet50** and **transfer learning** to automate the assessment of property damage, aligning with CAPE Analytics' mission of **computer vision-driven property intelligence.**

## **Folder & File Structure**
```
📦 modeling
 ┣ 📂 experiments
 ┃ ┣ 📜 train_v1.ipynb
 ┃ ┣ 📜 train_v2.ipynb
 ┃ ┣ 📜 train_v3.ipynb
 ┃ ┣ 📜 train_v4.ipynb
 ┃ ┗ 📜 compare_models.ipynb
 ┣ 📂 models
 ┃ ┣ 📜 base_model.py          # Base ResNet50 model (shared structure)
 ┃ ┣ 📜 model_v1.py            # First version (baseline)
 ┃ ┣ 📜 model_v2.py            # Modified hyperparameters
 ┃ ┣ 📜 model_v3.py            # Adding dropout, augmentation, optimizers
 ┃ ┗ 📜 model_v4.py            # Best-performing model
 ┣ 📂 results
 ┃ ┣ 📜 evaluation.py          # Model performance evaluation
 ┃ ┣ 📜 evaluation_metrics.csv # Stores evaluation results for different versions
 ┃ ┣ 📜 metrics.csv            # Tracks metrics during training
 ┃ ┗ 📜 model_comparisons.csv  # Compare multiple versions
 ┣ 📂 training
 ┃ ┣ 📜 trainer.py             # Training loop
 ┃ ┣ 📜 evaluator.py           # Evaluation & predictions
 ┃ ┗ 📜 interpret.py           # Grad-CAM & misclassification analysis
 ┣ 📂 utils
 ┃ ┣ 📜 metrics.py             # Custom metrics (F1-score, precision, recall)
 ┃ ┣ 📜 utils_data_loader.py   # Loads datasets with augmentation
 ┃ ┗ 📜 utils_visualization.py # Plots training curves, confusion matrices
 ┣ 📜 EDA.ipynb                # Exploratory Data Analysis
 ┣ 📜 preprocessing.ipynb      # Data cleaning, augmentation
 ┣ 📜 results_and_conclusion.ipynb # Model comparison & insights
 ┣ 📜 utils.py                 # Helper functions
 ┗ 📜 README.md                # Project documentation
```
## **Description of Files**
| **File** | **Description** |
|----------|----------------|
| `train_v1.ipynb - train_v4.ipynb` | Jupyter notebooks for different model versions |
| `compare_models.ipynb` | Final analysis comparing all models |
| `base_model.py` | Defines a general ResNet50 architecture to be used across versions |
| `model_v1.py` | Baseline ResNet50 model |
| `model_v2.py` | Modified learning rate, optimizer |
| `model_v3.py` | Adds dropout, different augmentations |
| `model_v4.py` | Best-performing model |
| `trainer.py` | Centralized training script (modular, reusable) |
| `evaluator.py` | Evaluates trained models & generates metrics |
| `interpret.py` | Uses Grad-CAM to interpret model decisions |
| `metrics.py` | Computes precision, recall, F1-score |
| `utils_data_loader.py` | Loads & preprocesses dataset |
| `utils_visualization.py` | Plots training/validation loss & accuracy |
| `EDA.ipynb` | Dataset analysis |
| `preprocessing.ipynb` | Data cleaning, augmentation & preparation |
| `results_and_conclusion.ipynb` | Summary of findings & best model selection |
| `utils.py` | Helper functions |
| `README.md` | Project overview & instructions |

---

## 🚀 Model Variants
| Model Version | Key Changes |
|--------------|------------|
| **Model v1** | Baseline ResNet50 (pretrained, frozen base, no fine-tuning) |
| **Model v2** | Enables fine-tuning, lower learning rate |
| **Model v3** | Uses AdamW optimizer, higher dropout (0.3) for regularization |
| **Model v4** | Adds learning rate scheduling (cosine decay), stronger regularization |

## 📊 Dataset
- **Images**: Preprocessed satellite imagery of storm-affected areas.
- **Labels**: `damage` vs. `no_damage`
- **Preprocessing Steps**: (Details to be added after training)

## 🔧 Training Pipeline
1. **Data Preprocessing & Augmentation** (`utils_data_loader.py`)
2. **Model Training** (`train_v1.py`, `train_v2.py`, etc.)
3. **Evaluation & Visualization** (`evaluation.py`, `utils_visualization.py`)
4. **Model Interpretation (Grad-CAM)** (`interpret.py`)

## 📈 Expected Results
- Model **performance metrics** (accuracy, F1-score) will be added **after training is complete**.
- The best-performing model will be identified and compared across versions.

## 📌 Next Steps
- Train all models and log performance.
- Analyze results using `compare_models.ipynb`.
---

This project demonstrates advanced **deep learning for geospatial analysis** and is structured to be scalable, reproducible, and modular. 
