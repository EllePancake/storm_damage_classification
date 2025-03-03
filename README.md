# storm_damage_classification




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

---

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
