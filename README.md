# Personality Cluster Classification: Machine Learning & Deep Learning Approach

**Team:** Unsupervised Learners
* Sawant Hrushikesh Rahul (IMT2023619)
* Akshat Mittal (IMT2023606)

---

## 1. Project Overview

This project focuses on a multiclass classification task aimed at predicting an individual's "Personality Cluster" (labeled A through E) based on anonymized behavioral and lifestyle data. The dataset includes features describing daily routines, social engagement, focus intensity, and upbringing influences.

The primary objective was to maximize the **Macro F1 Score** to account for significant class imbalance within the target variable, particularly the minority class (Cluster A). The final solution leverages Gradient Boosting (CatBoost) and Deep Learning (PyTorch) models, optimized via extensive hyperparameter tuning using Optuna.

## 2. Problem Statement

* **Problem Type:** Supervised Multiclass Classification
* **Target Variable:** `personality_cluster` (Classes: `Cluster_A` to `Cluster_E`)
* **Evaluation Metric:** The primary metric for model optimization and selection is the **Macro F1 Score** (to handle potential class imbalances).

## 3. Dataset

The dataset consists of features capturing diverse behavioral attributes.
* **`train.csv`**: Contains the training samples with features like `focus_intensity`, `consistency_score`, `support_environment_score`, and the target `personality_cluster`.
* **`test.csv`**: Contains the test samples for which predictions must be generated.

## 4. Methodology

Our approach utilizes a distinct preprocessing pipeline for different model architectures, followed by rigorous hyperparameter tuning using **Optuna**.

### 4.1. Preprocessing & Feature Engineering

1.  **Imputation & Handling:**
    * Missing values (if any) were handled based on the model type.
    * **CatBoost:** Utilized native support for categorical features (Ordered Boosting).
    * **Neural Network:** Used Label Encoding mapped to **Entity Embeddings** for categorical features.
    * **SVM:** Used **One-Hot Encoding** and **Standard Scaling**.

2.  **Feature Engineering:**
    * **Total Activity Score**: Aggregation of hobby, physical, creative, and altruistic engagement levels.
    * **Focus-Consistency Interaction**: A multiplicative feature capturing the relationship between `focus_intensity` and `consistency_score`.
    * **Support-Guidance Synergy**: Interaction between environmental support and the usage of external guidance.
    * **Efficiency Metrics**: Ratio-based features determining the efficiency of focus relative to consistency.
    * **Binning:** Discretized continuous variables like `focus_intensity` into bins.

### 4.2. Models and Architectures

We implemented and tuned three distinct architectures:

1.  **CatBoost Classifier:** Selected for its superior handling of categorical data and robustness on tabular datasets. Trained on GPU.
2.  **Tabular Neural Network (PyTorch):** A custom Multi-Layer Perceptron (MLP) utilizing:
    * Entity Embeddings for categorical features.
    * Batch Normalization and Dropout for regularization.
    * GELU activation functions.
    * Label Smoothing Loss to handle noisy labels.
3.  **Support Vector Machine (SVM):** Implemented as a baseline to capture non-linear decision boundaries using an RBF kernel.

### Hyperparameter Optimization
All models were rigorously tuned using **Optuna** with the Tree-structured Parzen Estimator (TPE) sampler.
* **Validation Strategy**: Stratified K-Fold Cross-Validation (5 Folds) was used to ensure stability and prevent overfitting to the majority class.
* **Metric**: Optimization was directed toward maximizing the Macro F1 Score.

## 4. Performance Results

The models were evaluated based on their validation Macro F1 Scores. The custom Neural Network performed exceptionally well, likely due to the effectiveness of entity embeddings and global optimization via 600+ trials.

## 5. Requirements and Installation

The project requires Python and the following libraries. GPU acceleration requires CUDA-enabled versions of PyTorch and CatBoost.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn optuna catboost torch torchvision
