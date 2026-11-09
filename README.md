# Stroke Risk Prediction: Machine Learning on Imbalanced Clinical Data

An end-to-end machine learning project to predict stroke risk from clinical and demographic data. The dataset is heavily imbalanced (only ~4.3% positive cases), so the project focuses on a question that matters in healthcare ML: **what happens when accuracy is not the metric you should trust?**

The work is split into two stages:

1. **Part 0: Baseline pipeline.** Data cleaning, EDA, leakage-free preprocessing and a tuned Logistic Regression.
2. **Part 1: Model comparison.** SVM, Decision Trees, Random Forest, Gradient Boosting and AdaBoost, compared with a focus on recall for the minority (stroke) class.

## Key Findings

- **Accuracy is misleading.** An unbalanced Logistic Regression reached **96% accuracy while detecting 0% of stroke patients**. Random Forest showed the same collapse in Part 1 (95.6% accuracy, almost no stroke cases detected at the default threshold).
- **Class balancing changes everything.** With `class_weight='balanced'`, Logistic Regression reached **69% recall** on the stroke class (29 of 42 test cases), at the cost of precision (~11%).
- **Boosting handled the imbalance best.** AdaBoost detected **26 of 42 stroke cases (recall 0.619)**, followed by Gradient Boosting (22 of 42, recall 0.524). SVM, Decision Tree and Random Forest detected far fewer, despite higher accuracy.
- **Outliers can be signal.** Extreme glucose values (>200 mg/dL, consistent with severe diabetes) were deliberately kept, while BMI was log-transformed to compress extreme values.
- **Pipelines prevent leakage.** Scaling was encapsulated in a scikit-learn `Pipeline`, so it is fitted only on training folds during cross-validation.

## Dataset

[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (Kaggle): 5,110 patients, 12 features (age, average glucose level, BMI, hypertension, heart disease, smoking status, work type, etc.).

| Item | Value |
|---|---|
| Rows after cleaning | 4,908 (201 rows with missing BMI and 1 `other` gender row removed) |
| Class balance | 4,699 healthy / 209 stroke (4.26%) |
| Split | 80/20 stratified, `random_state=42` (3,926 train / 982 test, 42 stroke cases in test) |

<p align="center">
  <img src="Parte%200/images/class_distribution.png" alt="Class distribution" width="55%">
</p>

### Exploratory analysis

<p align="center">
  <img src="Parte%200/images/distribuciones_univariadas.png" alt="Feature distributions" width="48%">
  <img src="Parte%200/images/correlacion_matrix.png" alt="Correlation matrix" width="48%">
</p>

Feature correlations are weak (max 0.40, age vs. BMI), so multicollinearity is not a concern. Glucose is right-skewed with a second cluster above 200 mg/dL, which is why those values were kept.

<p align="center">
  <img src="Parte%200/images/boxplots.png" alt="Outlier detection boxplots" width="70%">
</p>

## Methodology

### Preprocessing
- Removed rows with missing BMI (<5% of data) instead of imputing.
- Log-transformed BMI to reduce skew; kept glucose outliers.
- Binary variables label-encoded; `smoking_status` and `work_type` one-hot encoded (no artificial ordering, since "formerly smoked" showed a higher stroke rate than "smokes" in this sample).
- `StandardScaler` applied inside the pipeline, after the train/test split.
- Intermediate dataset versions (clean, outliers treated, encoded) are saved in `Parte 0/versions/`.

### Part 0: Logistic Regression baseline
- `Pipeline(StandardScaler -> LogisticRegression(class_weight='balanced'))`
- `GridSearchCV` (5-fold) over `C` and `max_iter`; best: `C=10`, `max_iter=1000`.
- Compared balanced vs. unbalanced models and the successive dataset versions.

### Part 1: Supervised learning comparison
All models use class balancing (`class_weight='balanced'`, or `compute_sample_weight` for Gradient Boosting) and 5-fold `GridSearchCV`.

| Family | What was explored |
|---|---|
| **SVM** | Linear / Polynomial / RBF / Sigmoid kernels; `C`, `gamma`, degree; support vector analysis; `SVC` vs `LinearSVC` vs `SGDClassifier` |
| **Decision Trees** | Overfitting baseline, `max_depth`, Gini vs. Entropy, pre-pruning (grid search), post-pruning (cost-complexity), feature importance |
| **Random Forest** | `n_estimators`, `max_features`, OOB score, Gini vs. permutation importance, Extra Trees, decision threshold tuning |
| **Boosting** | Gradient Boosting (`n_estimators` x `learning_rate`, `max_depth`, `subsample`) and AdaBoost |

## Results

### Part 0: Logistic Regression (stroke class)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Unbalanced | 0.96 | 0.00 | 0.00 | 0.00 |
| **Balanced** | 0.76 | 0.11 | **0.69** | 0.19 |

<p align="center">
  <img src="Parte%200/images/confusion_matrix.png" alt="Logistic Regression confusion matrix" width="40%">
</p>

### Part 1: Best configuration per model family (test set)

| Model | Accuracy | Precision | Recall | Weighted F1 |
|---|---|---|---|---|
| SVM (RBF) | 0.9257 | 0.103 | 0.095 | 0.9243 |
| Decision Tree (post-pruned) | 0.8982 | 0.108 | 0.190 | 0.9113 |
| Random Forest | 0.9562 | 0.000 | 0.000 | 0.9358 |
| Gradient Boosting | 0.7953 | 0.108 | 0.524 | 0.8530 |
| **AdaBoost** | 0.7525 | 0.103 | **0.619** | 0.8254 |

> Accuracy and weighted F1 are dominated by the majority class. **Recall on the stroke class is the metric that matters here**, which is why AdaBoost is selected as the best model despite having the lowest accuracy.

<p align="center">
  <img src="Parte%201/images/accuracy_vs_recall.png" alt="Accuracy vs Recall by model" width="70%">
</p>

<p align="center">
  <img src="Parte%201/images/confusion_matrix_adaboost.png" alt="AdaBoost confusion matrix" width="40%">
  <img src="Parte%201/images/eficiencia.png" alt="Training time vs recall" width="55%">
</p>

AdaBoost confusion matrix on the test set: 713 TN, 227 FP, 16 FN, 26 TP. In a screening context, a false positive means extra tests, while a false negative means a missed stroke risk. The model deliberately trades precision for recall.

### Other observations
- Decision trees overfit heavily without constraints (train accuracy 1.0, 23 levels, 322 leaves); cost-complexity pruning removed ~100 leaves with almost no loss in F1.
- Gradient Boosting overfits as tree depth grows, which is why shallow base learners generalise better.

<p align="center">
  <img src="Parte%201/images/evolucion_sobreajuste_gb.png" alt="Overfitting in Gradient Boosting" width="55%">
</p>

- Lowering the Random Forest decision threshold to 0.15 recovered some stroke cases (7 of 42) at the cost of many more false positives.
- `age`, `avg_glucose_level` and `bmi_log` were the dominant features in both the single tree and the Random Forest.

## Repository Structure

```
.
├── README.md
├── Parte 0/                          # Baseline pipeline
│   ├── P0_AA.ipynb
│   ├── P0_AA_Memoria.pdf             # Written report (Spanish)
│   ├── healthcare-dataset-stroke-data.csv
│   ├── versions/                     # Dataset after each preprocessing step
│   │   ├── 01_dataset_clean.csv
│   │   ├── 02_dataset_outliers.csv
│   │   └── 03_dataset_encoded.csv
│   └── images/                       # EDA and result figures
└── Parte 1/                          # Model comparison
    ├── Practica1.ipynb
    ├── Memoria_P1.pdf                # Written report (Spanish)
    ├── 03_dataset_encoded.csv
    └── images/                       # Confusion matrices, comparison plots
```

## Getting Started

```bash
git clone https://github.com/izancuesta14/stroke-risk-prediction-ml.git
cd stroke-risk-prediction-ml
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

The datasets are already included, so you can open the notebooks directly:

1. `Parte 0/P0_AA.ipynb`: EDA, preprocessing and Logistic Regression baseline.
2. `Parte 1/Practica1.ipynb`: SVM, trees, Random Forest and boosting comparison.

**Stack:** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn.

## Limitations and Future Work

- Very few positive cases (209 total, 42 in test), so metrics have high variance. Repeated or nested cross-validation would give more reliable estimates.
- Precision is low (~10-11%) for every model; none is suitable for clinical use as-is.
- Hyperparameter search optimised accuracy/weighted F1 rather than recall on the stroke class. Tuning directly for recall, F1 (macro) or PR-AUC is the obvious next step.
- Ideas to explore: SMOTE or other resampling, threshold optimisation for every model, cost-sensitive learning, probability calibration, and gradient boosting libraries such as XGBoost or LightGBM.

## Author

**Izan Cuesta Corbí**
University of Alicante, Escuela Politécnica Superior
Contact: icc87@alu.ua.es

*Originally developed as coursework for the Advanced Learning course (2025/2026). The written reports are in Spanish.*
