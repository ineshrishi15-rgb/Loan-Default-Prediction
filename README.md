# Loan Default Prediction Using Machine Learning

## Project Overview

This project develops a machine learning classification model to predict loan default risk using historical bank-loan applicant data. It uses Gaussian Naive Bayes and evaluates different class-imbalance handling techniques to improve the identification of potential loan defaulters.

The project focuses on data preprocessing, outlier treatment, multicollinearity reduction, model evaluation, and prediction of applicants with unknown loan outcomes.

## Objectives

- Analyse historical loan applicant data.
- Build a Gaussian Naive Bayes classification model.
- Reduce multicollinearity using Variance Inflation Factor (VIF).
- Handle class imbalance using different resampling techniques.
- Compare model performance using five-fold cross-validation.
- Generate default predictions and probabilities for unlabelled applicants.

## Tools & Technologies

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, imbalanced-learn, Statsmodels
- **Environment:** Jupyter Notebook
- **Machine Learning:** Gaussian Naive Bayes
- **Evaluation:** Accuracy, Precision, Recall, F1-score, Cross-validation

## Dataset

The project uses bank-loan applicant data containing:

- **700 labelled applicants** for model training and evaluation.
- **150 unlabelled applicants** for generating predictions.

The target variable represents whether an applicant defaulted on their loan.

## Project Workflow

### 1. Data Preprocessing
- Separated labelled and unlabelled applicant records.
- Handled missing values and outliers.
- Applied percentile-based outlier capping using training data to avoid data leakage.

### 2. Train-Test Split
- Used an 80:20 stratified train-test split.
- Preserved the class distribution across training and testing datasets.

### 3. Feature Selection
- Calculated Variance Inflation Factor (VIF).
- Applied iterative feature selection to reduce multicollinearity.
- Retained relevant predictors for model development.

### 4. Model Development
- Implemented a Gaussian Naive Bayes classifier.
- Established a baseline model.
- Evaluated different class-imbalance handling approaches:
  - Random Oversampling
  - Random Undersampling
  - SMOTE

### 5. Model Evaluation
- Used five-fold cross-validation to compare resampling techniques.
- Evaluated minority-class F1-score.
- Assessed final model performance using accuracy, precision, recall and F1-score.

### 6. Prediction
- Generated default predictions for 150 unlabelled applicants.
- Produced prediction probabilities to support risk assessment.

## Model Performance

The baseline model achieved approximately **76% test accuracy**, but its recall for the default class was only **39%**.

This highlighted the limitations of accuracy as a standalone evaluation metric when dealing with imbalanced classification data.

The project therefore compares resampling strategies using cross-validation, with particular emphasis on improving the identification of actual loan defaulters.

## Key Learning Outcomes

- Practical implementation of supervised machine learning.
- Handling imbalanced classification datasets.
- Feature selection using statistical techniques.
- Understanding and preventing data leakage.
- Model evaluation using cross-validation.
- Interpreting precision-recall trade-offs in credit-risk classification.

## Future Improvements

- Experiment with additional classification algorithms.
- Perform hyperparameter tuning.
- Explore ensemble-based classification models.
- Improve default detection while controlling false positives.
- Evaluate model calibration and business impact.

## Disclaimer

This project is intended for educational and portfolio purposes only. Predictions should not be used as the sole basis for real-world lending or credit decisions.
