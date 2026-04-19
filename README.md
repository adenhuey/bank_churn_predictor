# bank_churn_predictor
Random Forest Classifier of on Bank Churn Dataset

# Bank Customer Churn Classifier

A machine learning classification project built on the [Kaggle Playground Series S4E1](https://www.kaggle.com/competitions/playground-series-s4e1) dataset. The goal is to predict whether a bank customer will churn (leave the bank) based on demographic and account features.

---

## Dataset

- **Source:** Kaggle Playground Series — Season 4, Episode 1
- **Train size:** 165,034 rows
- **Features:** 10 (after dropping identifiers)
- **Target:** `Exited` — binary (0 = stayed, 1 = churned)
- **Class balance:** ~79% non-churn, ~21% churn (imbalanced)

### Features Used

| Feature | Description |
|---|---|
| CreditScore | Customer credit score |
| Geography | Country (France, Germany, Spain) |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years as a customer |
| Balance | Account balance |
| NumOfProducts | Number of bank products held |
| HasCrCard | Whether customer has a credit card |
| IsActiveMember | Whether customer is an active member |
| EstimatedSalary | Estimated annual salary |

---

## Project Workflow

1. **Load & explore data** — `df.info()`, `df.describe()`, null checks
2. **Preprocessing** — dropped identifier columns (`id`, `CustomerId`, `Surname`), one-hot encoded `Geography` and `Gender`
3. **EDA** — checked class balance, distributions, correlations
4. **Train/val split** — 80/20 split with `stratify=y` from `train.csv`
5. **Modeling** — trained and tuned multiple classifiers
6. **Evaluation** — compared models on accuracy, F1, precision, recall, ROC-AUC

---

## Models & Results

| Model | Accuracy |
|---|---|
| **Random Forest (tuned)** | **86.44%** |
| KNN (extended, k=29) | 85.84% |
| KNN (tuned) | 85.80% |
| Random Forest (iteration 2) | 85.58% |
| KNN (baseline) | 84.64% |
| Logistic Regression | 83.40% |

**Best model: Random Forest (first iteration)** — 86.44% accuracy, 89% ROC-AUC, 74% precision on churners.

### Random Forest — Best Parameters
```
n_estimators: 200
min_samples_split: 5
min_samples_leaf: 0.0001
max_depth: None
class_weight: balanced
```

### Classification Report (Random Forest)
```
              precision    recall  f1-score   support

           0       0.89      0.95      0.92     26084
           1       0.74      0.54      0.62      6923

    accuracy                           0.86     33007
   macro avg       0.82      0.74      0.77     33007
weighted avg       0.86      0.86      0.86     33007
```

---

## Key Findings

- **Age** was the strongest predictor of churn — older customers were significantly more likely to leave
- **Geography** mattered — German customers churned at a higher rate than French or Spanish customers
- **IsActiveMember** was highly predictive — inactive members were far more likely to churn
- **NumOfProducts** showed a non-linear relationship — customers with 3–4 products churned at much higher rates than those with 1–2
- Class imbalance (~79/21 split) required `class_weight='balanced'` to prevent the model from defaulting to majority class predictions

---

## Tech Stack

- Python 3.14
- pandas, numpy
- scikit-learn (RandomForestClassifier, LogisticRegression, KNeighborsClassifier)
- matplotlib
- Jupyter Notebook

---

## How to Run

```bash
git clone https://github.com/yourusername/bank-churn-classifier
cd bank-churn-classifier
pip install -r requirements.txt
jupyter notebook bank_churn_classifier.ipynb
```

Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/playground-series-s4e1/data) and place them in the project root.
