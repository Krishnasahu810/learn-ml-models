# 📊 Feature Scaling & Normalization

> A practical guide to Feature Scaling, Standardization, Normalization, and Robust Scaling in Machine Learning.

## 🧠 What is Feature Scaling?

Feature scaling transforms numerical features so that they are on comparable scales.

Example:

| Feature | Example Range |
|---|---:|
| Age | 18–80 |
| Salary | ₹20,000–₹2,00,000 |
| Experience | 0–40 years |
| Score | 0–100 |

Without scaling, some algorithms can give excessive influence to features with larger numerical values.

## 🎯 Why Do We Need Scaling?

Scaling is especially important for:

- K-Nearest Neighbors (KNN)
- K-Means
- DBSCAN
- Support Vector Machines (SVM)
- Logistic Regression
- Neural Networks

## 📐 Standardization — Z-Score

Standardization generally transforms data to have mean ≈ 0 and standard deviation ≈ 1.

### Formula

$$
z = \\frac{x - \\mu}{\\sigma}
$$

### Python

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 📏 Min-Max Normalization

Min-Max Scaling usually transforms values into the range 0 to 1.

### Formula

$$
x' = \\frac{x - x_{min}}{x_{max} - x_{min}}
$$

### Python

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

## 🛡️ Robust Scaling

RobustScaler uses the median and Interquartile Range (IQR), making it less sensitive to outliers.

### Formula

$$
x' = \\frac{x - Median}{IQR}
$$

### Python

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

## ⚖️ Scaling Comparison

| Technique | Uses | Outlier Sensitivity | Typical Range |
|---|---|---|---|
| StandardScaler | Mean + Standard Deviation | Higher | Not fixed |
| MinMaxScaler | Minimum + Maximum | Higher | 0–1 |
| RobustScaler | Median + IQR | Lower | Not fixed |

## 📊 Visual Concept

```text
Raw Features
     │
     ▼
Different Units & Ranges
     │
     ▼
Feature Scaling
     │
     ├───────────────┬───────────────┐
     ▼               ▼               ▼
StandardScaler   MinMaxScaler   RobustScaler
     │               │               │
     └───────────────┴───────────────┘
                     │
                     ▼
              Comparable Scale
                     │
                     ▼
               Machine Learning
```

## 🍷 Wine Dataset

This module contains the `wine_data.csv` dataset and a Jupyter Notebook demonstrating feature scaling and normalization.

### Dataset workflow

```text
wine_data.csv
      ↓
Load Dataset
      ↓
Explore Data
      ↓
Select Numerical Features
      ↓
Apply Scaling
      ↓
Compare Results
      ↓
Machine Learning Model
```

Load the dataset with:

```python
import pandas as pd

df = pd.read_csv('wine_data.csv')
df.head()
```

## 🚨 Avoid Data Leakage

Always fit the scaler on training data only.

### ❌ Incorrect

```python
X_test_scaled = scaler.fit_transform(X_test)
```

### ✅ Correct

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler learns its parameters from the training data and then applies them to the test data.

## 🤖 Algorithms and Scaling

### Scaling is usually important for:

- KNN
- K-Means
- DBSCAN
- SVM
- Logistic Regression
- Neural Networks

### Scaling is usually less important for:

- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost

## 🧠 Quick Revision

```text
Standardization
     ↓
Mean ≈ 0, Standard Deviation ≈ 1

Min-Max Normalization
     ↓
Usually 0 → 1

Robust Scaling
     ↓
Median + IQR
     ↓
Useful with outliers
```

## 📁 Project Structure

```text
Feature_Scaling_Normalization/
│
├── README.md
├── feature_scaling_normalization.ipynb
└── wine_data.csv
```

## 🚀 Learning Path

```text
Python
  ↓
NumPy & Pandas
  ↓
Data Visualization
  ↓
Feature Engineering
  ↓
Feature Scaling & Normalization  ← You are here
  ↓
Machine Learning
  ↓
Model Evaluation
  ↓
Real-World ML Projects
```

## ✅ Key Takeaways

- Feature scaling puts numerical features on comparable scales.
- StandardScaler uses mean and standard deviation.
- MinMaxScaler maps values to a chosen range.
- RobustScaler uses median and IQR.
- Scaling is important for many distance-, gradient-, and margin-based algorithms.
- Tree-based models generally do not require scaling.
- Always fit preprocessing steps on training data only.

## 📚 Resources

- https://scikit-learn.org/stable/modules/preprocessing.html
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html

---

⭐ Part of the `learn-ml-models` Machine Learning learning repository.

Keep learning. Keep experimenting. Keep building. 🚀