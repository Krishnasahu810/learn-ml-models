# 📊 Feature Scaling & Standardization

Feature Scaling is an important Machine Learning preprocessing technique.
It transforms numerical features so that they are on a comparable scale.

---

# 📚 Table of Contents

1. What is Feature Scaling?
2. Why Feature Scaling is Required
3. What is Standardization?
4. Z-Score
5. Standardization Formula
6. Mathematical Example
7. StandardScaler
8. fit(), transform() and fit_transform()
9. Train-Test Split
10. Data Leakage
11. Min-Max Scaling
12. Robust Scaling
13. Standardization vs Normalization
14. Algorithms That Need Scaling
15. Algorithms That Usually Don't Need Scaling
16. KNN Example
17. Logistic Regression Example
18. SVM Example
19. K-Means Example
20. PCA Example
21. Social Network Ads Example
22. Pipeline
23. Scaling New Data
24. Saving the Scaler
25. Common Mistakes
26. Interview Questions
27. Quick Revision

---

# 1. What is Feature Scaling?

Feature Scaling means transforming numerical features so that their values are on a similar scale.

Example:

| Feature | Example Range |
|---|---:|
| Age | 18 - 60 |
| Salary | 15,000 - 150,000 |

Age and Salary have very different numerical ranges.

Some Machine Learning algorithms are sensitive to these differences.

---

# 2. Why Feature Scaling is Required?

Suppose we have:

```text
Age = 30
Salary = 100000
```

If an algorithm calculates distance, Salary can dominate because its numerical values are much larger.

After scaling:

```text
Age    → approximately -2 to +2
Salary → approximately -2 to +2
```

Both features now have comparable magnitudes.

---

# 3. What is Standardization?

Standardization is a feature scaling technique that transforms data so that the feature has approximately:

- Mean = 0
- Standard Deviation = 1

It is also called **Z-Score Standardization**.

---

# 4. What is Z-Score?

A Z-score tells us how many standard deviations a value is away from the mean.

```text
z = 0
```

means the value is at the mean.

```text
z = 1
```

means the value is one standard deviation above the mean.

```text
z = -1
```

means the value is one standard deviation below the mean.

---

# 5. Standardization Formula

The formula is:

```text
z = (x - μ) / σ
```

Where:

- x = original value
- μ = mean
- σ = standard deviation
- z = standardized value

---

# 6. Mathematical Example

Consider:

```text
X = [10, 20, 30, 40, 50]
```

Mean:

```text
μ = 30
```

Population standard deviation is approximately:

```text
σ = 14.14
```

For x = 10:

```text
z = (10 - 30) / 14.14
z ≈ -1.41
```

For x = 30:

```text
z = (30 - 30) / 14.14
z = 0
```

For x = 50:

```text
z = (50 - 30) / 14.14
z ≈ 1.41
```

Therefore:

```text
Original:
10    20    30    40    50

Standardized:
-1.41  -0.71  0  0.71  1.41
```

---

# 7. StandardScaler

Scikit-learn provides `StandardScaler` for standardization.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

Standardize data:

```python
X_scaled = scaler.fit_transform(X)
```

---

# 8. fit(), transform() and fit_transform()

## fit()

`fit()` learns the statistics from the data.

```python
scaler.fit(X_train)
```

For StandardScaler, it learns the mean and standard deviation.

## transform()

`transform()` uses the learned statistics.

```python
X_test_scaled = scaler.transform(X_test)
```

## fit_transform()

`fit_transform()` performs both operations.

```python
X_train_scaled = scaler.fit_transform(X_train)
```

---

# 9. Train-Test Split

The correct order is:

```text
Dataset
   ↓
Train-Test Split
   ↓
Fit Scaler on Training Data
   ↓
Transform Training Data
   ↓
Transform Test Data
```

Example:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

# 10. Data Leakage

Data leakage occurs when information from the test data influences the training process.

❌ Wrong:

```python
X_scaled = scaler.fit_transform(X)

X_train, X_test = train_test_split(X_scaled)
```

❌ Also wrong:

```python
X_test_scaled = scaler.fit_transform(X_test)
```

✅ Correct:

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Golden Rule:**

```text
FIT       → Training Data
TRANSFORM → Training Data
TRANSFORM → Test Data
TRANSFORM → New Data
```

---

# 11. Min-Max Scaling

Min-Max Scaling transforms values into a fixed range, usually 0 to 1.

Formula:

```text
X_scaled = (X - X_min) / (X_max - X_min)
```

Example:

```text
X = [10, 20, 30, 40, 50]
Min = 10
Max = 50
```

For 30:

```text
(30 - 10) / (50 - 10)
= 20 / 40
= 0.5
```

Python:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

---

# 12. Robust Scaling

RobustScaler is useful when the dataset contains outliers.

It uses:

- Median
- Interquartile Range (IQR)

Example:

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

Example data with an outlier:

```text
10
11
12
13
14
1000
```

The value 1000 can strongly affect the mean and standard deviation.

RobustScaler is less sensitive to extreme values.

---

# 13. Standardization vs Normalization

| Feature | Standardization | Min-Max Scaling |
|---|---|---|
| Formula | `(x-mean)/std` | `(x-min)/(max-min)` |
| Mean | Approximately 0 | Not necessarily 0 |
| Std | Approximately 1 | Not necessarily 1 |
| Range | No fixed range | Usually 0 to 1 |
| Outlier sensitivity | High | High |

Standardization is commonly used when algorithms benefit from centered features.

Min-Max Scaling is useful when a bounded range is desirable.

---

# 14. Algorithms That Benefit From Scaling

Feature scaling is especially important for:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- K-Means Clustering
- Logistic Regression
- Linear Regression
- Neural Networks
- PCA

The reason depends on the algorithm.

Distance-based algorithms are especially sensitive to feature scale.

---

# 15. Algorithms That Usually Do Not Need Scaling

Tree-based algorithms generally do not require feature scaling.

Examples:

- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

These algorithms generally split data using feature thresholds.

---

# 16. KNN Example

KNN calculates distances between observations.

Example:

```text
Age = 25
Salary = 50000
```

If Salary is not scaled, it can dominate the distance calculation.

Therefore KNN normally benefits from feature scaling.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train_scaled, y_train)
```

---

# 17. Logistic Regression Example

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
```

Scaling can help optimization behave better when feature magnitudes are very different.

---

# 18. SVM Example

SVM is sensitive to feature scale.

```python
from sklearn.svm import SVC

model = SVC()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
```

---

# 19. K-Means Example

K-Means uses distances to assign observations to clusters.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init='auto'
)

kmeans.fit(X_scaled)
```

---

# 20. PCA Example

PCA is affected by feature variance.

If one feature has much larger variance because of its units, it can dominate PCA.

Therefore standardization is commonly performed before PCA.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

---

# 21. Social Network Ads Example

This folder contains:

```text
Social_Network_Ads.csv
```

Important features:

- Age
- EstimatedSalary

Target:

- Purchased

The goal is to predict whether a person purchases a product based on Age and EstimatedSalary.

## Load Dataset

```python
import pandas as pd

df = pd.read_csv('Social_Network_Ads.csv')

print(df.head())
print(df.info())
```

## Select Features

```python
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
```

## Split Data

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)
```

## Standardize

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Train Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_scaled, y_train)
```

## Predict

```python
y_pred = model.predict(X_test_scaled)

print(y_pred)
```

## Evaluate

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)
```

---

# 22. Pipeline

Pipeline combines preprocessing and model training.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
```

Advantages:

- Reduces data leakage risk
- Keeps preprocessing consistent
- Makes code cleaner
- Works well with cross-validation
- Useful for deployment

---

# 23. Scaling New Data

Suppose a new customer has:

```text
Age = 35
EstimatedSalary = 75000
```

Use the SAME scaler that was fitted during training.

```python
new_customer = [[35, 75000]]

new_customer_scaled = scaler.transform(new_customer)

prediction = model.predict(new_customer_scaled)

print(prediction)
```

Do not create and fit a new scaler for the new customer.

---

# 24. Saving the Scaler

For production Machine Learning applications, save the scaler.

```python
import joblib

joblib.dump(scaler, 'standard_scaler.pkl')
```

Load it later:

```python
scaler = joblib.load('standard_scaler.pkl')
```

Then:

```python
new_data_scaled = scaler.transform(new_data)
```

The same training scaler should be used for future predictions.

---

# 25. Common Mistakes

## Mistake 1: Scaling Before Splitting

❌ Wrong:

```python
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)
```

✅ Correct:

```python
X_train, X_test = train_test_split(X)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Mistake 2: Fitting on Test Data

❌ Wrong:

```python
X_test_scaled = scaler.fit_transform(X_test)
```

✅ Correct:

```python
X_test_scaled = scaler.transform(X_test)
```

## Mistake 3: Creating a New Scaler for New Data

❌ Wrong:

```python
new_scaler = StandardScaler()
new_data_scaled = new_scaler.fit_transform(new_data)
```

✅ Correct:

```python
new_data_scaled = scaler.transform(new_data)
```

---

# 26. Interview Questions

### Q1. What is Feature Scaling?

Feature Scaling is the process of transforming numerical features to a comparable scale.

### Q2. What is Standardization?

Standardization transforms features to approximately zero mean and unit standard deviation.

### Q3. What is the formula?

```text
z = (x - μ) / σ
```

### Q4. Which sklearn class is used?

```python
StandardScaler
```

### Q5. Does StandardScaler produce values between 0 and 1?

No.

StandardScaler produces values centered around zero with unit standard deviation.

### Q6. Which scaler usually produces values between 0 and 1?

```python
MinMaxScaler
```

### Q7. Should we fit the scaler on test data?

No.

Fit the scaler only on training data.

### Q8. Why does KNN need scaling?

KNN uses distance calculations, so features with larger numerical scales can dominate.

### Q9. Is StandardScaler affected by outliers?

Yes. It uses mean and standard deviation.

### Q10. Which scaler is more robust to outliers?

```python
RobustScaler
```

### Q11. Do Decision Trees require scaling?

Usually no.

### Q12. What is data leakage?

Data leakage occurs when information that should be unavailable during training influences the training process.

---

# 27. Quick Revision

## Standardization

```text
Mean ≈ 0
Standard Deviation ≈ 1
```

## Formula

```text
z = (x - μ) / σ
```

## StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Golden Rule

```text
FIT       → Training Data
TRANSFORM → Training Data
TRANSFORM → Test Data
TRANSFORM → New Data
```

---

# 📌 StandardScaler vs MinMaxScaler vs RobustScaler

| Scaler | Main Idea | Outlier Sensitivity |
|---|---|---|
| StandardScaler | Mean 0, Std 1 | High |
| MinMaxScaler | Usually 0 to 1 | High |
| RobustScaler | Median + IQR | Lower |

---

# 🎯 Final Takeaways

1. Feature scaling makes numerical features comparable.
2. Standardization is also called Z-score standardization.
3. StandardScaler produces approximately mean 0 and standard deviation 1.
4. StandardScaler does not force values between 0 and 1.
5. Always split the dataset before fitting the scaler.
6. Fit the scaler only on training data.
7. Use transform() for test and new data.
8. KNN, SVM, K-Means and PCA commonly benefit from scaling.
9. Tree-based models usually do not require scaling.
10. RobustScaler can be useful when significant outliers are present.
11. Pipeline is a good way to combine scaling and modeling.
12. Use the same fitted scaler when making predictions on new data.

---

# 📁 Folder Structure

```text
Standardization/
│
├── README.md
├── feature scaling standardization.ipynb
└── Social_Network_Ads.csv
```

---

# 🚀 Learning Path

After Feature Scaling and Standardization, continue with:

1. Data Preprocessing
2. Missing Values
3. Encoding
4. Outlier Detection
5. Feature Engineering
6. Linear Regression
7. Logistic Regression
8. KNN
9. SVM
10. Decision Trees
11. Random Forest
12. PCA
13. Cross Validation
14. Hyperparameter Tuning
15. Model Evaluation

---

# ⭐ Remember

```text
TRAINING DATA
     ↓
FIT SCALER
     ↓
TRANSFORM TRAINING DATA
     ↓
TRANSFORM TEST DATA
     ↓
TRAIN MODEL
     ↓
SAVE SCALER + MODEL
     ↓
NEW DATA
     ↓
SAME SCALER
     ↓
PREDICTION
```

**Never fit the scaler on the test dataset.**