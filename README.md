# IBM Employee Performance Prediction

Utilizing IBM’s HR Analytics dataset to uncover what influences employee performance scores and building a machine learning model to predict them.

---

## 📊 Dataset
- **Source**: [IBM HR Analytics Employee Attrition & Performance Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Size**: 1470 rows × 35 columns
- **Target Variable**: `PerformanceRating`

---

## 📌 Objective
- Understand what drives employee performance ratings.
- Visualize trends and disparities in compensation and ratings.
- Build a predictive model to classify employee performance.

---

## 🧪 Tools & Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
```

---

## 🛠️ Load & Prepare Data
```python
# Load the dataset
df = pd.read_csv("IBM-HR-Employee-Attrition.csv")

# Drop irrelevant columns
df.drop(columns=["EmployeeNumber", "Over18", "StandardHours", "EmployeeCount"], inplace=True)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
```

---

## 📈 Exploratory Visualizations
```python
# Distribution of Performance Ratings
plt.figure(figsize=(6, 4))
sns.countplot(x='PerformanceRating', data=df, palette='Set2')
plt.title("Distribution of Performance Ratings")
plt.tight_layout()
plt.savefig("Visuals/Distribution of Performance Ratings.png")

# Boxplot for PercentSalaryHike by PerformanceRating
plt.figure(figsize=(6, 4))
sns.boxplot(x='PerformanceRating', y='PercentSalaryHike', data=df, palette='Set3')
plt.title("PercentSalaryHike by Performance Rating")
plt.tight_layout()
plt.savefig("Visuals/PriceSalaryHike by Performance Rating.png")
```

---

## 🧠 Modeling
```python
# Define features and target
X = df.drop(columns=["PerformanceRating"])
y = df["PerformanceRating"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Train Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

---

## 📊 Evaluation Visuals
```python
# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("Visuals/Confusion Matrix.png")

# Feature Importance
importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importances.head(10), x="Importance", y="Feature", palette="viridis")
plt.title("Top 10 Feature Importances for Predicting Performance Rating")
plt.tight_layout()
plt.savefig("Visuals/Top 10 Feature Importances for Predicting Performance Rating.png")
```

---

## ✅ Key Insights
- `PercentSalaryHike` is the top predictor of performance ratings.
- Most employees are rated "3" – strong class imbalance.
- Base compensation (e.g., `MonthlyRate`) has weak correlation with performance.

---

## 🧠 Future Work
- Use class balancing techniques like SMOTE or class weights.
- Try additional models: XGBoost, Logistic Regression.
- Add interaction features (e.g., salary change × years at company).

---

## 🖥 How to Run
```bash
# Step 1: Install requirements
pip install pandas matplotlib seaborn scikit-learn

# Step 2: Run the Python script
python employee_performance_model.py
```

---

## 👤 Author
**Amged Alamari**  
Baruch College — Computer Information Systems  
LinkedIn: [linkedin.com/in/amgedalamari](https://www.linkedin.com/in/amgedalamari)

