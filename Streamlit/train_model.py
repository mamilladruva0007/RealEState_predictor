import pandas as pd
import numpy as np
import pickle

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("data/india_housing_prices.csv")

categorical_cols = ["State", "City", "Property_Type", "Furnished_Status"]
numeric_cols = [
    "BHK","Size_in_SqFt","Price_in_Lakhs","Price_per_SqFt",
    "Year_Built","Floor_No","Total_Floors",
    "Age_of_Property","Nearby_Schools","Nearby_Hospitals",
    "Build_Year","Build_Decade"
]

# Convert categorical → numbers (simple encoding)
for col in categorical_cols:
    df[col] = df[col].astype("category").cat.codes

# Target 1 (classification)
df["Good_Investment"] = (df["Price_per_SqFt"] <= df["Price_per_SqFt"].median()).astype(int)

# Feature matrix
X = df[categorical_cols + numeric_cols].values
y_class = df["Good_Investment"].values
y_reg = df["Price_in_Lakhs"].values

# Normalize numeric columns
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std

# -------------------------------
# SIMPLE LINEAR REGRESSION (NUMPY)
# -------------------------------
def train_linear_regression(X, y):
    Xb = np.c_[np.ones((X.shape[0], 1)), X]
    w = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y
    return w

# -------------------------------
# SIMPLE LOGISTIC REGRESSION
# -------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.01, epochs=500):
    Xb = np.c_[np.ones((X.shape[0], 1)), X]
    w = np.zeros(Xb.shape[1])

    for _ in range(epochs):
        z = Xb @ w
        preds = sigmoid(z)
        grad = Xb.T @ (preds - y) / len(y)
        w -= lr * grad

    return w

print("Training models on Python 3.14...")

# Train Models
clf_weights = train_logistic_regression(X_norm, y_class)
reg_weights = train_linear_regression(X_norm, y_reg)

# Save models
pickle.dump({
    "weights": clf_weights,
    "mean": X_mean,
    "std": X_std
}, open("model_classifier.pkl", "wb"))

pickle.dump({
    "weights": reg_weights,
    "mean": X_mean,
    "std": X_std
}, open("model_regression.pkl", "wb"))

print("Training completed successfully for Python 3.14!")
print("Saved: model_classifier.pkl, model_regression.pkl")
