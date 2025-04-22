import numpy as np                    
import pandas as pd                  
import matplotlib.pyplot as plt       
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# --------  create synthetic dataset ----------------------------------------
# We generate 200 random “days” with realistic ranges.
rng = np.random.default_rng(seed=42)                  
N_SAMPLES = 200

# Calories sampled uniformly from 1200–3000
calories = rng.uniform(1200, 3000, N_SAMPLES)

# Protein sampled uniformly from 50–200 g
protein = rng.uniform(50, 200, N_SAMPLES)

# Build feature matrix X: each row = [calories, protein]
X = np.column_stack((calories, protein))

# Apply the success rule to label the data
# 1   diet target met        (≤1800 cals AND ≥100g protein)
# 0   diet target not met
y = ((calories <= 1800) & (protein >= 100)).astype(int)

# Show the first 10 generated rows so you can inspect the raw data
print("\nFIRST 10 RAW SAMPLES")
print(pd.DataFrame(X, columns=["Calories", "Protein"]).head(10).round(1))
print("First 10 labels:", y[:10], "\n")

# split into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

# Feature scaling
# standardise each column to mean 0, std 1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train the perceptron 
clf = Perceptron(max_iter=1000, eta0=0.01, random_state=0)
clf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = clf.predict(X_test_scaled)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print("METRICS ON TEST SET")
print(f"  Accuracy : {acc:.3f}")
print(f"  Precision: {prec:.3f}")
print(f"  Recall   : {rec:.3f}")
print(f"  F1 score : {f1:.3f}")

print("\nCONFUSION MATRIX [TN FP; FN TP]")
print(confusion_matrix(y_test, y_pred))

#Visualise decision boundary
w_cal_scaled, w_prot_scaled = clf.coef_[0]
b_scaled = clf.intercept_[0]

# Means and stds used by the scaler
mean_cal, mean_prot = scaler.mean_
std_cal,  std_prot  = np.sqrt(scaler.var_)

def decision_line(cal):
    return (-(w_cal_scaled/std_cal) * (cal - mean_cal) * std_prot
            - b_scaled * std_prot) / w_prot_scaled + mean_prot

# Create scatter plot of raw (calories, protein) with labels
plt.figure(figsize=(6, 5))
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
            color="red",  label="Cheat (0)", alpha=0.7)
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
            color="green", label="Clean (1)", alpha=0.7)

# Draw the perceptron’s decision boundary
cal_range = np.linspace(1200, 3000, 200)
plt.plot(cal_range, decision_line(cal_range), color="blue", linewidth=2,
         label="Perceptron boundary")

plt.axvline(1800, color="gray", linestyle="--", linewidth=1, label="1800 kcal rule")
plt.axhline(100,  color="gray", linestyle="--", linewidth=1, label="100g protein rule")

plt.xlabel("Calories")
plt.ylabel("Protein (g)")
plt.title("Calories vs. Protein with Perceptron Decision Line")
plt.legend()
plt.tight_layout()
plt.show()