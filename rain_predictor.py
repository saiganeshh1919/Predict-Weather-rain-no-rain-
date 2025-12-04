# Simple Rain Prediction using Logistic Regression

# Install required packages (Run once):
# pip install numpy pandas scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Small Simple Dataset
# -----------------------------
data = {
    "Humidity": [90, 85, 70, 60, 40, 30, 20, 95, 88, 55],
    "Temperature": [25, 28, 30, 32, 35, 38, 40, 22, 24, 33],
    "Rain": [1, 1, 1, 0, 0, 0, 0, 1, 1, 0]  # 1 = Rain, 0 = No Rain
}

df = pd.DataFrame(data)

# -----------------------------
# 2. Features and Target
# -----------------------------
X = df [["Humidity", "Temperature"]]
y = df ["Rain"]

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# -----------------------------
# 4. Train Model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Test Accuracy
# -----------------------------
pred = model.predict(X_test)

print("Predicted:", pred)
print("Actual:   ", y_test.values)
print("Accuracy:", accuracy_score(y_test, pred))

# -----------------------------
# 6. Predict Custom Input
# -----------------------------
new_data = [[85, 26]]  # Humidity, Temperature
print("Rain Prediction (1=Yes, 0=No):", model.predict(new_data)[0])