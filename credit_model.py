
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("german_credit_data.csv")  # Make sure the CSV is in the same folder

# Drop unnecessary index column if present
df.drop(columns=['Unnamed: 0'], inplace=True)

# Simulate CreditRisk column (1 = good, 0 = bad) — only for demo
np.random.seed(42)
df['CreditRisk'] = np.random.choice([0, 1], size=len(df))

# Encode all object (text) columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Fill missing values with 0 (simple strategy)
df.fillna(0, inplace=True)

# Features and target
X = df.drop('CreditRisk', axis=1)
y = df['CreditRisk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# ========== OUTPUT ==========

print("\n" + "="*40)
print("🔍 MODEL EVALUATION RESULTS")
print("="*40)

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy Score: {acc:.2f}")

print("\n🧩 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Bad (0)", "Good (1)"], 
            yticklabels=["Bad (0)", "Good (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Bad Credit (0)", "Good Credit (1)"]))

joblib.dump(model, 'credit_scoring_model.pkl')
print("\n📦 Model saved as 'credit_scoring_model.pkl'")
print("="*40)
