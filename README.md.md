
# Credit Scoring Model

This project is part of my Machine Learning internship at CodeAlpha.  
It predicts whether a person is creditworthy based on financial and demographic data.

## 📌 Objective
To build a machine learning model that classifies individuals as creditworthy or not using classification algorithms.

## 💻 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Random Forest Classifier
- Matplotlib & Seaborn
- Joblib

## 📂 Dataset
- Source: A simplified version of the German Credit Data.
- Note: The `CreditRisk` column (target variable) is randomly generated for demo purposes.
  In a real-world scenario, you should use actual labeled data.

## 🚀 How to Run

1. Make sure you have Python installed.
2. Place the dataset file `german_credit_data.csv` in the same directory as the script.
3. Install required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn joblib
    ```

4. Run the script:
    ```bash
    python credit_model.py
    ```

## 📈 Output
- Prints a clean accuracy score
- Displays a confusion matrix as a heatmap
- Shows a full classification report

## 📦 Model File
The trained model is saved as `credit_scoring_model.pkl` and can be loaded for later use.

## 🎥 Demo
[(https://www.linkedin.com/posts/deepak-khubchandani-9b2a44324_machinelearning-python-creditscoring-activity-7332150710442029056-d4Qm?utm_source=share&utm_medium=member_ios&rcm=ACoAAFIGHJYBOmujlYd7cY8L4DXtguG2FrKbFMw)]

## ✅ Internship Details
- Internship by CodeAlpha
- Task: Credit Scoring Model
