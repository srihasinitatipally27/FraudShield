import streamlit as st
import pandas as pd

st.title("Fraudshield")
st.write("AI-Based Fraud Detection System")

# Load dataset
df = pd.read_csv("data/creditcard.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.write("Dataset Shape:", df.shape)

st.subheader("Transaction Summary")

normal = (df["Class"] == 0).sum()
fraud = (df["Class"] == 1).sum()

st.write("Normal Transactions:", normal)
st.write("Fraudulent Transactions:", fraud)

# Prepare data for machine learning
X = df.drop("Class", axis=1)
y = df["Class"]

st.subheader("Machine Learning Data")

st.write("Features:", X.shape)
st.write("Target:", y.shape)

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

st.subheader("Train/Test Split")

st.write("Training data:", X_train.shape)
st.write("Testing data:", X_test.shape)


from imblearn.over_sampling import SMOTE

# Balance the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

st.subheader("Balanced Training Data")

st.write("Before SMOTE:", y_train.value_counts())
st.write("After SMOTE:", y_train_resampled.value_counts())


from xgboost import XGBClassifier

# Train XGBoost model
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train_resampled, y_train_resampled)

st.subheader("Model Training")
st.success("XGBoost model trained successfully!")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.subheader("Model Performance")

st.write("Accuracy:", round(accuracy, 4))
st.write("Precision:", round(precision, 4))
st.write("Recall:", round(recall, 4))
st.write("F1 Score:", round(f1, 4))


import joblib

# Save the trained model
joblib.dump(model, "models/fraud_model.pkl")

st.success("Model saved successfully!")


# Fraud Prediction
st.subheader("Fraud Prediction")

st.write("Enter transaction details to check whether the transaction is fraudulent.")

time_value = st.number_input("Transaction Time", min_value=0.0, value=0.0)
amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)

v_values = []

for i in range(1, 29):
    value = st.number_input(f"V{i}", value=0.0)
    v_values.append(value)

if st.button("Check Transaction"):

    transaction = [[time_value] + v_values + [amount]]

    prediction = model.predict(transaction)[0]

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Normal Transaction")