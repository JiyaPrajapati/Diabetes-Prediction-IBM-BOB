import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("diabetes.csv")

# Separate features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n===== Diabetes Prediction Project =====")
print(f"Dataset records: {len(data)}")
print(f"Model accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm)

# Take user input
print("\nEnter patient details:")

pregnancies = float(input("Pregnancies: "))
glucose = float(input("Glucose: "))
blood_pressure = float(input("Blood Pressure: "))
skin_thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
diabetes_pedigree = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

# Create input data
input_data = pd.DataFrame([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
]], columns=X.columns)

# Make prediction
prediction = model.predict(input_data)

print("\n===== Prediction Result =====")

if prediction[0] == 1:
    print("Prediction: Diabetes likely")
else:
    print("Prediction: Diabetes not likely")

print("\nNote: This is an educational machine learning project and not a medical diagnosis.")