# -*- coding: utf-8 -*-
"""HeartDiseasePrediction using MLP.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eAmotTCaYfIUKgu9uD5eJ5RHw-389kJ_
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Load the dataset
data = pd.read_csv("heart.csv")  # Ensure this path is correct

# Split into features and target
X = data.drop('target', axis=1).values
y = data['target'].values

# Step 2: Preprocess the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Define the MLP model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Step 4: Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# Step 5: Get user input for prediction
def get_user_input():
    print("\nEnter the following health metrics to predict heart disease:")
    age = float(input("Age: "))
    sex = float(input("Sex (1 = Male, 0 = Female): "))
    cp = float(input("Chest pain type (0, 1, 2, or 3): "))
    trestbps = float(input("Resting blood pressure: "))
    chol = float(input("Serum cholesterol in mg/dl: "))
    fbs = float(input("Fasting blood sugar > 120 mg/dl (1 = True, 0 = False): "))
    restecg = float(input("Resting electrocardiographic results (0, 1, or 2): "))
    thalach = float(input("Maximum heart rate achieved: "))
    exang = float(input("Exercise induced angina (1 = Yes, 0 = No): "))
    oldpeak = float(input("ST depression induced by exercise relative to rest: "))
    slope = float(input("Slope of the peak exercise ST segment (0, 1, or 2): "))
    ca = float(input("Number of major vessels (0-3) colored by fluoroscopy: "))
    thal = float(input("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect): "))

    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    user_input = scaler.transform(user_input)  # Apply the same scaling as the training data
    return user_input

# Step 6: Predict and interpret the result
user_data = get_user_input()
prediction = model.predict(user_data)

# Output the prediction
if prediction >= 0.5:
    print("\nThe model predicts that this individual is likely to have heart disease.")
else:
    print("\nThe model predicts that this individual is unlikely to have heart disease.")