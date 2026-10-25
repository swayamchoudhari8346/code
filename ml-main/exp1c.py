# Install required libraries (run in terminal / command prompt):
# pip install numpy matplotlib scikit-learn
#py -3.11 -m pip install numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.arange(1, 11).reshape(-1, 1)
Y = [1,1,1,1,1,0,0,0,0,0]

model = LogisticRegression().fit(X, Y)
age = float(input("Enter car age: "))

prediction = model.predict([[age]])[0]
probability = model.predict_proba([[age]])[0]

print("Category:", "High Resale Value" if prediction == 1 else "Low Resale Value")
print("Low Resale Probability:", round(probability[0], 2))
print("High Resale Probability:", round(probability[1], 2))

x_test = np.linspace(0, 12, 300).reshape(-1, 1)
plt.scatter(X, Y, label="Training Data")
plt.plot(x_test, model.predict_proba(x_test)[:, 1], "r-", label="Logistic Curve")
plt.axhline(0.5, color="gray", linestyle="--", label="Threshold")
plt.scatter(age, probability[1], color="green", marker="*", s=200, label="Prediction")
plt.xlabel("Car Age")
plt.ylabel("Probability of High Resale Value")
plt.title("Logistic Regression")
plt.ylim(-0.05, 1.05)
plt.grid()
plt.legend()
plt.show()