# Install required libraries (run in terminal / command prompt):
# pip install numpy matplotlib scikit-learn
#py -3.11 -m pip install numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.arange(1, 11).reshape(-1, 1)
Y = [22, 20, 18, 17, 15, 13, 12, 10, 9, 7]

model = LinearRegression().fit(X, Y)
print("Intercept:", round(model.intercept_, 2))
print("Slope:", round(model.coef_[0], 4))
print("Equation: Y =", round(model.intercept_, 2), "+", round(model.coef_[0], 4), "X")

age = float(input("Enter car age: "))
price = model.predict([[age]])[0]
print("Predicted Price: $", round(price, 2), "thousand")

plt.scatter(X, Y, label="Actual Data")
plt.plot(X, model.predict(X), "r-", label="Regression Line")
plt.scatter(age, price, color="green", marker="*", s=200, label="Prediction")
plt.xlabel("Car Age")
plt.ylabel("Selling Price ($1000s)")
plt.title("Simple Linear Regression")
plt.grid()
plt.legend()
plt.show()