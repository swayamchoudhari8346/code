# Install required libraries (run in terminal / command prompt):
# pip install numpy matplotlib scikit-learn
#py -3.11 -m pip install numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[1,15000,1.2],[2,20000,1.4],[3,30000,1.5],
              [4,35000,1.6],[5,45000,1.8],[6,50000,2.0],
              [7,60000,2.0],[8,70000,2.2],[9,80000,2.4],
              [10,90000,2.5]])
Y = [8.5,8,7.2,6.8,6,5.5,4.8,4.2,3.8,3.2]

model = LinearRegression().fit(X, Y)
print("Intercept:", round(model.intercept_, 4))
print("Coefficients:", np.round(model.coef_, 4))

age = float(input("Enter car age: "))
mileage = float(input("Enter mileage: "))
engine = float(input("Enter engine size: "))

price = model.predict([[age, mileage, engine]])[0]
print("Predicted Price: ₹", round(price, 2), "Lakhs")

plt.scatter(X[:, 0], Y, label="Actual Data")
plt.plot(X[:, 0], model.predict(X), "r-", label="Predicted Values")
plt.scatter(age, price, color="green", marker="*", s=200, label="Prediction")
plt.xlabel("Car Age")
plt.ylabel("Selling Price (₹ Lakhs)")
plt.title("Multiple Linear Regression")
plt.grid()
plt.legend()
plt.show()
