import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset: Age, Mileage
X = np.array([
    [1,15000], [2,20000], [3,30000], [4,35000], [5,45000],
    [6,50000], [7,60000], [8,70000], [9,80000], [10,90000]
])
Y = [8.5,8,7.2,6.8,6,5.5,4.8,4.2,3.8,3.2]

model = LinearRegression().fit(X, Y)

print("Intercept:", round(model.intercept_, 2))
print("Coefficients:", np.round(model.coef_, 4))

age = float(input("Enter car age: "))
mileage = float(input("Enter mileage: "))

price = model.predict([[age, mileage]])[0]
print("Predicted Price: ₹", round(price, 2), "Lakhs")

plt.scatter(X[:,0], Y)
plt.plot(X[:,0], model.predict(X))
plt.xlabel("Car Age")
plt.ylabel("Selling Price")
plt.show()
