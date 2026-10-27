import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.arange(1, 11).reshape(-1, 1)
Y = [22, 20, 18, 17, 15, 13, 12, 10, 9, 7]

model = LinearRegression().fit(X, Y)

print("Intercept:", round(model.intercept_, 2))
print("Slope:", round(model.coef_[0], 4))

age = float(input("Enter car age: "))
price = model.predict([[age]])[0]
print("Predicted Price: $", round(price, 2), "thousand")

plt.scatter(X, Y)
plt.plot(X, model.predict(X))
plt.xlabel("Car Age")
plt.ylabel("Selling Price")
plt.show()
