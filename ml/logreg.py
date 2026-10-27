import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.arange(1, 11).reshape(-1, 1)
Y = [1,1,1,1,1,0,0,0,0,0]

model = LogisticRegression().fit(X, Y)

age = float(input("Enter car age: "))
pred = model.predict([[age]])[0]
prob = model.predict_proba([[age]])[0]

print("Category:", "High Resale Value" if pred else "Low Resale Value")
print("Low Resale Probability:", round(prob[0], 2))
print("High Resale Probability:", round(prob[1], 2))

x = np.linspace(1, 10, 100).reshape(-1, 1)
plt.scatter(X, Y)
plt.plot(x, model.predict_proba(x)[:, 1])
plt.axhline(0.5, linestyle="--")
plt.scatter(age, prob[1], marker="*", s=150)
plt.xlabel("Car Age")
plt.ylabel("High Resale Probability")
plt.show()
