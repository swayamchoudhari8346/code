import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X, y = X[:100, :2], y[:100]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='linear').fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            facecolors='none', edgecolors='red')

w, b = model.coef_[0], model.intercept_[0]
x = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

plt.plot(x, -(w[0]*x+b)/w[1], 'k-')
plt.plot(x, -(w[0]*x+b-1)/w[1], 'k--')
plt.plot(x, -(w[0]*x+b+1)/w[1], 'k--')

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
