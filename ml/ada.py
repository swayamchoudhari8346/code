import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

model = AdaBoostClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred) * 100

print("Accuracy:", accuracy_score(y_test, pred) * 100, "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

plt.bar(["Boosting"], [acc])
plt.ylabel("Accuracy (%)")
plt.show()
