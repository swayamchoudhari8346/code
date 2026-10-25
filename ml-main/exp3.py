# Install required libraries (run in terminal / command prompt):
# pip install matplotlib scikit-learn
#py -3.11 -m pip install numpy matplotlib scikit-learn

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3,
    random_state=42, stratify=data.target
)

models = {
    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=50, random_state=42
    ),
    "Boosting": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=50, random_state=42
    )
}

accuracies = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred) * 100
    accuracies.append(acc)

    print("\n", name)
    print("Accuracy:", round(acc, 2), "%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(
        y_test, pred, target_names=data.target_names
    ))

print("\nAccuracy Difference:", round(abs(accuracies[0]-accuracies[1]), 2), "%")
print("Better Technique:", ["Bagging", "Boosting"][accuracies[1] > accuracies[0]]
      if accuracies[0] != accuracies[1] else "Both Equal")

plt.bar(models.keys(), accuracies, width=0.5)
plt.ylabel("Accuracy (%)")
plt.title("Bagging vs Boosting Accuracy")
plt.ylim(0, 105)

for i, v in enumerate(accuracies):
    plt.text(i, v+1, f"{v:.2f}%", ha="center")

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.show()