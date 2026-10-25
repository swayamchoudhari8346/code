# Install required libraries (run in terminal / command prompt):
# pip install numpy matplotlib scikit-learn
#py -3.11 -m pip install numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = np.array([
[1,55,58,52,60,54],[1.5,60,62,58,63,59],
[2,64,65,62,66,63],[2.5,68,70,66,69,67],
[3,72,74,70,73,72],[3.5,75,78,73,76,75],
[4,78,81,77,79,79],[4.5,82,84,80,82,83],
[5,85,87,84,85,86],[5.5,88,90,87,88,89],
[6,91,93,90,91,92],[6.5,94,95,93,94,94],
[7,96,97,95,96,96],[1.2,57,60,54,61,56],
[2.2,63,67,61,65,64],[3.2,70,72,68,71,69],
[4.2,79,82,76,80,80],[5.2,86,88,83,87,87],
[6.2,92,94,91,92,93],[7.2,97,98,96,97,97]
])

X, score = data[:,:5], data[:,5]
y = (score >= 70).astype(int)

print("Original Shape:", X.shape)
print("Employees:", len(X), "Features:", X.shape[1])

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Reduced Shape:", X_pca.shape)
print("Explained Variance:", pca.explained_variance_ratio_)
print("Total Variance Preserved:", sum(pca.explained_variance_ratio_))
print("PCA Components:\n", pca.components_)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression().fit(X_train, y_train)
pred = model.predict(X_test)

print(classification_report(
    y_test, pred, target_names=["Low Performance", "High Performance"]
))

plt.scatter(X[y==0,0], score[y==0], marker='x', label="Low Performance")
plt.scatter(X[y==1,0], score[y==1], label="High Performance")
plt.xlabel("Experience (Years)")
plt.ylabel("Final Performance Score")
plt.title("Original Dataset")
plt.legend()
plt.grid()
plt.show()

plt.scatter(X_pca[y==0,0], X_pca[y==0,1], marker='x', label="Low Performance")
plt.scatter(X_pca[y==1,0], X_pca[y==1,1], label="High Performance")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Dimensionality Reduction")
plt.legend()
plt.grid()
plt.show()