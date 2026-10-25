# Install required libraries (run in terminal / command prompt):
# pip install numpy matplotlib scikit-learn
#py -3.11 -m pip install numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X = np.array([
    [20,22],[21,25],[22,24],[23,27],[24,26],
    [25,29],[26,28],[27,31],[28,30],[29,32],
    [40,65],[41,68],[42,70],[43,67],[44,72],
    [45,75],[46,73],[47,78],[48,76],[49,80],
    [60,35],[61,38],[62,37],[63,40],[64,42],
    [65,41],[66,44],[67,46],[68,45],[69,48],
    [30,90],[75,15],[85,95],[18,80],[55,10]
])

print("Data points:", len(X))
print("Features:", X.shape[1])

X_scaled = StandardScaler().fit_transform(X)
labels = DBSCAN(eps=0.45, min_samples=4).fit_predict(X_scaled)

clusters = len(set(labels)) - (-1 in labels)
print("Number of clusters:", clusters)
print("Cluster labels:", labels)
print("Noise points:", np.sum(labels == -1))

plt.scatter(X[:,0], X[:,1])
plt.title("Original Dataset")
plt.xlabel("Age")
plt.ylabel("Annual Spending (Thousands)")
plt.grid()
plt.show()

for label in set(labels):
    points = X[labels == label]
    plt.scatter(points[:,0], points[:,1],
                marker='x' if label == -1 else 'o',
                label='Noise' if label == -1 else f'Cluster {label+1}')

plt.title("DBSCAN Clustering")
plt.xlabel("Age")
plt.ylabel("Annual Spending (Thousands)")
plt.legend()
plt.grid()
plt.show()