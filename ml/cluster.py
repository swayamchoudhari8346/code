import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN, KMeans

X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)

db_labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)
km_labels = KMeans(n_clusters=2, random_state=42).fit_predict(X)

print("DBSCAN clusters:", len(set(db_labels)) - (-1 in db_labels))
print("Noise points:", list(db_labels).count(-1))

fig, ax = plt.subplots(1, 3, figsize=(14, 4))

ax[0].scatter(X[:, 0], X[:, 1])
ax[0].set_title("Before (no labels)")

ax[1].scatter(X[:, 0], X[:, 1], c=km_labels)
ax[1].set_title("KMeans (cuts the moons wrongly)")

ax[2].scatter(X[:, 0], X[:, 1], c=db_labels)
ax[2].set_title("DBSCAN (follows the shapes)")

plt.show()
