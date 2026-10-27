import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X, y = load_iris(return_X_y=True)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

print("Original Shape:", X.shape)
print("After PCA Shape:", X_pca.shape)
print("Variance Preserved:", round(sum(pca.explained_variance_ratio_), 3))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].scatter(X[:, 0], X[:, 1], c=y)
ax[0].set(title="Original (2 of 4 features)", xlabel="Sepal Length", ylabel="Sepal Width")

ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y)
ax[1].set(title="After PCA (4 features -> 2)", xlabel="Principal Component 1", ylabel="Principal Component 2")

plt.show()
