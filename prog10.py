import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
data=load_breast_cancer()
X=data.data
X=StandardScaler().fit_transform(X)
kmeans=KMeans(n_clusters=2,random_state= 0)
label=kmeans.fit_predict(X)
pca=PCA(n_components=2)
x_pca=pca.fit_transform(X)
plt.scatter(x_pca[:,0],x_pca[:,1],c=label,cmap='bwr')
plt.title("PCA kmeans cluster")
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.show()
