import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets._samples_generator import make_blobs
from scipy.spatial import distance_matrix
from scipy.cluster import hierarchy

x,y=make_blobs(n_samples=50, centers=[[4,4],[-2,-1],[10,4],[2,2]],cluster_std=0.9)

print(x.shape)
print(y)

plt.scatter(x[:,0],x[:,1],marker="o")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

agglom=AgglomerativeClustering(n_clusters=4,linkage="complete")

agglom.fit(x,y)
# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(6,4))

x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)

X1 = (x - x_min) / (x_max - x_min)

for i in range(X1.shape[0]):
    plt.text(X1[i, 0], X1[i, 1], str(y [i]),
             color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})
    
plt.xticks([])
plt.yticks([])

plt.scatter(X1[:, 0], X1[:, 1], marker='.')
plt.show()

dist=distance_matrix(x,x)
print(dist)

z=hierarchy.linkage(dist,"complete")
dendrogram=hierarchy.dendrogram(z)
plt.show()
