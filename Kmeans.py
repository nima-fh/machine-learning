import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs

np.random.seed(0)

x,y=make_blobs(n_samples=5000,centers=[[4,4], [-2, -1], [2, -3], [1, 1]],cluster_std=0.9)
print(x[0:20])
plt.scatter(x[:,0],x[:,1] ,marker=".")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()      

k_means=KMeans(n_clusters=4,init="k-means++",n_init=12)
k_means.fit(x)
k_means_label=k_means.labels_
print(k_means_label)

k_means_cluster_Center=k_means.cluster_centers_
print(k_means_cluster_Center)

# plotting 
fig=plt.figure(figsize=(6,4))
color=plt.cm.sepctral(np.linspace(0,1,len(k_means_label)))

ax=fig.add_subplot(1,1,1)