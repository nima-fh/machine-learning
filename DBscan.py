import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs


def Create_datapoint(n_samples,centers,cluster_std):
    x,y=make_blobs(n_samples=n_samples,centers=centers,cluster_std=cluster_std)
    
    x=StandardScaler().fit_transform(x)
    return x,y

x,y=Create_datapoint(1500,[[4,2],[-2,-1],[2,1]],0.5)
print(x.shape)
print(y)

# modeling

db=DBSCAN(eps=0.3,min_samples=7).fit(x,y)
label=db.labels_
print(label)

core=np.zeros_like(label,dtype=bool)
core[db.core_sample_indices_]=True
print(core)

n_cluster=len(set(label))-(1 if -1 in label else 0)
print(n_cluster)

unique_labels=set(label)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (label == k)

    # Plot the datapoints that are clustered
    xy = x[class_member_mask & core]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

    # Plot the outliers
    xy = x[class_member_mask & ~core]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)
    
plt.show()
