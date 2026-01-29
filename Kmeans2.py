import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs

df=pd.read_csv("G:\work\Data science\machine_learning\machine_learning_with_python_jadi-main\Cust_Segmentation.csv")
# preprocessing 
df=df.drop("Address",axis=1)
print(df)
print(df.describe())

x=df.values[:,1:]
x=np.nan_to_num(x)
scaler=StandardScaler()
scaler.fit_transform(x)

# modeling 

cluster_num=3
k_means=KMeans(n_clusters=cluster_num,n_init=12,init="k-means++")
k_means.fit(x)
print(k_means.labels_)

df["clus_num"]=k_means.labels_

print(df.head())

print(df.groupby("clus_num").mean())


