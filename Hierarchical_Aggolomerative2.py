import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets._samples_generator import make_blobs
from scipy.spatial import distance_matrix
from scipy.cluster import hierarchy
from sklearn.preprocessing import MinMaxScaler

data=pd.read_csv("G:\work\Data science\machine_learning\machine_learning_with_python_jadi-main\cars_clus.csv")
print(data)
print(data.describe())
print(data.shape)

# data["100km/L"]=236.25/data["mpg"]
# print(data)
data[["sales","resale","type","price","engine_s","horsepow","wheelbas","width","length","curb_wgt","fuel_cap","mpg","lnsales"]]=data[["sales","resale","type","price","engine_s","horsepow","wheelbas","width","length","curb_wgt","fuel_cap","mpg","lnsales"]].apply(pd.to_numeric,errors='coerce')

data=data.dropna()
data=data.reset_index(drop=True)
print(data.shape)
data["100KM/L"]=236.25/data["mpg"]
useful_data=data[["engine_s","horsepow","wheelbas","width","length","curb_wgt","fuel_cap","mpg","100KM/L"]]
useful_data=np.asanyarray(useful_data)
print(useful_data)

# normalise 

scaler=MinMaxScaler()
normal_data=scaler.fit_transform(useful_data)