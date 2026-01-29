import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


data=pd.read_csv("G:\work\Data science\machine_learning\machine_learning_with_python_jadi-main\\teleCust1000t.csv")

print(data.head(10))
print(data.describe())
print(data["custcat"].value_counts())

data.hist("income",bins=50)
plt.show()

x=data[["region","tenure","age","marital","address","income","ed","employ","retire","gender","reside"]].values
y=data["custcat"].values
print(x)
print(y)

# Normalize data

x=preprocessing.StandardScaler().fit(x).transform(x.astype(float))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
print(f"train set:{x_train.shape,y_train.shape}")
print(f"test set:{x_test.shape,y_test.shape}")

# train 
k=6
knn=KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)

# test
y_predict=knn.predict(x_test)
print(y_predict)

# evaluate 

print(f"train model accurecy: {metrics.accuracy_score(y_train,knn.predict(x_train))}")
print(f"test model accurecy: {metrics.accuracy_score(y_test,y_predict)}")

ks=10
accrucy=np.zeros(ks-1)
for i in range(1,10):
    knn=KNeighborsClassifier(n_neighbors=i).fit(x_train,y_train)
    y_predict=knn.predict(x_test)
    accrucy[i-1]=metrics.accuracy_score(y_test,y_predict)
print (accrucy)

plt.plot(range(1,ks),accrucy,"g")
plt.xlabel("K number")
plt.ylabel("accurecy")
plt.tight_layout()
plt.show()