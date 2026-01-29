import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data=pd.read_csv("Titanic.csv")
print(data.head(10))
print(data.describe())
data.hist()
plt.show()

x=data[["survived","name","sex","age","sibsp","parch","ticket","fare","cabin","embarked","boat","home.dest"]].values
y=data["pclass"].values
print(x)
print(y)

# normalize data 
# scaler=preprocessing.StandardScaler()
# x=scaler.fit(x)

# split 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
print("train set",x_train,y_train)
print("test set",x_test,y_test)

# train 
ks=10
accurecy=np.zeros(ks-1)
for i in range(1,10):
    knn=KNeighborsClassifier(n_neighbors=i).fit(x_train,y_train)
    y_predict=knn.predict(x_test)
    accurecy[i-1]=metrics.accuracy_score(y_test,y_predict)
print(accurecy)

        