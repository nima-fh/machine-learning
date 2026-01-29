import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


iris=pd.read_csv("G:\work\Data science\machine_learning\CSV\iris.csv")
print(iris)
iris.hist()
plt.show()

print(iris.info())
print(iris["variety"].value_counts())
plt.scatter(x=iris["sepal_length"],y=iris["sepal_width"])
plt.xlabel('sepal_length')
plt.ylabel("sepal_width")
plt.show()

features=iris.drop("variety",axis=1)
target=iris["variety"]

le=LabelEncoder()
target_encoded=le.fit_transform(target)

"""spiliting"""
x_train,x_test,y_train,y_test=train_test_split(features,target_encoded,test_size=0.2,random_state=40)
print(y_train)

pipe=Pipeline(
   [ ("scaler",StandardScaler()),
    ("model",LogisticRegression(max_iter=1000)) ]
)

model=pipe.fit(x_train,y_train)

y_predict=model.predict(x_test)

print("fscore is :",f1_score(y_test,y_predict,average="weighted"))
print(y_test)
print(y_predict)







