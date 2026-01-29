import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

Housing=pd.read_csv("Housing.csv")

"""spliting data"""

features=Housing.drop("price",axis=1)
target=Housing["price"]

x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=1)
print("train set:",x_train.shape,y_train.shape)
print("test set:",x_test.shape,y_test.shape)

"""visualization"""

plt.scatter(x=x_train["area"],y=y_train ,c=x_train["bedrooms"])
plt.xlabel("area")
plt.ylabel("price")
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.show()

"""category and numeric"""

category=["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea","furnishingstatus"]
category_data=features[category]
numeric_data=features.drop(category,axis=1)
print(category_data)

ordinal=OrdinalEncoder()
x_train[category]=ordinal.fit_transform(x_train[category])
x_test[category]=ordinal.transform(x_test[category])

"""logaritm_scaler"""
Housing.hist()
plt.show()
Housing["price"].hist(bins=30)
plt.show()
Housing["price"].apply(np.log).hist(bins=30)
plt.show()
log_scaler=FunctionTransformer(np.log1p,inverse_func=np.expm1)
x_train["area"]=log_scaler.fit_transform(x_train[["area"]])
x_test["area"]=log_scaler.transform(x_test[["area"]])
y_train=log_scaler.fit_transform(y_train.values.reshape(-1,1))
y_test=log_scaler.transform(y_test.values.reshape(-1,1))

""" scaler """
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
print(x_test)

""" train model """
model=LinearRegression()
model.fit(x_train,y_train)

"""predict """

ylog_predict=model.predict(x_test)
y_predict=log_scaler.inverse_transform(ylog_predict.reshape(-1,1))
real_y_test=log_scaler.inverse_transform(y_test.reshape(-1,1))

"""score"""
print("R2_score is: ",r2_score(real_y_test,y_predict))

