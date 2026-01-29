import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

Housing=pd.read_csv("Housing.csv")
# df=pd.read_csv("G:\work\Data science\machine_learning\machine_learning_with_python_jadi-main\Cust_Segmentation.csv")
# preprocesing and spliting 
# print(Housing.head())
# print(Housing.info())
# print(Housing.describe())
# print(Housing["mainroad"].value_counts())
# print(Housing["guestroom"].value_counts())
# print(Housing["basement"].value_counts())
# print(Housing["hotwaterheating"].value_counts())
# print(Housing["airconditioning"].value_counts())
# print(Housing["prefarea"].value_counts())
# print(Housing["furnishingstatus"].value_counts())
# Housing["bedrooms"].value_counts().sort_index().plot.bar()

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
x_train[category]=ordinal.fit_transform(category_data)
x_test[category]=ordinal.transform(category_data)

# long method of conver category data to numeric 

# def Clean_data(col):
#     Housing[col]=Housing[col].replace(["yes","no"],[1,0])
#     return Housing[col]

# Clean_data("mainroad")
# Clean_data("guestroom")
# Clean_data("basement")
# Clean_data("hotwaterheating")
# Clean_data("airconditioning")
# Clean_data("prefarea")
# Housing["furnishingstatus"]=Housing["furnishingstatus"].replace(["unfurnished","furnished","semi-furnished"],[0,1,2])

"""logaritm_scaler"""
Housing.hist()
plt.show()
Housing["price"].hist(bins=30)
plt.show()
Housing["price"].apply(np.log).hist(bins=30)
plt.show()
log_scaler=FunctionTransformer(np.log,inverse_func=np.exp)
x_train["area"]=log_scaler.fit_transform(x_train[["area"]])
x_test["area"]=log_scaler.transform(x_test[["price"]])
y_train=log_scaler.fit_transform(y_train)
y_test=log_scaler.transform(y_train)
# Housing["area"]=Housing["area"].apply(np.log)
# Housing["price"]=Housing["price"].apply(np.log)

# prepear data and cleaning data in another dataset
# df=df.drop("Address",axis=1)
# print(df)
# print(df.info())
# print(df["Defaulted"].value_counts())
# df_array=np.asanyarray(df)
# simple_imputer=SimpleImputer(strategy="most_frequent")
# simple_imputer.fit_transform(df_array)
# # df=df.dropna(subset="Defaulted")
# df_without_na=pd.DataFrame(simple_imputer.fit_transform(df_array),columns=df.columns)
# print(df_without_na.info())
# print(df_without_na["Defaulted"].value_counts())

""" scaler """
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
print(x_test)

""" train model """
model=LinearRegression()
model.fit(x_train,y_train)

# predict 

ylog_predict=model.predict(x_test)
y_predict=log_scaler.inverse_transform(ylog_predict)
print(y_predict)
real_y_test=log_scaler.inverse_transform(y_test)