import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.ticker as ticker

HousingData = pd.read_csv("G:\\work\\Data science\\CSV\\Housing.csv")
print(HousingData.head())

useful_data=HousingData[["price","area","bedrooms","bathrooms","stories","mainroad","guestroom","basement","parking"]]
print(useful_data.head())

useful_data["mainroad"]=useful_data["mainroad"].map({"yes":1,"no":0})
useful_data["guestroom"]=useful_data["guestroom"].map({"yes":1,"no":0})
useful_data["basement"]=useful_data["basement"].map({"yes":1,"no":0})

plt.scatter(useful_data.area, useful_data.price, color='blue')
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend(["Area"])
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.show()

plt.scatter(useful_data.bedrooms, useful_data.price, color='red')
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.legend(["Bedrooms"])
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.show()

plt.scatter(useful_data.bathrooms, useful_data.price, color='green')
plt.xlabel("Bathrooms")
plt.ylabel("Price")
plt.legend(["Bathrooms"])
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.show()

plt.scatter(useful_data.stories, useful_data.price, color='orange')
plt.xlabel("Stories")
plt.ylabel("Price")
plt.legend(["Stories"])
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.show()

mask=np.random.rand(len(useful_data))<0.8
train=useful_data[mask]
test=useful_data[~mask]
print("Train data size:", len(train))
print("Test data size:", len(test))

regr=linear_model.LinearRegression()
x=np.asanyarray(train[["area","bedrooms","bathrooms","stories","mainroad","guestroom","basement","parking"]])
y=np.asanyarray(train["price"])
regr.fit(x,y)
print("Coefficients:", regr.coef_)
print("Intercept:", regr.intercept_)

plt.scatter(useful_data.area, useful_data.price, color='blue')
plt.scatter(useful_data.bedrooms, useful_data.price, color='red')
plt.scatter(useful_data.bathrooms, useful_data.price, color='green')
plt.scatter(useful_data.stories, useful_data.price, color='orange')
plt.plot(train.area, regr.coef_[0]*train.area + regr.coef_[1]*train.bedrooms + regr.coef_[2]*train.bathrooms + regr.coef_[3]*train.stories + regr.coef_[4]*train.mainroad + regr.coef_[5]*train.guestroom + regr.coef_[6]*train.basement + regr.coef_[7]*train.parking + regr.intercept_, '-r')
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend(["Area","Bedrooms","Bathrooms","Stories"])
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.show()

test_x=np.asanyarray(test[["area","bedrooms","bathrooms","stories","mainroad","guestroom","basement","parking"]])
test_y=np.asanyarray(test["price"])
test_y_pred=regr.predict(test_x)
print("Mean absolute error(MAE):", np.mean(np.absolute(test_y_pred - test_y)))
print("Mean squared error(MSE):", np.mean((test_y_pred - test_y)**2))
print("R2_score:", r2_score(test_y, test_y_pred))

