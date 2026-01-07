import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

car_data=pd.read_csv("G:\work\Data science\machine_learning\machine_learning_with_python_jadi-main\FuelConsumption.csv")

useful_columns=car_data[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
print(useful_columns.head())
plt.scatter(useful_columns.ENGINESIZE, useful_columns.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")  
plt.ylabel("CO2 Emissions")
plt.show()

mask=np.random.rand(len(car_data))<0.8
train=useful_columns[mask]
test=useful_columns[~mask]

train_x=np.asanyarray(train[["ENGINESIZE"]])
train_y=np.asanyarray(train[["CO2EMISSIONS"]])

test_x=np.asanyarray(test[["ENGINESIZE"]])
test_y=np.asanyarray(test[["CO2EMISSIONS"]])

poly=PolynomialFeatures(degree=2)
train_x_poly=poly.fit_transform(train_x)
print(train_x_poly)

clf=linear_model.LinearRegression()
train_y_=clf.fit(train_x_poly, train_y)
print("Coefficients: ", clf.coef_)
print("Intercept: ", clf.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX=np.arange(0.0, 10.0, 0.1)
YY=clf.intercept_[0] + clf.coef_[0][1]*XX + clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, YY, '-r')
plt.xlabel("Engine Size")  
plt.ylabel("CO2 Emissions")
plt.show()


test_x_poly=poly.fit_transform(test_x)
test_y_=clf.predict(test_x_poly)
print("Mean Squared Error (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y, test_y_))