import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score

car_data=pd.read_csv("G:\work\Data science\machine_learning\machine_learning_with_python_jadi-main\FuelConsumption.csv")

useful_columns=car_data[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY","CO2EMISSIONS"]]
print(useful_columns.head())

plt.scatter(useful_columns.ENGINESIZE, useful_columns.CO2EMISSIONS, color='blue')
plt.scatter(useful_columns.CYLINDERS, useful_columns.CO2EMISSIONS, color='red')
plt.scatter(useful_columns.FUELCONSUMPTION_CITY, useful_columns.CO2EMISSIONS, color='green')
plt.scatter(useful_columns.FUELCONSUMPTION_HWY, useful_columns.CO2EMISSIONS, color='orange')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.legend(["Engine Size","Cylinders","Fuel Consumption City","Fuel Consumption Hwy"])
plt.show()

mask = np.random.rand(len(useful_columns)) < 0.8
train = useful_columns[mask]
test = useful_columns[~mask]
print("Train data size:", len(train))
print("Test data size:", len(test))

regr=linear_model.LinearRegression()
x=train[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY"]]
y=train["CO2EMISSIONS"]
regr.fit(x,y)
print("Coefficients:", regr.coef_)
print("intercept:", regr.intercept_)

plt.scatter(useful_columns.ENGINESIZE, useful_columns.CO2EMISSIONS, color='blue')
plt.scatter(useful_columns.CYLINDERS, useful_columns.CO2EMISSIONS, color='red')
plt.scatter(useful_columns.FUELCONSUMPTION_CITY, useful_columns.CO2EMISSIONS, color='green')
plt.scatter(useful_columns.FUELCONSUMPTION_HWY, useful_columns.CO2EMISSIONS, color='orange')
plt.plot(train.ENGINESIZE, regr.coef_[0]*train.ENGINESIZE + regr.coef_[1]*train.CYLINDERS + regr.coef_[2]*train.FUELCONSUMPTION_CITY + regr.coef_[3]*train.FUELCONSUMPTION_HWY + regr.intercept_, '-r')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.legend(["Engine Size","Cylinders","Fuel Consumption City","Fuel Consumption Hwy"])
plt.show()

test_x = test[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY"]]
test_y = test["CO2EMISSIONS"]
test_y_pred = regr.predict(test_x)
print("Mean absolute error (MAE):", np.mean(np.absolute(test_y_pred - test_y)))
print("Mean squared error (MSE):", np.mean((test_y_pred - test_y) ** 2))
print("R2 score:", r2_score(test_y, test_y_pred))




