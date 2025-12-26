import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score

Co2_df = pd.read_csv("./machine_learning_with_python_jadi-main/FuelConsumption.csv")
print(Co2_df)
print(Co2_df.describe())

Useful_valueCo2_df = Co2_df[["MODELYEAR", "ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]

print(Useful_valueCo2_df.query("ENGINESIZE >= 8"))

# Useful_valueCo2_df.plot(kind="scatter", x="ENGINESIZE", xlabel="ENGINESIZE", y="CO2EMISSIONS", ylabel="CO2EMISSIONS",title="ENGINESIZE vs CO2EMISSIONS")

# Useful_valueCo2_df.plot(kind="scatter", x="FUELCONSUMPTION_COMB", xlabel="FUELCONSUMPTION_COMB", y="CO2EMISSIONS", ylabel="CO2EMISSIONS",title="FUELCONSUMPTION_COMB vs CO2EMISSIONS")

# Useful_valueCo2_df.plot(kind="scatter", x="CYLINDERS", xlabel="CYLINDERS", y="CO2EMISSIONS", ylabel="CO2EMISSIONS",title="CYLINDERS vs CO2EMISSIONS")

Useful_valueCo2_df.hist(bins=50, figsize=(10,5))
plt.tight_layout()

# plt.show()

mask=np.random.rand(len(Useful_valueCo2_df))<0.8
train=Useful_valueCo2_df[mask]
test=Useful_valueCo2_df[~mask]
print(mask)
print(~mask)
print(train)
print(test)

fig=plt.figure()
ax1=fig.add_subplot(111)
ax1.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
ax1.scatter(test.ENGINESIZE, test.CO2EMISSIONS, color="red")
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")

regr=linear_model.LinearRegression()
train_x=np.asanyarray(train[["ENGINESIZE"]])
train_y=np.asanyarray(train[["CO2EMISSIONS"]])
regr.fit(train_x, train_y)
print("Coefficients: ", regr.coef_)
print("Intercept: ", regr.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")


test_x=np.asanyarray(test[["ENGINESIZE"]])
test_y=np.asanyarray(test[["CO2EMISSIONS"]])
test_y_pred=regr.predict(test_x)

print("mean absolute error(MAE):",np.mean(np.absolute(test_y_pred - test_y)))
print("mean squared error(MSE):",np.mean(np.absolute(test_y_pred - test_y)**2))
print("R2_score:",r2_score(test_y, test_y_pred))

plt.show()
# simple_linear_reggresion.py

