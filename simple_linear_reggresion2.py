import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score

HousingData = pd.read_csv("G:\\work\\Data science\\CSV\\Housing.csv")
print(HousingData.head())

HousingData['formatted_price'] = HousingData['price'].apply(lambda x: f"${x:,.0f}")


useful_coloumns=HousingData[['area','bedrooms','bathrooms','stories','mainroad','basement','parking','formatted_price','price']]
# Format the price column with commas
print(useful_coloumns.head())
print(useful_coloumns.describe())

useful_coloumns.plot(x='area', y='price', style='o',title='Area vs Price',xlabel='Area in sq ft',ylabel='Price in $')

rand=np.random.rand(len(useful_coloumns))
train=useful_coloumns[rand<0.8]
test=useful_coloumns[rand>=0.8]
print(f"Training data size: {len(train)}")
print(f"Testing data size: {len(test)}")

fig=plt.figure()
ax1=fig.add_subplot(111)
ax1.scatter(train.area, train.price, color="blue", label='Train Data')
ax1.scatter(test.area, test.price, color="red", label='Test Data')
plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
 
regr=linear_model.LinearRegression()
train_x=np.asanyarray(train[['area']])
train_y=np.asanyarray(train[['price']])
regr.fit(train_x, train_y)
print("Coefficients: ", regr.coef_)
print("Intercept: ", regr.intercept_)
plt.scatter(train.area, train.price, color="blue")
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("Area (sq ft)")
plt.show()
test_x=np.asanyarray(test[['area']])
test_y=np.asanyarray(test[['price']])
test_y_pred=regr.predict(test_x)
r2=r2_score(test_y, test_y_pred)
print("Mean absolute error(MAE):",np.mean(np.absolute(test_y_pred - test_y)))
print("Mean squared error(MSE):",np.mean((test_y_pred - test_y)**2))
print("R2_score:",r2)