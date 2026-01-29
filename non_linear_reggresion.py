import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

x=np.arange(-5,5,0.1)
y=3*(x**3)+2*(x**2)+4*(x)+3
y_noise=20* np.random.normal(size=x.size)
y_data=y+y_noise
plt.plot(x,y_data,'b.')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Noisy Quadratic Data')
plt.show()

x2=np.arange(-5,5,0.1)
y2=np.exp(x2)
plt.plot(x2,y2,'r.')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Exponential Data')
plt.show()

X = np.arange(-5.0, 5.0, 0.1)

Y = np.log(X)

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

X2 = np.arange(-5.0, 5.0, 0.1)


Y2 = 1-4/(1+np.power(3, X-2))

plt.plot(X2,Y2) 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

china_GDP=pd.read_csv("china_gdp.csv")
print(china_GDP.head(10))
print(china_GDP.describe())

plt.scatter(china_GDP.Year,china_GDP.Value)
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()

def sigmoid(x,beta1,beta2):
    y=1/(1+np.exp(-beta1*(x-beta2)))
    return y

beta1=0.2
beta2=2000
x_data=china_GDP["Year"]
Y_data=china_GDP["Value"]
y_predict=sigmoid(x_data,beta1,beta2)

plt.plot(x_data,y_predict)
plt.plot(x_data,Y_data,"k")
plt.xlabel("Year")
plt.ylabel("GDP")
plt.show()
# normalize data 
xdata=x_data/np.max(x_data)
ydata=y_data/np.max(y_data)

curve=curve_fit(sigmoid,xdata,ydata)
print(" beta_1 = %f, beta_2 = %f" % (curve[0], curve[1]))

z = np.linspace(1960, 2015, 55)
z = z/max(z)
plt.figure(figsize=(8,5))
c = sigmoid(z, *curve)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(z,c, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

        