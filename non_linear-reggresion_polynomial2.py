# Import required libraries
import numpy as np                      # Numerical computations
import matplotlib.pyplot as plt         # Plotting graphs
import pandas as pd                     # Data handling
from sklearn import linear_model        # Machine learning models
from sklearn.metrics import r2_score    # Model evaluation metric
from sklearn.preprocessing import PolynomialFeatures  # Polynomial feature creation

# Load the dataset from CSV file
iceCream_data = pd.read_csv("Ice_cream selling data.csv")

# Display first 10 rows of the dataset
print(iceCream_data.head(10))

# Show statistical summary of the dataset
print(iceCream_data.describe())

# Scatter plot: Temperature vs Ice Cream Sales
plt.scatter(iceCream_data.Temperature,
            iceCream_data.Ice_Cream_Sales,
            color='blue')
plt.xlabel("Temperature (°C)")  
plt.ylabel("Ice Cream Sold")
plt.legend(["Ice Cream Sold vs Temperature"])
plt.show()

# Split data into training (80%) and testing (20%) randomly
mask = np.random.rand(len(iceCream_data)) < 0.8
train = iceCream_data[mask]
test = iceCream_data[~mask]

# Extract input (X) and output (y) for training
train_x = np.asanyarray(train[["Temperature"]])
train_y = np.asanyarray(train[["Ice_Cream_Sales"]])

# Extract input (X) and output (y) for testing
test_x = np.asanyarray(test[["Temperature"]])
test_y = np.asanyarray(test[["Ice_Cream_Sales"]])

# Create polynomial features of degree 2 (x and x²)
poly = PolynomialFeatures(degree=2)

# Transform training input into polynomial features
train_x_poly = poly.fit_transform(train_x)
print(train_x_poly)

# Create a linear regression model
clf = linear_model.LinearRegression()

# Train the model using polynomial features
train_y_ = clf.fit(train_x_poly, train_y)

# Print model coefficients and intercept
print("Coefficients: ", clf.coef_)
print("Intercept: ", clf.intercept_)

# Plot training data points
plt.scatter(train.Temperature, train.Ice_Cream_Sales, color='blue')

# Generate temperature values for curve plotting
XX = np.arange(-5.0, 35.0, 0.1)

# Polynomial regression equation: y = a + b*x + c*x²
yy = (clf.intercept_[0]
      + clf.coef_[0][1] * XX
      + clf.coef_[0][2] * np.power(XX, 2))

# Plot regression curve
plt.plot(XX, yy, '-r')
plt.xlabel("Temperature (°C)")  
plt.ylabel("Ice Cream Sold")
plt.show()

# Transform test data into polynomial features
test_x_poly = poly.fit_transform(test_x)

# Predict ice cream sales for test data
test_y_ = clf.predict(test_x_poly)

# Calculate Mean Squared Error (MSE)
print("Mean Squared Error (MSE): %.2f" %
      np.mean((test_y_ - test_y) ** 2))

# Calculate R² score (model accuracy)
print("R2-score: %.2f" % r2_score(test_y, test_y_))
