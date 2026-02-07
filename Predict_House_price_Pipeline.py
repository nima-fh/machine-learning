import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker 
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.preprocessing import StandardScaler,FunctionTransformer,OneHotEncoder
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
import joblib



Housing=pd.read_csv("Housing.csv")

"""spliting data"""

features=Housing.drop("price",axis=1)
target=Housing["price"]

x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=40)
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
numeric_col=features.drop(category,axis=1).columns
print(category_data)

numeric_pip=Pipeline([
        ("log",FunctionTransformer(np.log1p)),
        ("scale",StandardScaler())
    ])
category_pip=Pipeline([
    ("encoder",OneHotEncoder(drop="first",handle_unknown="ignore"))
])

preprocessing=ColumnTransformer([
    ("numeric",numeric_pip,numeric_col),
    ("category",category_pip,category)
])

"""finding best model """
modles={
    "linear":LinearRegression(),
    "ridge":Ridge(),
    "randoForest":RandomForestRegressor(n_estimators=200,random_state=40),
    "boost":GradientBoostingRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,random_state=40)
    }
cv=KFold(n_splits=5,shuffle=True,random_state=40)

result={}

for name,model in modles.items():
    pipe=Pipeline([
    ("preprocess",preprocessing),
    ("model",model)
])  
    final_model=TransformedTargetRegressor(
    regressor=pipe,
    func=np.log1p,
    inverse_func=np.expm1
)
    scores=cross_val_score(final_model,features,target,scoring="r2",cv=cv)

    result[name]=(scores.mean(),scores.std())
    

for model,(mean,std) in result.items():
    print(f"{model}: R2 = {mean:.4f} ± {std:.4f}")


pipe=Pipeline([
    ("preprocess",preprocessing),
    ("model",Ridge())
])  

final_model=TransformedTargetRegressor(
    regressor=pipe,
    func=np.log1p,
    inverse_func=np.expm1
)
final_model.fit(x_train,y_train)

y_pred=final_model.predict(x_test)

"""score"""
print("R2_score is: ",r2_score(y_test,y_pred))
print("MSE is: ",mean_squared_error(y_test,y_pred))

joblib.dump(final_model,"housing_model.pkl")
# load=joblib.load("housing_model.pkl")
# print(load)