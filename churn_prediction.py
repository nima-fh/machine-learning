import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


churn_data=pd.read_csv("G:\work\Data science\machine_learning\CSV\ChurnData.csv")

print(churn_data)
churn_data.info()
print(churn_data.corr()["churn"])
churn_data.hist()
plt.show()

features=churn_data.drop("churn",axis=1)
target=churn_data["churn"]

x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=40,stratify=target)

numeric_pipe=Pipeline([
    ("scaler",StandardScaler())
])

preprocessing=ColumnTransformer([
    ("numeric",numeric_pipe,features.columns)
])

# """finding best model """
# modles={
#     "knn":KNeighborsClassifier(),
#     "tree":DecisionTreeClassifier(),
#     "randoForest":RandomForestClassifier(n_estimators=200,random_state=40),
#     "logistic":LogisticRegression()
#     }
# cv=KFold(n_splits=5,shuffle=True,random_state=40)

# result={}

# for name,model in modles.items():
#     pipe=Pipeline([
#     ("preprocess",preprocessing),
#     ("model",model)
# ])  
#     scores=cross_val_score(pipe,features,target,scoring="accuracy",cv=cv)

#     result[name]=(scores.mean(),scores.std())
    

# for model,(mean,std) in result.items():
#     print(f"{model}: accuracy = {mean:.4f} ± {std:.4f}")

pipe=Pipeline([
    ("preprocessing",preprocessing),
    ("model",KNeighborsClassifier())
])

model=pipe.fit(x_train,y_train)

y_predict=model.predict(x_test)

print(f"f1 score is : {f1_score(y_test,y_predict)}")
