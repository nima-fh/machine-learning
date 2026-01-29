import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix



df=pd.read_csv("G:\work\Data science\machine_learning\CSV\cell_samples.csv")

print(df)
print(df.info())
print(df["BareNuc"].value_counts())
print(df["Class"].value_counts())

df=df.replace("?",np.nan)

for col in df.columns:
    df[col]=pd.to_numeric(df[col])

feauters=df.drop("Class",axis=1)
target=df["Class"]

x_train,x_test,y_train,y_test=train_test_split(feauters,target,test_size=0.2,random_state=40)

preprocess_pipline=Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("scaler",StandardScaler())
])

coltransform=ColumnTransformer([
    ("numeric",preprocess_pipline,feauters.columns)
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
#     ("preprocess",coltransform),
#     ("model",model)
# ])  
#     scores=cross_val_score(pipe,feauters,target,scoring="accuracy",cv=cv)

#     result[name]=(scores.mean(),scores.std())
    

# for model,(mean,std) in result.items():
#     print(f"{model}: accuracy = {mean:.4f} ± {std:.4f}")


pipe=Pipeline([
("preprocess",coltransform),
("model",LogisticRegression(random_state=40))
])  
model=pipe.fit(x_train,y_train)
y_predict=model.predict(x_test)

print(f"f1score is : {f1_score(y_test,y_predict,pos_label=4)}")
print(f"accuracy is : {accuracy_score(y_test,y_predict)}")
print(f"confusion_matrix is : {confusion_matrix(y_test,y_predict)}")




