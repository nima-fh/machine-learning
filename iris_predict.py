import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score,confusion_matrix


iris=pd.read_csv("G:\work\Data science\machine_learning\CSV\iris.csv")
print(iris)
iris.info()
iris.hist()
plt.show()
features=iris.drop("variety",axis=1)
target=iris["variety"]
print(target.value_counts())
print(target.shape)

x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=40,stratify=target)

preprocessing=ColumnTransformer([
    ("num_pipe",StandardScaler(),features.columns)
])


pipe=Pipeline([
    ("preprocessing",preprocessing),
    ("model",LogisticRegression(random_state=40))
])

model=pipe.fit(x_train,y_train)
pred=model.predict(x_test)

print("F1 Score:",f1_score(y_test,pred,average="weighted"))
print("Confusion Matrix:\n",confusion_matrix(y_test,pred))

print()

