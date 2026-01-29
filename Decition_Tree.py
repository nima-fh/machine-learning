import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

Pation_data=pd.read_csv("G:\work\Data science\machine_learning\machine_learning_with_python_jadi-main\drug200.csv")

print(Pation_data.head(10))
print(Pation_data.describe())
print(Pation_data.count())

# preprocessing 
x=Pation_data[["Age","Sex","BP","Cholesterol","Na_to_K"]].values 
y=Pation_data["Drug"].values      
print(x[0:10])

le_sex=preprocessing.LabelEncoder()
le_sex.fit(["M","F"])
x[:,1]=le_sex.transform(x[:,1])

le_BP=preprocessing.LabelEncoder()
le_BP.fit(["LOW","NORMAL","HIGH"])
x[:,2]=le_BP.transform(x[:,2])

le_Cholesterol=preprocessing.LabelEncoder()
le_Cholesterol.fit(["HIGH","NORMAL"])
x[:,3]=le_Cholesterol.transform(x[:,3])

print(x[0:10])
# spiliting data 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# modeling 
drugtree=DecisionTreeClassifier(criterion="entropy",max_depth=4)
drugtree.fit(x_train,y_train)

predtree=drugtree.predict(x_test)
print(predtree[0:5])
print(y_test[0:5])

print("accurecy of this model is:",accuracy_score(y_test,predtree))


