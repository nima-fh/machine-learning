import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,jaccard_score,f1_score

churn_data=pd.read_csv("G:\work\Data science\machine_learning\machine_learning_with_python_jadi-main\ChurnData.csv")

print(churn_data.head(10))
print(churn_data.describe())
print(churn_data.shape)
print(churn_data.columns)

useful_data=churn_data[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
useful_data['churn']=useful_data['churn'].astype(int)

x=np.asanyarray(useful_data[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless']])
y=np.asanyarray(useful_data["churn"])

# normalize data 
scaler=preprocessing.StandardScaler()
x=scaler.fit(x).transform(x)
print(x[:10])

# spiliting 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
print("Train set:",x_train.shape,y_train.shape)
print("test set:",x_test.shape,y_test.shape)

# modeling 

LR= LogisticRegression(C=0.01,solver='liblinear').fit(x_train,y_train)
Y_predict=LR.predict(x_test) 
print(Y_predict)
print(y_test)

Y_probability=LR.predict_proba(x_test)
print(Y_probability)

js=jaccard_score(y_test,Y_predict, pos_label=1)
print(js)

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, Y_predict, labels=[1,0]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, Y_predict, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
plt.show()

print(f1_score(y_test,Y_predict))