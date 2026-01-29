import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score,classification_report,confusion_matrix ,jaccard_score
import itertools

cell_data=pd.read_csv("G:\work\Data science\machine_learning\machine_learning_with_python_jadi-main\cell_samples.csv")

print(cell_data.head(10))
print(cell_data.describe())
print(cell_data["BareNuc"].value_counts())

ax = cell_data[cell_data['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_data[cell_data['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign',ax=ax);
plt.show()

print(cell_data.dtypes)

# preprocessing 
cell_data=cell_data[pd.to_numeric(cell_data['BareNuc'],errors='coerce').notnull()]
cell_data['BareNuc']=cell_data['BareNuc'].astype('int')
print(cell_data.dtypes)

x=np.asanyarray(cell_data[["Clump","UnifSize","UnifShape","MargAdh","SingEpiSize","BareNuc","BlandChrom","NormNucl","Mit"]])
y=np.asanyarray(cell_data['Class'])
# spliting 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
print("train set:",x_train.shape,y_train.shape)
print("test set:",x_test.shape,y_test.shape)

# modeling 

clf=svm.SVC(kernel='rbf')
clf.fit(x_train,y_train)

# test 

y_predict=clf.predict(x_test)
print(y_test)
print(y_predict)

print("f1 score is:",f1_score(y_test,y_predict,average='weighted'))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
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
    # Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_predict, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, y_predict))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')
plt.show()

print(jaccard_score([0, 1, 0, 0, 1, 1, 1],[0, 1, 0, 1, 1, 1, 0]))