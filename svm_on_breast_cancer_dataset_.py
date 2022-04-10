# -*- coding: utf-8 -*-
"""SVM  on Breast cancer dataset .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RmeqVppEB1csMWqALWWKRMvJpF8iO78N
"""

from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
data = load_breast_cancer()

print(data.keys())

df = pd.DataFrame(data.data, columns=data.feature_names)
print(df)

df['target'] = data.target
print()

df.shape

df.head()

x = data.data

y = data.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, train_size = 0.7, random_state=25)

"""creating SVM classifier"""

from sklearn import svm

clf = svm.SVC(kernel='linear')

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

"""What is Recall Metrics? - is the fraction of the total amount of relevant instances that were actually retrieved.

# Logistic Regression
"""

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Greys_r" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

y_pred_proba = logreg.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

