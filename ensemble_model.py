# -*- coding: utf-8 -*-
"""Ensemble model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zc1F9Yufdtkjic3Mlp-QGLLBfCtD91wI
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

data = load_iris()

print(data.keys())

df = pd.DataFrame(data.data, columns=data.feature_names)
print(df)

df.head()

x = data.data

y = data.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, train_size = 0.7, random_state=25)

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=2000)
clf.fit(x_train,y_train)

feature_imp = pd.Series(clf.feature_importances_,index=data.feature_names).sort_values(ascending=False)
feature_imp

from sklearn import metrics
y_pred=clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(x_train, y_train)
y_pred = gb.predict(x_test)

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

"""Did the Gradient Boosting model perform better? Ans. No

Are there any reservations about GB
and why? Ans. GBM models are prone to overfitting 
"""

