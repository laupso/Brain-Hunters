#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:19:31 2020

@author: Brain Hunter
"""

#%% 1-Import important libraries
import numpy as np
import basics_steinmetz as bs
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap
# import pandas as pd


#%% 2- Data retrieval
alldat = bs.import_dataset()

#%% Calculate normalized firing rate
X,y=bs.collect_spks(alldat, NT=92, balance_correct_incorrect=True, brain_regions = 'all')
#%%3- Split the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X.mean(axis=2).T, 1*y, test_size = 0.2, random_state = 0)

#%% 4- Apply Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%5. Apply LDA
lda = LDA(n_components = 1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

#%%6. Fit Logistic Regression to the Training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#%%7. Predict the Test set results
y_pred = classifier.predict(X_test)

#%%8. Check the accuracy by Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy=accuracy_score(y_test,y_pred)

#%%9. Visualize the Test set results -- needs 16 dimenstion-- fix
X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01))

# plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 0],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Logistic Regression (Test set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()

plt.scatter(X_set,y_set)
plt.show()