#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 15:29:55 2018

@author: virajdeshwal
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

file = pd.read_csv('Social_Network_Ads.csv')
#we are including the two index from our dataset and finding the corelation between them.

X = file.iloc[:,[2,3]].values
y= file.iloc[:,4].values


from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)


#we need to do the feature scaling in the K Nearest Neighbors to get the accurate prediction.

from sklearn.preprocessing import StandardScaler
#Scaling the X_set to be in the same range.
scaling = StandardScaler()

x_train = scaling.fit_transform(x_train)
x_test = scaling.fit_transform(x_test)
#Call the model library. In this case we will call KNN.
from sklearn.neighbors import KNeighborsClassifier
#initialize the model
#n_neighbors is a tuneable value
model = KNeighborsClassifier(n_neighbors= 5, metric = 'minkowski', p=2)
model.fit(x_train, y_train)
#predict the model
y_pred = model.predict(x_test)

#import the confusion matrix
from sklearn.metrics import confusion_matrix

#show the true positive and false positive through the confusion matrix.
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print('\n\n print the confusion matrix for true and false prediction rate.\n\n')
print(conf_matrix)

plt.imshow(conf_matrix)
plt.title('Graphical representation of Prediction of how many people will buy the SUV')
plt.xlabel('AGE')
plt.ylabel('Estimated Salary')
plt.show()
print('\n\n____________________________________________________________\n\n')



# Visualising the Training set results

print ('\n\n Lets Visualize the results from Training set of the KNeighbors\n\n')
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNeighbors (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
print('\n\n_______________________________________________________________\n\n')
# Visualising the Test set results
print ('\n\n Lets Visualize the results from Testing set of the KNeighbors\n\n')
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNeighbors (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print('\n\n\n Hence the accuracy of the KNN is:',accuracy)
print('\n\n Done :)')
print('\n\n REMEMBER n_neighbors is a tuneable value. Dont forget to tune it,to get a better accuracy :)')