# K nearest algorithm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.read_csv('teleCust1000t.csv')
# print(df.head())

# counts distinct values on a given column -> value_counts()
df['custcat'].value_counts()

# list column names
# print(df.columns)

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
# print(X[0:5])

y = df['custcat'].values
# print(y[0:5])

# normalize the data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
# print(X[0:5])

# split the train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
# print ('Train set:', X_train.shape,  y_train.shape)
# print ('Test set:', X_test.shape,  y_test.shape)

## CLASSIFICATION TECHNIQUE
# K-NEAREST NEIGHBOURS
from sklearn.neighbors import KNeighborsClassifier
k = 10  # nearest point number, usually 4
# Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
# print(neigh)

# Predict
yhat = neigh.predict(X_test)
print(yhat[0:5])

# ACCURACY EVALUTION
from sklearn import metrics
# jaccard index
# print('k:', k)
# print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
# print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# changing the K value vs Accuracy calculation
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
yhats = []

for n in range(1,Ks):

    #Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    yhats.append(yhat)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)


    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print(mean_acc)
print(std_acc)


## plot the graph K vs Accuracy
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
yhat_knn = yhats[mean_acc.argmax()]
from sklearn.metrics import jaccard_score
print('jaccard:', jaccard_score(y_test, yhat_knn, pos_label = 4, average='micro') )