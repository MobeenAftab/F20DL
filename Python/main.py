'''
    step 1: Format data
        - Load CSV file
        - Return appropriate numpy array structure
        - Set the n_samples and n_features attributes

    Refrences:
        http://pandas-docs.github.io/pandas-docs-travis/api.html#dataframe
        https://medium.com/dunder-data/selecting-subsets-of-data-in-pandas-6fcd0170be9c
        http://pandas.pydata.org/pandas-docs/stable/10min.html
        http://scikit-learn.org/stable/datasets/index.html#loading-other-datasets
'''

import pandas as pd
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# index_col=0?
data = pd.read_csv('S:/fer2018/csv/fer2018.csv', header=None, index_col = False)

index = data.index
columns = data.columns
values = data.values


# create a list of features
feature_cols = columns
# use the list to select a subset of the original dataframe
x = data[feature_cols]
# print the first 5 rows
# print(x.head(), "\n", x.shape)

# select first column in array
y = data.iloc[:,0]
# print(y.head(), "\n", y.shape)

def printDF():
    '''
        Return information about dataframe
    '''
    print(data.head())      
    print(data.shape)
    print("index:\n", index)
    print("columns:\n", columns)
    print("values:\n", values)

# printDF()

'''
    Splitting x and y into training and testing sets
'''

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

# default split is 0.75 for training and 0.25 for testing

def printTTS():
    print(
        "x_train:", x_train.shape, "\n",
        "y_train:", x_train.shape, "\n"
        "x_test:", x_train.shape, "\n"
        "y_test:", x_train.shape, "\n"
        )

# printTTS()

'''
    Step 2: Build KNN model, k=3
'''

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print("KNeighborsClassifier\t k=3:\n", y_pred)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:\t", accuracy) # 0.34407044137316095
