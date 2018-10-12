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

        feature selection using SelectKBest and scoring functions
        http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
'''

import pandas as pd
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import train_test_split


data = pd.read_csv('../data/converted/fer2018.csv', header=None, index_col = False)

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


printDF()

'''
    Splitting x and y into training and testing sets
'''

# default split is 0.75 for training and 0.25 for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, shuffle=True)


def printTTS():
    print(
        "x_train:", x_train.shape, "\n",
        "y_train:", y_train.shape, "\n"
        "x_test:", x_test.shape, "\n"
        "y_test:", y_test.shape, "\n"
        )


printTTS()



'''
    Step 2:
        - Build KNN model, k=3
'''


def train_KNearestNeighbour(attributes, class_atr):

    n_neighbours = 3
    print("Training KNN Model with n = ", n_neighbours)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(attributes, class_atr)

    return knn


knn = train_KNearestNeighbour(x_train, y_train)

print("Testing on", x_test.shape[0], "instances")
score = knn.score(x_test, y_test)
print("Score:\t", score)

print("END PROGRAM")

# select = SelectPercentile(percentile=20)
# select = SelectKBest(k = 10)
# select.fit(x_train, y_train)
# x_train_selected = select.transform(x_train)

# print(x_train_selected.shape)
# print(format(x_train_selected.shape))
# print(select.get_support())

# testselect = SelectKBest(k = 10)
# x_test_selected = select.transform(x_test)



# These 3 lines do the same as knn.score(x_test, y_test)
# y_pred = knn.predict(x_test)
# accuracy = metrics.accuracy_score(y_test, y_pred)
# print("KNeighborsClassifier\t k=3:\n", y_pred)



