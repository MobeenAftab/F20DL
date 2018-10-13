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
from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split

input_file_path = '../data/converted/fer2018angry.csv'

print("Reading in and converting", input_file_path)
data = pd.read_csv(input_file_path, header=None, index_col=False)

index = data.index
columns = data.columns
values = data.values


# create a list of features
feature_cols = columns[1::]
# use the list to select a subset of the original dataframe
x = data[feature_cols]
# print the first 5 rows
# print(x.head(), "\n", x.shape)

# select first column in array (AND OVERRIDE THE CLASS ATTRIBUTE DATA TYPE)
y = data.iloc[:,0].astype('bool')
# y = data.iloc[:,0].astype('category')


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
    print("x.head\n", x.head())
    print("y.head\n", y.head())


printDF()

'''
    Splitting x and y into training and testing sets
'''

# Change this to 0 if doing selectKBest on whole dataset
# and want to keep the shuffling functionality of data
test_size = 0.25

# default split is 0.75 for training and 0.25 for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, shuffle=True, test_size=test_size)


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


# knn = train_KNearestNeighbour(x_train, y_train)
#
# print("Testing on", x_test.shape[0], "instances")
# score = knn.score(x_test, y_test)
# print("Score:\t", score)



'''
    Selecting K best attributes based on a Univariate feature selection
    top 2, 5, and 10.
    
'''


def selectKBest(k):

    print("Selecting", k, "Best")
    selector = SelectKBest(chi2, k=k).fit(x_train, y_train)
    # selected_x = selector.transform(x_train)
    # print("select scores\n", selector.scores_)
    # print("select pvalues\n", selector.pvalues_)

    idxs_selected = selector.get_support(indices=True)
    print("------", k, " best attribute indices are: ", idxs_selected, "-------")

# https://stackoverflow.com/questions/39839112/the-easiest-way-for-getting-feature-names-after-running-selectkbest-in-scikit-le

selectKBest(2)
selectKBest(5)
selectKBest(10)
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


print("END PROGRAM")
