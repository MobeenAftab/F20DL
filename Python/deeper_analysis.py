'''

    Step 6 - Improving the classification using the selected attributes from  Step 5

    This script runs kNN on the main dataset using the top 2, 5, and 10 attributes gathered
    from the fer2018EMOTION datasets


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

'''
    Libraries being used
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

'''
    Global variables
        - Saving file path of each emotion dataset
'''

fer2018 = '../data/converted/fer2018.csv'
fer2018angry = '../data/converted/fer2018angry.csv'
fer2018disgust = '../data/converted/fer2018disgust.csv'
fer2018fear = '../data/converted/fer2018fear.csv'
fer2018happy = '../data/converted/fer2018happy.csv'
fer2018neutral = '../data/converted/fer2018neutral.csv'
fer2018sad = '../data/converted/fer2018sad.csv'
fer2018surprise = '../data/converted/fer2018surprise.csv'

# Set dataset and file path here
input_file_path = fer2018

# Create data frame from dataset
print("Reading in and converting", input_file_path)
data = pd.read_csv(input_file_path, header=None, index_col=False)

# Remove duplicate rows
print(data.shape)
data = data.drop_duplicates(keep=False)
print(data.shape)

# Extract specific properties from data frame
index = data.index
columns = data.columns
values = data.values

# Create a list of features
feature_cols = columns[1::]


# top attributes gathered from using selectKBest() with chi2 univariate selection
# on individual emotion datasets
top2_attrs_chi2 = [1361, 1362, 29, 30, 769, 817, 1848, 1895, 52, 243, 550, 598, 0, 48]
top5_attrs_chi2 = [1361, 1362, 1408, 1409, 1652, 23, 24, 25, 29, 30, 622, 670, 769, 817, 865, 622, 670, 769, 817, 865, 52, 100, 147, 195, 243
, 550, 598, 599, 646, 647, 0, 1, 48, 96, 144]
top10_attrs_chi2 = [1313,1361,1362,1408,1409,1422,1608,1652,1655,1656,17,18,23,24,25,26,27,28,29,30,622,670,718,721,766,769,817,818,865,910,1847,1848,1849,1850,1894,1895,1896,1897,1943,1944,5,52,99,100,147,148,195,196,243,244,549,550,551,597,598,599,600,646,647,648,0,1,47,48,49,95,96,143,144,192]


'''
uncomment based on what you need to test!
'''
# x = data[feature_cols]
# x = data[top2_attrs_chi2]
# x = data[top5_attrs_chi2]
x = data[top10_attrs_chi2]

# Select first column in array (AND OVERRIDE THE CLASS ATTRIBUTE DATA TYPE)
# y = data.iloc[:,0].astype('bool')
y = data.iloc[:, 0].astype('category')


def printDataFrameMatrix():
    '''
        print information about dataframe
    '''
    print('****Printing data frame properties****\n')
    print('Data haed:\n', data.head(), "\n")
    print('Data Shape:\n', data.shape, "\n")
    print('x Head:\n', x.head(), "\n")
    print('x Shape:\n', x.shape, "\n")
    print('y Shape:\n', y.head(), "\n")
    print('y Shape:\n', y.shape, "\n")
    print('index:\n', index, "\n")
    print('columns:\n', columns, "\n")
    print('values:\n', values, "\n")
    print('****Printing data frame properties end****\n\n')


'''
    splitting x and y into training and testing sets

    parameters
        test_size:
            default split is 0.75 for training and 0.25 for testing
            set this to 0 when doing selectKBest on whole dataset

        shuffle: randomly shuffle data frame
'''


def printTTS():
    '''
        Print out training test split shape
    '''
    print(
        "x_train:\t", x_train.shape, "\n",
        "y_train:\t", y_train.shape, "\n",
        "x_test:\t", x_test.shape, "\n",
        "y_test:\t", y_test.shape, "\n"
    )


# Change this to 0 if doing selectKBest on whole dataset
# and want to keep the shuffling functionality of data
test_size = 0.25

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, shuffle=True, test_size=test_size)

# printTTS()


'''
    Step 2:
        - Build KNN model, k=3
'''


def train_KNearestNeighbour(attributes, class_atr):
    """
        Build and return a KNN model

        Parameters

        attributes: training set of the attributres of the instance class

        class_atr: training set of the the class itribute to predict

    """
    n_neighbours = 3
    print("Training KNN Model with n = ", n_neighbours)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(attributes, class_atr)

    return knn


def modelResults(knn, x_test_selected):
    """
        Evaluate the KNN for selectKBest and print the results

        Warning will be trwon if training and test set are not the same dimention or size

        Print the following:
            Confusion matrix
            Classification report
            roc_curve and roc_auc_score
            Accuracy

        Parameters

            knn: a trained KNN model

            x_test_selected: array from selectKBest
    """
    print("Testing on", x_test_selected.shape[0], "instances\n")
    score = knn.score(x_test_selected, y_test)
    y_pred = knn.predict_proba(x_test_selected)
    probs = pd.DataFrame(y_pred)
    print("all probabilities:\n", probs, "\n")
    for c in range(7):
        likely = probs[probs[c] > 0.5]
        print ("class" + str(c) + " probability > 0.5:\n", likely)
        print ("indexes of likely class" + str(c) + ":", likely.index.tolist(), "\n")

    #
    # print("confusion_matrix:\n", confusion_matrix(y_test, y_pred), '\n')
    # print("classification_report:\n", classification_report(y_test, y_pred), '\n')
    #
    # print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    #
    # y_pred_proba = knn.predict_proba(x_test_selected)[:, 1]
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    #
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr, tpr, label='Knn')
    # plt.xlabel('fpr')
    # plt.ylabel('tpr')
    # plt.title('Knn(n_neighbors=3) ROC curve')
    # plt.show()
    # print('roc accuracy score:\t', roc_auc_score(y_test, y_pred_proba), '\n')

    print("KNN Accuracy:\t", score, '\n')
    print("y_pred: \t", y_pred)



def selectKBest(k):
    """
        Evaluate association links between class and attribute selection.

        algorith is ran on both the training and test sets or else there will be an error when values are
            passed to trainAndEvaluateKNNSelectKBest().

        call trainAndEvaluateKNNSelectKBest(x_train_selected, x_test_selected) with both the new training and test sets for evaluation

        parameters:
            k: Number of best attributes to select and train

    """
    print("Selecting", k, "Best")
    selector = SelectKBest(chi2, k=k).fit(x_train, y_train)
    x_train_selected = selector.transform(x_train)
    x_test_selected = selector.transform(x_test)

    # printSelectKBest(k, selector, x_train_selected, x_test_selected)

    trainAndEvaluateKNNSelectKBest(x_train_selected, x_test_selected)


def printSelectKBest(k, selector, x_train_selected, x_test_selected):
    """
        print properties of the selectKBest(k)

        parameters:
            k = k best to chose

            selector: model of trained selector

            x_train_selected: array of k best attributes from training set

            x_test_selected: array of k best attributes from test set
    """
    # print selector model results
    idxs_selected = selector.get_support(indices=True)
    print("------", k, " best attribute indices are: ", idxs_selected, "-------")
    print("select scores\n", selector.scores_)
    print("select pvalues\n", selector.pvalues_)

    # print x_train_selected properties
    print('x_train_selected Shape:\n', x_train_selected.shape, '\n')
    print(format(x_train_selected.shape), '\n')

    # print x_test_selected properties
    print('x_test_selected Shape:\n', x_test_selected.shape, '\n')
    print(format(x_train_selected.shape), '\n')


def trainAndEvaluateKNN():
    """
        auxiliary function to call train_KNearestNeighbour() on full dataset
        takes no parameters and evaluates the currently loaded test set
    """
    print('***Begin training KNN***')
    knn = train_KNearestNeighbour(x_train, y_train)
    print('***Begin evaluating KNN***')
    modelResults(knn, x_test)


def trainAndEvaluateKNNSelectKBest(x_train_selected, x_test_selected):
    """
        auxiliary function to call train_KNearestNeighbour(x_train_selected, y_train) on full selectKBest attributes

        x
        parameters:

            x_train_selected: array of selectKBest attributes from training set

            x_test_selected:
                array of selectKBest attributes from test set
                variable is only used to pass test set into modelResults() for evaluation

    """
    print('\t***Begin training KNN SelectKBest***\n')
    knn = train_KNearestNeighbour(x_train_selected, y_train)
    print('\t***Begin evaluating KNN***\n')
    modelResults(knn, x_test_selected)

def main():


    printDataFrameMatrix()
    trainAndEvaluateKNN()

    print('END PROGRAM\n\n')



main()

