import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression


def get_data_described(fileroute):
    """FUNCTION MADE TO PREDICT DATA FROM A DATASET"""
    data = pd.read_csv(fileroute)
    print(data.columns)
    print(data.describe())
    print(data.dtypes)
    return data


def check_data_null_values(data):
    """CHECKING IF NULL VALUES EXISTS AND EACH COLUMNS' VALUES"""
    columns_with_nulls = [col for col in data.columns if data[col].isnull().any() > 0]
    print("Columns with null values\n" , columns_with_nulls)
    for col in data.columns:
        print(data[col].value_counts())


def encode_categoricald_data(data, col):
    """FUNCTION MADE TO ENCODE CATEGORICAL DATA"""
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    print(data.head(5))
    return data

def show_mutual_information(data):
    """PRITING MUTUAL INFORMATION FOR THIS DATASET IN ORDER TO HAVE
       A HAVE AN APPROACH TO HOW EACH FEATURE IS GOING TO AFFECT THE
       PREDICTION"""
    features = data.copy()
    labels = features.pop("not.fully.paid")

    for colname in features.select_dtypes("object"):
        features[colname], _ = features[colname].factorize()
    
    discrete_features = features.dtypes == int
    mi_scores = mutual_info_classif(features,labels, discrete_features = discrete_features)
    mi_scores = pd.Series(mi_scores, name = "MI scores", index = features.columns)
    mi_scores = mi_scores.sort_values(ascending=True)
    width = np.arange(len(mi_scores))
    ticks = list(mi_scores.index)
    plt.barh(width, mi_scores)
    plt.yticks(width,ticks)
    plt.title('Mutual info regression')
    #sns.relplot(x ='int.rate', y = "not.fully.paid", data = data )
    plt.show()




def data_visaulization(data = None, column = None, condition = None, type =None):
    """FUNCTION MADE TO CHECK DIFFERENT TYPE OF PLOTS DEPENDING ON PARAMETERS"""
    #p = data.hist(figsize = (15,15))
    match type:
        case 1:
            sns.set_style("darkgrid")
            plt.hist(data[column].loc[data[condition]==1], bins = 30, label ='Credit.Policy = 1')
            plt.hist(data[column].loc[data[condition]==0], bins = 30, label = 'Credit.Policy = 0')
            plt.legend()
            plt.show()
        case 2:
            plt.figure(figsize=(12,6))
            sns.countplot(data = data, x = 'purpose', hue='not.fully.paid')
            plt.show()
        case 3:
            plt.figure(figsize=(12,6))
            sns.jointplot(data = data, x ='fico', y= 'int.rate')
            plt.show()

def model_creating_prediction(data):
    """USING TRAIN TEST SPLIT THIS TIME SINCE OUR DATASET ISN'T TOO SHORT.
       MODEL SELECTED IS DECISION TREE CLASSIFIER WITH A GRIDSEARCH SO WE 
       CAN USE THE BEST PARAMETERS FOR OUR SEARCH
       FINALLY PRINTING RESULTS REPORT AFTER PREDICTIONS"""
       
    features = data.drop("not.fully.paid", axis = 'columns')
    labels = data['not.fully.paid']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.3, random_state=101)
    model = DecisionTreeClassifier()
    param_grid = {'max_depth': [2,3,4,5,6,7,8,9,10,11,13,15,20]}
    kFold = StratifiedKFold(n_splits = 5)
    grid_search = GridSearchCV(model, param_grid, scoring='recall_weighted', cv= kFold, return_train_score=True)
    grid_search.fit(X_train, y_train)

    best_parameters = grid_search.best_params_
    best_scores = grid_search.best_score_

    tree_model = DecisionTreeClassifier(max_depth=2)
    tree_model.fit(X_train,y_train)
    y_prediction = tree_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_prediction)
    print(accuracy)

    classific = classification_report(y_test, y_prediction)

    confusion_matrix1 = confusion_matrix(y_test,y_prediction)

    print(classific)
    print(confusion_matrix1)

data = get_data_described('data/loan_data.csv')
show_mutual_information(data)
#check_data_null_values(data)
#data = encode_categoricald_data(data,'purpose')
#model_creating_prediction(data)
#data_visaulization(data = data,type = 3)


