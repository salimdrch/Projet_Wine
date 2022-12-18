import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error
from pprint import pprint
import plotly.graph_objs as go
from plotly.offline import iplot
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib 
from sklearn import preprocessing
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('../../../datasource/Wines.csv')
df = pd.DataFrame(data, columns=[ 'fixed acidity', 'volatile acidity','citric acid' , 'residual sugar' ,'chlorides' , 'free sulfur dioxide','density', 'pH' ,'sulphates','alcohol','quality'])

# Create Classification version of target variable
df["good wine"] = [1 if i >= 7 else 0 for i in df['quality']]
X = df.drop(['quality','good wine'], axis = 1)
X = data.drop(["quality"], axis = 1)
y = df["good wine"]
# See proportion of good vs bad wines
y.value_counts()

# Normalize feature variables
X_features = X
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)



grid_params = {
    'n_estimators': [100, 300, 500, 700, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]  
}

forest = RandomForestClassifier(random_state = 1)
modelF = forest.fit(X_train, y_train)
y_predF = modelF.predict(X_test)
grid_search = GridSearchCV(
    estimator=forest,
    param_grid=grid_params,
    scoring='accuracy',
    cv=4, # number of folds
    n_jobs=-1 # all available computing power
)


grid_search.fit(X_train, y_train)
forestOpt = RandomForestClassifier(random_state = 1, max_depth = 15, n_estimators = 500, min_samples_split = 2, min_samples_leaf = 1)

                                   
modelOpt = forestOpt.fit(X_train, y_train)
y_pred = modelOpt.predict(X_test)