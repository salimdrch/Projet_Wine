import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error
from pprint import pprint
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn import preprocessing
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from csv import writer
from pydantic import BaseModel


class Wine(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    quality: float
    id: int

# Create & train with wine data Model
def trainModel (dataFile: str) -> RandomForestClassifier:
    data = pd.read_csv(dataFile)
    df = pd.DataFrame(data, columns=[ 'fixed acidity', 'volatile acidity','citric acid' , 'residual sugar' ,'chlorides' , 'free sulfur dioxide','density', 'pH' ,'sulphates','alcohol','quality'])

# Create Classification version of target variable
    df["good wine"] = [1 if i >= 7 else 0 for i in df['quality']]
    X = df.drop(['quality','good wine'], axis = 1)
    y = df["good wine"]

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
    return modelOpt
    
def addWine (dataFile :str, wine : Wine) ->None : 
    data = pd.read_csv(dataFile)
    df = pd.DataFrame(data, columns=[ 'fixed acidity', 'volatile acidity','citric acid' , 'residual sugar' ,'chlorides' , 'free sulfur dioxide','density', 'pH' ,'sulphates','alcohol','quality'])
    listWine : list = []
    listWine.append(str(wine.fixedActivity))
    listWine.append(wine.volatileAcidity)
    listWine.append(wine.citricAcid)
    listWine.append(wine.residualSugar)
    listWine.append(wine.chlorides)
    listWine.append(wine.freeSulfurDioxide)
    listWine.append(wine.toalSulfurDioxide)
    listWine.append(wine.density)
    listWine.append(wine.ph)
    listWine.append(wine.sulphates)
    listWine.append(wine.alcohol)
    listWine.append(wine.quality)
    with open(dataFile, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(listWine)
        f_object.close()

def predictiWine( wine : Wine) -> int :
    modelOpt = joblib.load('rf_classifier.plk')
    y_pred = modelOpt.predict(wine)
    addWine('../../../datasource/Wines.csv', wine)
    return(y_pred)

def serealizer() -> None :
    modelOpt = trainModel('../../../datasource/Wines.csv')
    joblib.dump(modelOpt, 'rf_classifier.pkl')
    return(None)

def perfectWine(dataFile : str)-> Wine :
    data = pd.read_csv(dataFile)
    df = pd.DataFrame(data, columns=[ 'fixed acidity', 'volatile acidity','citric acid' , 'residual sugar' ,'chlorides' , 'free sulfur dioxide','density', 'pH' ,'sulphates','alcohol','quality'])
    perfectWine= Wine(fixed_acidity= (np.average(df[df["good wine"] == 1].fixed_acidity)),
                      volatile_acidity= (np.average(df[df["good wine"] == 1].volatile_acidity)),
                      citric_acid=(np.average(df[df["good wine"] == 1].citric_acid)),
                      residual_sugar=(np.average(df[df["good wine"] == 1].residual_sugar)),
                      chlorides=(np.average(df[df["good wine"] == 1].chlorides)),
                      free_sulfur_dioxide=(np.average(df[df["good wine"] == 1].free_sulfur_dioxide)),
                      total_sulfur_dioxide=(np.average(df[df["good wine"] == 1].total_sulfur_dioxide)),
                      density=(np.average(df[df["good wine"] == 1].density)),
                      pH=(np.average(df[df["good wine"] == 1].pH)),
                      sulphates=(np.average(df[df["good wine"] == 1].sulphates)),
                      alcohol=(np.average(df[df["good wine"] == 1].alcohol)),
                      quality=(np.average(df[df["good wine"] == 1].quality)),
                      id=(np.max(df.id)+1),)

    addWine('../../../datasource/Wines.csv', perfectWine)
    return(perfectWine)
