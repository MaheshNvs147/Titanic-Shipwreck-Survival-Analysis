import warnings 
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix,mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifacts, log_metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression , Ridge , Lasso
from sklearn.ensemble import RandomForestRegressor
#import xgboost as xgb
#import torch 
#import torch.nn as nn

warnings.filterwarnings('ignore', category=RuntimeWarning)
pd.set_option('display.max_columns', None)
plt.style.use(style='ggplot')
#%matplotlib inline

if __name__ == '__main__':
    print('Starting the experiment')
    
    #mlflow.set_tracking_uri("http://127.0.0.0:5000")
    mlflow.set_experiment(experiment_name = 'Titanic')
    df = pd.read_csv("/Users/maheshnukala/Desktop/IIIT/PartB/Week2/download.csv")
    df.head()
    df.isnull().sum()
    df=df.replace('male','1').replace('female','0')
    df['Sex']=df['Sex'].astype(int)
    df=df.replace('S','1').replace('C','3').replace('Q','2')
    df['Embarked'] = df['Embarked'].replace(np.nan, '3')
    df['Embarked']=df['Embarked'].astype(int)
    df.describe()
    df['Age'] = df['Age'].replace(np.nan, 29.6)
    df = df.drop(['Name'], axis=1)
    df = df.drop(['Ticket'], axis=1)
    df = df.drop(['PassengerId'], axis=1)
    df = df.drop(['Cabin'], axis=1)
    df_feature = df.drop(['Survived'], axis=1)
    df_target = df.Survived
    df_feature.shape, df_target.shape
    corre=df.corr()
    corre
    from sklearn.model_selection import train_test_split 

    X_train, X_validation, y_train, y_validation = train_test_split(df_feature, df_target, test_size=0.25, random_state=50)
    X_train.shape, X_validation.shape, y_train.shape, y_validation.shape
    log_param("Train shape",X_train.shape )
    log_param("Test shape",X_validation.shape )
    #model_entropy = DecisionTreeClassifier(criterion = "gini",
                            #    max_depth=10, min_samples_leaf=3)

    #model_entropy.fit(X_train, y_train)
    print("Model trained")

    #train_accuracy = model_entropy.score(X_train, y_train)  # performance on train data
    #test_accuracy = model_entropy.score(X_validation, y_validation)  # performance on test data
        
    #log_metric("Accuracy for this run", test_accuracy)
    #pred_test = model_entropy.predict(X_validation)
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 500)
    #regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
  
    # fit the regressor with x and y data
    classifier.fit(X_train, y_train)  
    train_accuracy = classifier.score(X_train, y_train)  # performance on train data
    test_accuracy = classifier.score(X_validation, y_validation) 
    
    log_metric("Accuracy for this run", test_accuracy)
    pred = classifier.predict(X_validation)
    perf_metrics = {"precision_test": precision_score(y_validation, pred, average = 'micro'), 
                        "recall_test": recall_score(y_validation, pred, average = 'micro'),
                        "f1_score_test": f1_score(y_validation, pred, average = 'micro')}

    log_metrics(perf_metrics)
    mlflow.sklearn.log_model(classifier, "Decision Tree Model")