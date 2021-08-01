import numpy as np # linear algebra
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
dir_path = os.path.dirname(os.path.realpath(__file__))
df_train = pd.read_csv(dir_path+'/input/seattle_rain_train.csv')
print(list(df_train.columns))
train_data, validation_data = train_test_split(df_train, test_size = 0.2, random_state = 1)
print("train size =", len(train_data))
print("validation size =", len(validation_data))
features = [
    'PRCP',
    'TMAX',
    'TMIN',
    'RAIN',
    'TMIDR',
    'TRANGE',
    'MONTH',
    'SEASON',
    'YEST_RAIN',
    'YEST_PRCP',
    'SUM7_PRCP',
    'SUM14_PRCP',
    'SUM30_PRCP'
]
target = 'TMRW_RAIN'

def print_score(classifier, name):
    train_preds = classifier.predict(train_data[features])
    validation_preds = classifier.predict(validation_data[features])
    train_accuracy = accuracy_score(train_data[target], train_preds)
    validation_accuracy = accuracy_score(validation_data[target], validation_preds)
    print(name, "train accuracy =", train_accuracy)
    print(name, "validation accuracy =", validation_accuracy)
    print()
knn = KNeighborsClassifier(n_neighbors=50,weights='uniform')
##knn = KNeighborsClassifier(n_neighbors=15,weights='distance')
knn.fit(train_data[features], train_data[target])
print_score(knn,'knn')
##hyperparameters = {'n_neighbors': [1,2,3,5,10,15,20],'weights':['uniform','distance'],'leaf_size':[5,10,20,30,50]}
##search = GridSearchCV(KNeighborsClassifier(), hyperparameters, cv=10, return_train_score=True)
##search.fit(train_data[features], train_data[target])
##print(search.best_params_)
##print_score(search, "gridCV")
