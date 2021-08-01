import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
dir_path = os.path.dirname(os.path.realpath(__file__))
df_train = pd.read_csv(dir_path+'/input/seattle_rain_train.csv')
test_data = pd.read_csv(dir_path+'/input/seattle_rain_test.csv')
print(list(df_train.columns))
print(list(test_data.columns))
df_train['DATE_INT'] = df_train['DATE'].apply(lambda x : int(x.replace('-', '')))
train_data, validation_data = train_test_split(df_train, test_size = 0.2, random_state = 1)
print("train size =", len(train_data))
print("validation size =", len(validation_data))
print("test size =", len(test_data))
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

##final_layer = StackingClassifier(
##    estimators=[('ridge', RidgeClassifier()),
##                ('log', LogisticRegression(random_state=0)),
##                ('svc', SVC(C=1, gamma=1e-6, kernel='rbf'))],
##                final_estimator=LogisticRegression(random_state=0)
##)
##multi_layer_Classifier = StackingClassifier(
##    estimators=[('rf', RandomForestClassifier(max_depth=5,random_state=0)),
##                ('gbrt', GradientBoostingClassifier(random_state=0))],
##                final_estimator=final_layer
##)
##multi_layer_Classifier.fit(train_data[features], train_data[target])
##print_score(multi_layer_Classifier, "multi layer")


results = pd.DataFrame(columns=['C1','C2','C3','A','Score'])
C1_list = [10**j for j in range(0,4)]
C2_list = [10**j for j in range(0,4)]
C3_list = [10**j for j in range(0,4)]
A_list = [10**-j for j in range(0,11,2)]
try:
    for A in A_list:
        for C1 in C1_list:
            for C2 in C2_list:
                for C3 in C3_list:
                    print("A =",A," C1 =",C1," C2 =",C2," C3 =",C3)
                    multi = StackingClassifier(
                        estimators=[('rf', RandomForestClassifier(max_depth=5,random_state=0)),
                                    ('gbrt', GradientBoostingClassifier(random_state=0))],
                                    final_estimator=StackingClassifier(
                                        estimators=[('ridge', RidgeClassifier(alpha=A)),
                                                    ('log', LogisticRegression(C=C1,random_state=0)),
                                                    ('svc', SVC(C=C2, gamma=1e-6, kernel='rbf'))],
                                                    final_estimator=LogisticRegression(C=C3,random_state=0)
                                    )
                    )
                    multi.fit(train_data[features], train_data[target])
                    train_score = accuracy_score(multi.predict(train_data[features]), train_data[target])
                    validation_score = accuracy_score(multi.predict(validation_data[features]), validation_data[target])
                    print("train accuracy = ",train_score,"\tvalidation accuracy = ",validation_score)
                    results = results.append({'C1':C1, 'C2':C2, 'C3':C3, 'A':A, 'Score':(train_score, validation_score)}, ignore_index=True)
##                    results = results.append({'C1':C1, 'C2':C2, 'C3':C3, 'A':A}, ignore_index=True)
except:
    results.to_csv('results.csv')
else:
    print(results)
    results.to_csv('results.csv')
