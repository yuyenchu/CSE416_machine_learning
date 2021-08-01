import numpy as np
import pandas as pd
from decimal import Decimal
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
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
                    results = results.append({'C1':C1, 'C2':C2, 'C3':C3, 'A':A, 'Score':(1.1, 1.2)}, ignore_index=True)
except:
    results.to_csv('results2.csv')
else:
    results['val_score']=results['Score'].apply(lambda x: x[1])
    print(results)
    results.to_csv('results2.csv')
