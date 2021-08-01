import pandas as pd
import numpy as np
from decimal import Decimal
data = pd.read_csv("results.csv")
#print(data['Score'].apply(lambda x: double(str(x)[x.index(',')+1:x.index(')')])))
data['val_score']=data['Score'].apply(lambda x: Decimal(str(x)[x.index(',')+1:x.index(')')]))
print( data.iloc[0])
data.to_csv("results.csv")
