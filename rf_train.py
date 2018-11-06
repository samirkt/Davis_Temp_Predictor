import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

features = pd.read_csv('data.csv')

#Print dataframe contents and properties
print('\nData (5):')
print(features.head(5))
print('\nShape of data:')
print(features.shape)
print('\nStats:')
print(features.describe())

#Extract target data
target = np.array(features['TMAX'])
features = features.drop('TMAX',axis=1)

#Extract training features to array
feature_list = list(features.columns)
features = np.array(features)

#Split data into testing and training
train_features,test_features,train_target,test_target = train_test_split(features,target,test_size=0.25,random_state=42)

#Establish baseline
baseline = test_features[:, feature_list.index('TOBS')]

