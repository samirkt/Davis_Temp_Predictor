import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

features = pd.read_csv('data.csv')

#Print dataframe contents and properties
print '\nData (5):\n',features.head(5)
print '\nShape of data:\n',features.shape
print '\nStats:\n',features.describe()

#Extract target data
target = np.array(features['TMAX'])
features = features.drop('TMAX',axis=1)

#Extract training features to array
feature_list = list(features.columns)
features = np.array(features)

#Print means of each data column (identify nans)
print '\n',np.mean(target),
for i in range(0,12):
    print np.mean(features[:,i]),
print

#Split data into testing and training
train_features,test_features,train_target,test_target = train_test_split(features,target,test_size=0.25,random_state=42)

#Establish baseline
baseline = 2*test_features[:, feature_list.index('TOBS')]-test_features[:,feature_list.index('TMIN')]
avg_baseline_err = abs(baseline - test_target)

print '\nBase vs. Target:\n',[baseline,test_target,avg_baseline_err]
print '\nAverage Baseline Error:\n',round(np.mean(avg_baseline_err),2)

rf = RandomForestRegressor(n_estimators=1000,random_state=42)
rf.fit(train_features,train_target)

predictions = rf.predict(test_features)
error = abs(predictions-test_target)
print '\nPrediction Error: ',round(np.mean(error),2)
print 'Prediction Accuracy: ',round(np.mean(100*(1-(error/test_target))),2),'%'
