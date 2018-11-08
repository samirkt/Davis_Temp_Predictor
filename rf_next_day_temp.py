import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

filename = 'data_with_next_temp.csv'
test_filename = 'data_2018.csv'

features = pd.read_csv(filename)
features_2018 = pd.read_csv(test_filename)


print '\n\n***DATA STATISTICS***\n\n'

#Print dataframe contents and properties
print '\nData (5):\n',features.head(5)
print '\nShape of data:\n',features.shape
print '\nStats:\n',features.describe()

#Extract target data
max_t = np.array(features['TMAX'])
max_t_next = np.array(features['TMAX_NXT'])

#Extract training features to array
features = features.drop(['YEAR','TMAX','TMAX_NXT','SN32','SN33','SN35'],axis=1)
feature_list = list(features.columns)
features = np.array(features)

#Convert 2018 data to array
features_2018 = np.array(features_2018)



### Split 2017 data into testing and training ###

#Split data into testing and training
ratio = 0.20
train_features,test_features,train_target,test_target = train_test_split(features,max_t_next,test_size=ratio,random_state=42)
train_max_t,test_max_t = train_test_split(max_t,test_size=ratio,random_state=42)

#Establish baseline
baseline = test_max_t
avg_baseline_err = abs(baseline - test_target)

print '\nBase vs. Target:\n',[baseline,test_target,avg_baseline_err]
print '\nAverage Baseline Error:\n',round(np.mean(avg_baseline_err),2)






raw_input('Press enter to continue...')

print '\n\n***BUILDING MODEL FOR 2017 DATA***\n\n'

rf = RandomForestRegressor(n_estimators=2000,random_state=42)
rf.fit(train_features,train_target)

predictions = rf.predict(test_features)
error = abs(predictions-test_target)
print '\nPrediction Error: ',round(np.mean(error),2)
print 'Prediction Accuracy: ',round(np.mean(100*(1-(error/test_target))),2),'%'

print '\n\n***INTERACTIVE TEMP PREDICTION FOR 2018***\n\n'
while 1:
    #Get user input
    month = input('Enter month (1-12): ')
    day = input('Enter day (1-31): ')
    sx32 = input('Enter soil temp at 10 cm: ')
    sx33 = input('Enter soil temp at 20 cm: ')
    sx35 = input('Enter soil temp at 50 cm: ')
    usr_input = np.array([month,day,sx32,sx33,sx35])
    usr_input = usr_input.reshape(1,-1)
    
    #Make prediction
    prediction = rf.predict(usr_input)
    print '\nPredicted Max Temp Tomorrow: ',round(prediction[0],2),' degrees'
    try:
        #Print actual max temperature
        date_of_interest = features_2018[np.where((features_2018[:,1]==month)*(features_2018[:,2]==day))][0]
        actual = date_of_interest[4]
        print '\nActual Max Temp Tomorrow: ',actual,' degrees\n\n'
    except:
        #Actual value does not exist in database
        print '\nDate for 2018 not found!\n\n'
