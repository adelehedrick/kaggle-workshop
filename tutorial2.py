import pandas as pd
import numpy as np

df = pd.read_csv('data/train.csv')
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

###########################################################
# We need to fill the NaN values rather than drop them
# Improvments go here for training
###########################################################

#this is what we had before
df = df.dropna()

# fill the Age column with the mean age
#age_mean = df['Age'].mean()
#df['Age'] = df['Age'].fillna(age_mean)

#import scipy.stats mode
#from scipy.stats import mode

#Fill Embarked with most frequently occurring value (mode)
#mode_embarked = mode(df['Embarked'])[0][0]
#df['Embarked'] = df['Embarked'].fillna(mode_embarked)

###########################################################
# end improvements for training
###########################################################

df['Gender'] = df['Sex'].map({'female': 0, 'male':1}).astype(int)

###########################################################
# Dummy variable improvement here
###########################################################

#what we had before
df['Port'] = df['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)

#df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)

###########################################################
# end dummy variables
###########################################################

df = df.drop(['Sex', 'Embarked'], axis=1)

cols = df.columns.tolist()
cols = [cols[1]] + [cols[0]] + cols[2:]
df = df[cols] 


train_data = df.values

##########################################################
# Modify with parameter tuning
##########################################################

from sklearn.ensemble import RandomForestClassifier

#before
rfc = RandomForestClassifier(n_estimators = 100)

# import grid search
#from sklearn.model_selection import GridSearchCV

# parameters and their ranges to try
#parameter_grid = {
#    'max_features': [0.5, 1.],
#    'max_depth': [5., None]
#}

# search for best parameters and use that for our Classifier
#rfc = GridSearchCV(RandomForestClassifier(n_estimators = 100), parameter_grid, cv=5, verbose=3)

# this wont need to change
model = rfc.fit(train_data[0:,2:], train_data[0:,0])

###########################################################
# end parameter tuning
###########################################################

df_test = pd.read_csv('data/test.csv')

df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

###########################################################
# We need to fill the NaN values rather than drop them
# Improvments go here for training
###########################################################

#this is what we had before
df_test = df_test.fillna(0)

# fill the Age column with the mean age
#df_test['Age'] = df_test['Age'].fillna(age_mean)

#create a pivot table of fare means
#fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')

#fill in the fares with the value from the pivot table if they are missing
#df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x:
#                            fare_means[x['Pclass']] if pd.isnull(x['Fare'])
#                            else x['Fare'], axis=1)

###########################################################
# end improvements for training
###########################################################

df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male':1})

###########################################################
# Dummy variable improvement here
###########################################################

#what we had before
df_test['Port'] = df_test['Embarked'].map({'C':1, 'S':2, 'Q':3})

#df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')], axis=1)
###########################################################
# end dummy variables
###########################################################

df_test = df_test.drop(['Sex', 'Embarked'], axis=1)

test_data = df_test.values

pred = model.predict(test_data[:,1:])

result = np.c_[test_data[:,0].astype(int), pred.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('titanic2.csv', index=False)

