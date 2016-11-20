import pandas as pd
import numpy as np

# 1. Loading the Data Using Pandas
# using pandas we can read the csv file directly into a  dataframe 
#df = pd.read_csv('data/train.csv')

# 2. Preview the Data
# take a peek at the first ten lines of the csv file
#print(df.head(10))

# 3. Remove irrelevant data
#df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

#4. Remove Missing Values
#df = df.dropna()

#5. Convert All Data to Numerical Data
#print(df.info())
#print(df['Sex'].unique())

#Create the Gender column which is a numerical representation of Sex
#df['Gender'] = df['Sex'].map({'female': 0, 'male':1}).astype(int)

#print(df['Embarked'].unique())

#Create the Port column which is a numerical representation of Port
#df['Port'] = df['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)

#remove irrelevant columns
#df = df.drop(['Sex', 'Embarked'], axis=1)

#print(df.info())

#6. Arrange DataFrame for Training
#cols = df.columns.tolist()
#print(cols)

#cols = [cols[1]] + [cols[0]] + cols[2:]
#df = df[cols] #remakes the DataFrame with the new order of cols

#7. Train a Model
#print(df.head(10))

#get training data from df values
#train_data = df.values

#import RandomForestClassifier from sklearn.ensemble
#from sklearn.ensemble import RandomForestClassifier

#create your classifier
#rfc = RandomForestClassifier(n_estimators = 100)

#train a model with the training data
#model = rfc.fit(train_data[0:,2:], train_data[0:,0])


#8. Make predictions

#load test data into another DataFrame
#df_test = pd.read_csv('data/test.csv')

#prepare test data
#df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
#df_test = df_test.fillna(0)
#df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male':1})
#df_test['Port'] = df_test['Embarked'].map({'C':1, 'S':2, 'Q':3})
#df_test = df_test.drop(['Sex', 'Embarked'], axis=1)

#test_data = df_test.values

#make predictions!!
#pred = model.predict(test_data[:,1:])

#9. Format predictions for Kaggle submission

#combine the PassengerId column with the predictions
#result = np.c_[test_data[:,0].astype(int), pred.astype(int)]

#add a header row
#df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])

#output result to titanic1.csv
#df_result.to_csv('titanic1.csv', index=False)

