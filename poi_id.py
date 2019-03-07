#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale, Imputer, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tester
import pandas as pd 
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#Because these features will have to undergo treatment which will differ
#I felt the code would be cleaner if features were clustered by data type

identifier_features = ['poi']

income_features = ['salary',
                   'bonus',
                   'long_term_incentive',
                   'deferred_income',
                   'deferral_payments',
                   'loan_advances',
                   'expenses',
                   'director_fees',
                   'total_payments',
                   'other']

stock_features = ['exercised_stock_options',
                  'restricted_stock',
                  'restricted_stock_deferred',
                  'total_stock_value']

email_features = ['to_messages',
                  'from_messages',
                  'from_poi_to_this_person',
                  'from_this_person_to_poi',
                  'shared_receipt_with_poi']
                

my_features = ['to_poi_ratio',
			   'from_poi_ratio',
			   'shared_poi_ratio',
			   'bonus_to_salary',
			   'bonus_to_total']
                

features_list = identifier_features + income_features + stock_features + email_features 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#For the data cleaning part, my experience is that the pandas toolkit is adequate.
#So, I will create a dataframe from the data

df = pd.DataFrame.from_dict(data_dict, orient='index')

#Transforming NaNs in a treatable form
df = df.replace('NaN', np.nan)
df = df[features_list]

print(df.info())

# enron61702insiderpay.pdf states that NaN for financial data means zero
df[income_features] = df[income_features].fillna(value=0)
df[stock_features] = df[stock_features].fillna(value=0)

# enron61702insiderpay.pdf also states that NaN for email data means missing info
imp = Imputer(missing_values='NaN', strategy='mean', axis = 0)

df_poi = df[df['poi'] == True]
df_nonpoi = df[df['poi'] == False]

df_poi.ix[:, email_features] = imp.fit_transform(df_poi.ix[:, email_features])
df_nonpoi.ix[:, email_features] = imp.fit_transform(df_nonpoi.ix[:, email_features])
df = df_poi.append(df_nonpoi)

#Upon visualizing the dataframe, some format errors were found.
#These errors were corrected here:

# Retrieve the incorrect data for Belfer
belfer_financial = df.ix['BELFER ROBERT', 1:15].tolist()
# Delete the first element to shift left and add on a 0 to end as indicated in financial data
belfer_financial.pop(0)
belfer_financial.append(0)
# Reinsert corrected data
df.ix['BELFER ROBERT', 1:15] = belfer_financial

# Retrieve the incorrect data for Bhatnagar
bhatnagar_financial = df.ix['BHATNAGAR SANJAY', 1:15].tolist()
# Delete the last element to shift right and add on a 0 to beginning
bhatnagar_financial.pop(-1)
bhatnagar_financial = [0] + bhatnagar_financial
# Reinsert corrected data
df.ix['BHATNAGAR SANJAY', 1:15] = bhatnagar_financial

#The outliers I decided to drop were a company and the high execs who weren't
#POIs, because I felt they might distort the data

df.drop(axis=0, labels = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'], inplace=True)
df.drop(axis=0, labels=['FREVERT MARK A', 'LAVORATO JOHN J', 'WHALLEY LAWRENCE G', 'BAXTER JOHN C'], inplace=True)

###Code to find number of POIs:

n_poi = df['poi'].value_counts()
print(n_poi)

### Task 3: Create new feature(s)

df['to_poi_ratio'] = df['from_poi_to_this_person'] / df['to_messages']
df['from_poi_ratio'] = df['from_this_person_to_poi'] / df['from_messages']
df['shared_poi_ratio'] = df['shared_receipt_with_poi'] / df['to_messages']
df['bonus_to_salary'] = df['bonus'] / df['salary']
df['bonus_to_total'] = df['bonus'] / df['total_payments']
df.fillna(value = 0, inplace=True)

# Scale the data frame 
scaled_df = df.copy()
scaled_df.ix[:,1:] = scale(scaled_df.ix[:,1:])

### Store to my_dataset for easy export below.
my_dataset = scaled_df.to_dict(orient='index')

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#Testing different Classifiers. Is commented in the final version for performance.]

## Create and test the Gaussian Naive Bayes Classifier
#clf = GaussianNB()
#tester.dump_classifier_and_data(clf, my_dataset, features_list)
#tester.main()
## Create and test the Decision Tree Classifier
#clf = DecisionTreeClassifier()
#tester.dump_classifier_and_data(clf, my_dataset, features_list)
#tester.main()
## Create and test the Support Vector Classifier
#clf = SVC(kernel='linear')
#tester.dump_classifier_and_data(clf, my_dataset, features_list)
#tester.main()
## Create and test the K Means clustering classifier
#clf = KMeans(n_clusters=2)
#tester.dump_classifier_and_data(clf, my_dataset, features_list)
#tester.main()

data = featureFormat(my_dataset, features_list, sort_keys = True)
y, X = targetFeatureSplit(data)
X = np.array(X)
y = np.array(y)

### Code used to assess the optimal amount of parameters to use in the model
##Tuning the classifier parameters using GridSearchCV

#pipe = Pipeline([
#        ('scaler', StandardScaler()),
#        ('selector', SelectKBest()),
#        ('classifier', DecisionTreeClassifier())
#    ])

#param_grid = {
#    'selector__k': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
#    'classifier__criterion': ['gini','entropy'],
#    'classifier__min_samples_split': [2, 4, 10, 20],
#    'classifier__max_depth': [None, 5, 10, 20],
#    'classifier__max_features': [None, 'sqrt', 'auto']
#}

#tree_grid = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)

#tree_grid.fit(X, y)    
#print "Best parameters: {}".format(tree_grid.best_params_)


features_list = identifier_features + income_features + stock_features + email_features + my_features


## Run pipeline with optimized features
clf = Pipeline([
    ('select_features', SelectKBest(k=17)),
    ('classify', DecisionTreeClassifier(criterion='entropy', max_depth=None, max_features=None, min_samples_split=20))
])




tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
