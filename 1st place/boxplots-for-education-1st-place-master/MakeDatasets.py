
##############################################################################
# MakeDatasets.py - script to create datasets in a format conducive to
# multi-multi classification and submission generation.
#
# For: DrivenData.org/ERS's BoxPlots for Education Competition
# License: MIT (http://machinelearner.net/boxplots-for-education-1st-place/LICENSE.txt)
#
# Files produced:
# (A) TrainLabels.csv - extracts training labels from TrainingData.csv
# (B) TrainPredictors.csv - same was TrainingData.csv but just predictor columns (no labels)
# (C) TestData2.csv - reorders columns in TestData.csv to match order in TrainPredictors.csv
##############################################################################

import pandas as pd
import numpy as np

# Get the original datasets
train = pd.read_csv('origdata/TrainingData.csv')
test = pd.read_csv('origdata/TestData.csv')
sample = pd.read_csv('origdata/SubmissionFormat.csv')


#########################
# Create TrainLabels.csv
#########################

# Use column names from SubmissionFormat files.  We will add these columns to TrainLabels.csv
cols = sample.columns.tolist()

# Start the new data frame.  Note Unnamed: 0 is just the name pandas gives to no-name id column
trainLabels = pd.DataFrame(train['Unnamed: 0'])

# Add each column to TrainLabels.csv
for column in cols:
    if column == 'Unnamed: 0': continue
    parts = column.split('__')
    trainLabels[column] = np.where(train[parts[0]] == parts[1],1,0)

# Write the TrainLabels.csv to disk
trainLabels.to_csv('trainLabels.csv',index=False)


#########################
# Create TrainPredictors.csv
#########################

# Remove the label fields so we are just left with predictors
del train['Function']
del train['Use']
del train['Sharing']
del train['Reporting']
del train['Student_Type']
del train['Position_Type']
del train['Object_Type']
del train['Pre_K']
del train['Operating_Status']

# Avoid commas quotes as they complicate parsing later.
# Commas and escaping quotes would be ok if we use pandas or csv package to
# load training data, but for performnce we use pypy and manually parse the
# training file using split(), and it was too much work to deal with the
# quotes and commas at the time.
for column in train.columns.tolist():
	if train[column].dtype == 'object':
		train[column] = train[column].str.replace(',',' ')
		train[column] = train[column].str.replace('"','')

train.to_csv('trainPredictors.csv',index=False)


#########################
# Create TestData2.csv
#########################

train_col_order = ['Unnamed: 0','Object_Description',
 'Text_2',
 'SubFund_Description',
 'Job_Title_Description',
 'Text_3',
 'Text_4',
 'Sub_Object_Description',
 'Location_Description',
 'FTE',
 'Function_Description',
 'Facility_or_Department',
 'Position_Extra',
 'Total',
 'Program_Description',
 'Fund_Description',
 'Text_1']

# Remove commas and quotes again
for column in test.columns.tolist():
	if test[column].dtype == 'object':
		test[column] = test[column].str.replace(',',' ')
		test[column] = test[column].str.replace('"','')

test[train_col_order].to_csv('TestData2.csv',index=False)
