
# all imports
import pandas as pd
import numpy as np
import re
from scipy import sparse
from sklearn import preprocessing, linear_model
import sklearn.feature_extraction.text as txt

def clean(s):
	"""
	Function for cleaning the text data. Convert to Unicode and check for missing values
	"""
    try:
        return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()
    except:
        return " ".join(re.findall(r'\w+', "no_text",flags = re.UNICODE | re.LOCALE)).lower()

# load training and test datasets and sample submission
train = pd.read_csv('data/TrainingData.csv')
test = pd.read_csv('data/TestData.csv')
sample = pd.read_csv('data/SubmissionFormat.csv')

# extract training and test columns and separate labels
training_data = train[['FTE','Facility_or_Department', 'Function_Description','Fund_Description',
                       'Job_Title_Description', 'Location_Description','Object_Description',
                       'Position_Extra', 'Program_Description', 'SubFund_Description',
                       'Sub_Object_Description', 'Text_1', 'Text_2','Text_3','Text_4', 'Total']]

labels = np.array(train[['Function','Object_Type','Operating_Status','Position_Type','Pre_K', 'Reporting',
                'Sharing','Student_Type', 'Use']])

test_data = test[['FTE','Facility_or_Department', 'Function_Description','Fund_Description',
                       'Job_Title_Description', 'Location_Description','Object_Description',
                       'Position_Extra', 'Program_Description', 'SubFund_Description',
                       'Sub_Object_Description', 'Text_1', 'Text_2','Text_3','Text_4', 'Total']]

# encode provided labels to numbers
label_encoder = preprocessing.LabelEncoder()
for i in range(labels.shape[1]):
    labels[:,i] = label_encoder.fit_transform(np.array(labels[:,i]))

# drop FTE and Total columns
training_data = training_data.drop('FTE', axis = 1)
training_data = training_data.drop('Total', axis = 1)

test_data = test_data.drop('FTE', axis = 1)
test_data = test_data.drop('Total', axis = 1)

# data cleaning for remaining columns
training_data['Facility_or_Department'] = training_data['Facility_or_Department'].apply(clean)
training_data['Function_Description'] = training_data['Function_Description'].apply(clean)
training_data['Fund_Description'] = training_data['Fund_Description'].apply(clean)
training_data['Job_Title_Description'] = training_data['Job_Title_Description'].apply(clean)
training_data['Location_Description'] = training_data['Location_Description'].apply(clean)
training_data['Object_Description'] = training_data['Object_Description'].apply(clean)
training_data['Position_Extra'] = training_data['Position_Extra'].apply(clean)
training_data['Program_Description'] = training_data['Program_Description'].apply(clean)
training_data['SubFund_Description'] = training_data['SubFund_Description'].apply(clean)
training_data['Sub_Object_Description'] = training_data['Sub_Object_Description'].apply(clean)
training_data['Text_1'] = training_data['Text_1'].apply(clean)
training_data['Text_2'] = training_data['Text_2'].apply(clean)
training_data['Text_3'] = training_data['Text_3'].apply(clean)
training_data['Text_4'] = training_data['Text_4'].apply(clean)

test_data['Facility_or_Department'] = test_data['Facility_or_Department'].apply(clean)
test_data['Function_Description'] = test_data['Function_Description'].apply(clean)
test_data['Fund_Description'] = test_data['Fund_Description'].apply(clean)
test_data['Job_Title_Description'] = test_data['Job_Title_Description'].apply(clean)
test_data['Location_Description'] = test_data['Location_Description'].apply(clean)
test_data['Object_Description'] = test_data['Object_Description'].apply(clean)
test_data['Position_Extra'] = test_data['Position_Extra'].apply(clean)
test_data['Program_Description'] = test_data['Program_Description'].apply(clean)
test_data['SubFund_Description'] = test_data['SubFund_Description'].apply(clean)
test_data['Sub_Object_Description'] = test_data['Sub_Object_Description'].apply(clean)
test_data['Text_1'] = test_data['Text_1'].apply(clean)
test_data['Text_2'] = test_data['Text_2'].apply(clean)
test_data['Text_3'] = test_data['Text_3'].apply(clean)
test_data['Text_4'] = test_data['Text_4'].apply(clean)

# create a single new column for cleaned text data
training_data["combined"] = [' '.join(row) for row in training_data[training_data.columns].values]
test_data["combined"] = [' '.join(row) for row in test_data[test_data.columns].values]

# initialize TFIDF vectorizer and Hashing Vectorizer
tfidf = txt.TfidfVectorizer(ngram_range=(2, 1), max_df=1.0, min_df=10)
hsv = txt.HashingVectorizer()

# fit tfidf and hashing vectorizer to train data
tfidf.fit(training_data['combined'])
hsv.fit(test_data['combined'])

# transform the training and test datasets to obtain a sparse matrix
X_tfidf = tfidf.transform(training_data['combined'])
X_test_tfidf = tfidf.transform(test_data['combined'])

X_hsv = hsv.transform(training_data['combined'])
X_test_hsv = hsv.transform(test_data['combined'])

X = sparse.hstack((X_hsv, X_tfidf))
X_test = sparse.hstack((X_test_hsv, X_test_tfidf))

# training and prediction on the the test dataset
preds1 = []
preds2 = []
preds3 = []
for i in range(labels.shape[1]):
    print "label = ", i
    sgd1 = linear_model.SGDClassifier(loss = 'log', n_iter = 120, alpha = 0.000001)
    sgd2 = linear_model.SGDClassifier(loss = 'log', n_iter = 120, penalty = 'l1')
    sgd3 = linear_model.SGDClassifier(loss = 'log', n_iter = 120, penalty = 'elasticnet')
    sgd1.fit(X, labels[:,i].astype(int))
    sgd2.fit(X, labels[:,i].astype(int))
    sgd3.fit(X, labels[:,i].astype(int))
    if i == 0:
        preds1 = sgd1.predict_proba(X_test)
        preds2 = sgd2.predict_proba(X_test)
        preds3 = sgd3.predict_proba(X_test)
    else:
        preds1 = np.hstack((preds1,sgd1.predict_proba(X_test)))
        preds2 = np.hstack((preds2,sgd2.predict_proba(X_test)))
        preds3 = np.hstack((preds3,sgd3.predict_proba(X_test)))

# weighted average of all predictions is the final prediction for test set.
preds = (preds1 + preds2 + preds3)/3.0

for i in range(1, len(sample.columns)):
    colname = sample.columns[i]
    sample[str(colname)] = probs[:,i-1]

sample.to_csv('Log_Model.csv', index = False)
