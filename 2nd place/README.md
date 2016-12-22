![Banner Image](https://s3.amazonaws.com/drivendata/comp_images/4.jpg)
# Box-Plots for Education - 2nd Place
<br><br>
# Model Descriptions

Solution divide on 3 parts:

1. Log model by XXXXXX

2. RF model by XXXXXX

3. Merge two models and replace all duplicates from test set
XXXXXX describe his own part. I describe part 2 and part 3.

### 1.
The first models were developed by XXXXXX which used Stocastic Gradient Descent. This model was later combined with XXXXXX's
Random Forest model and duplicate removal to obtain 2nd rank in the competition. The idea was to keep the model as simple as possible.

The problem was treated as an NLP problem rather than a machine learning problem with some structured dataset. Most of the NLP problems
proceed with cleaning of the dataset and tokenization. We also started with basic cleaning and removal of certain columns. The columns,
"FTE" and "Total" were removed since they consisted a lot of NaN values and did not bring any usefulness to the model. The remaining columns
were tokenized and later joined as a single column in the training and testing datasets.

The labels extracted from the training data were encoded columnwise and the problem was treated as a multiclass classification problem, i.e,
for every column in the labels, a separate model was trained.

The features used were the general NLP features like TFIDF (term frequency-inverse document frequency) transformation and hashing trick.
We used sklearn for our implementation. Several ngrams like (1,1), (2,1), (2,2) etc. were tried and it was found that an ngram range of
(2,1) performs best on the provided dataset. We eliminated words with less than 10 occurences. The final features were TFIDF + Hashing.

We started with a single model approach and trained Stocastic Gradient Classifier with logistic loss on the whole dataset.
This approach alone gave a competitive score on the Leaderboard.

Since multiclass-multilabel-logloss penalizes heavily for incorrect predictions, the model was made stable and generalizable with
lower logoloss by adding two extra SGD classifiers with different penalties. The predictions from these three models were combined in a
weighted average manner to obtain the final predictions for the test set.

Models :
	SGD Classifier: loss = log, alpha = 0.000001, iterations = 120, penalty = l2
	SGD Classifier: loss = log, alpha = 0.0001, iterations = 120, penalty = l1
	SGD Classifier: loss = log, alpha = 0.0001, iterations = 120, penalty = elastic net

Final Model = (w1*M1 + w2*M2 + w3*M3)

Different weights for the three models were found out using both the cross-validation and using the leaderboard response. Since the test dataset
was entirely different from the training dataset, the cross-validation step played a very small role and we were entirely dependent on the leaderboard
response. Given the size of the datasets and careful feature extraction without depending on the training set, we were not worried about overfitting at all.

### 2.
There are many class columns in data. The standard methods – encode it (one-vs-rest, hash, etc…) and apply some linear algorithms. Therefore, the best way to improve model is to create principally different method from class of linear methods.
The main idea for RF model is very simple. We just concatenate train and test sets, and then for each column we count frequencies of each value. Then we sort these frequencies and replace it to its disjoint rank. For example:
Data column: ‘a’, ‘d’, ‘a’, ‘c’, ‘e’, ‘b’, ‘d’, ‘c’, ‘a’
So, frequencies for each value:
‘a’ = 3, ‘c’ = 2, ‘d’ = 2, ‘b’ = 1, ‘e’ = 1
Replace to its disjoint rank:
‘a’ -> 0, ‘c’ -> 1, ‘d’ -> 2, ‘b’ -> 3, ‘e’ -> 4
Transformed data column: 0, 2, 0, 1, 4, 3, 2, 1, 0
We concatenate train and test sets, transformed it as described and then split to train and test sets again. Then we can use Random Forest Classifier on this data, because it turned into numerical data. We will use it to predict each variable separately.

### 3.
So, we have two different models. Then we normalized each of them and merged with coefficients:
Final model = alpha * Log model + (1 – alpha) * RF model
Also we used such trick – if for one rows all class columns in train data matches for row in test data and if for all such rows in train data they have the same answer, then we replace our model answer into test to this answer from train. This is unlikely to be useful in business and real problems. But on this problem it helps a little.

Thanks!
