![Banner Image](https://s3.amazonaws.com/drivendata/comp_images/4.jpg)
# Box-Plots for Education - 1st Place
<br><br>
# Entrant Background and Submission Overview

### Mini-bio
My name is XXXXXX and I am an Engineering Manager at XXXXXX. I recently
completed my Masters in Predictive Analytics at XXXXXX. Although I
started my current job in engineering leadership, I have increasingly been working
more on the data science side of the house, for example I am a part of an initiative to
upgrade our inhouse
sentiment classification algorithms and general NLP capabilities
within XXXXXX social products. I plan to transition to a fulltime
Data Scientist
role by this Spring.

### High Level Summary of Submission
My model is based on Online Learning, specifically a Logistic Regression model that
uses the hashing trick and stochastic gradient descent with an adaptive learning rate.
I owe a big debt of gratitude to [tingrtu](http://www.kaggle.com/users/185835/tinrtgu) for
showing the effectiveness of this technique for multi
text classification in other
competitions. I started with tingrtu’s [Python-based
online learner](http://bit.ly/1ItCVcv),
and in addition I used Python/Pandas for Data Prep & EDA, then straight Python for
Feature Engineering.

Feature engineering strategy for the text-based
features figured prominently into my
model’s performance. While I used some standard techniques for creating 2way
and
3way
interactions between the original features, what made the biggest difference
was decomposing the text features into tokens or single word features (about 0.09 on
the public leaderboard. See code and below for details). This allowed for the
model to correctly classify documents even if the original feature text was coded
slightly differently in different examples.

From a competition point of view, using Online Learning worked well because the
feature hashing is a sparse representation that reduces the model's memory footprint,
and SGD makes quick passes through the data. This made it computationally feasible
and fast to try a lot of changes and cover ground more quickly. Other algorithms like
random forests with sklearn or deep neural networks required more computational
horsepower than my particular laptop could give. After the competition, I would like to
scale up my machine learning to a cluster and try some of these approaches.

For ERS’s implementation, I also recommend the Online Learning approach because
it is a simple model that is easy to code and roll out. It also has the advantage of
being “online” and incrementally updatable in a production system. This means it
probably stands a better chance of keeping up to date with variations in how the
various text fields are recorded over time, versus a model that needs to be retrained
in batch updated in a code release.

More details can be found
[Here](http://nbviewer.ipython.org/url/machinelearner.net/boxplotsforeducation1stplace/BoxPlots_First_Place_Model.ipynb)

### Omitted Work

There were other tools/algorithms like Boosted Trees (xgboost), Logistic Regression
(via sklearn, Vowpal Wabbit), and Naive Bayes (sklearn) which are all capable of
working with sparse representations of the feature hashes, and I wanted to experiment
with more. Initially I thought that all of these could do at least as well as the Online
Learning approach, even if they took more work to set up 104 models in each tool
(tingrtu’s online learner is set up to handle multiple models simultaneously and
seamlessly).

I experimented with these algorithms by exporting the exact same feature hashes from
the Online Learning model (i.e. all models shared the exact same features).
Then I wrote the code needed to fit 104 models using the specific tools (xgboost,
sklean, vowpal wabbit) and aggregated the results using Python/Pandas.

As it turns out, I could not get a better performing model out of any of them, but I would
say that was mostly due to lack of time. They all seemed to do well insample
and in
cross validation, but performed more poorly than the Online Learning approach on the
test dataset (ranging from 20-100%
worse in terms of average log loss). An important
note though is that these are initial results only and I did no tuning for these models or
detailed investigation into why the results were so different. So the fact that they
underperformed for me is more a reflection only having time to bet on one horse :). I
definitely want to come back to experimenting with these approaches more after the
competition, and understand the adjustments I need to make to improve on the initial
results.

### Tools Used
I used Python/Pandas for data prep and EDA. The data prep code is in the code
submission. I also used a combination of matplotlib and ggplot (for python) for
visualizations this
was mainly to visualize differences in predicted probabilities for
different models.

And iPython notebooks were absolutely essential for productivity during Data Prep,
EDA, and Modeling.

### Model Evaluation
At one point I used “accuracy” (average percent correct for each of the 104 models) as
a measure, using a holdout test dataset. That gave me a more human-interpretable
metric for understanding that the model was doing really well on unseen examples
within the training set.

In general, I abandoned cross validation and holdout testing because there wasn’t a super strong correlation between my holdout test set results and the public leaderboard score. I eventually developed the intuition that the insample
log loss
gave me some sense of how the model would do on the public leaderboard, and that
was as good as it was going to get.

### Future Steps
I probably squeezed everything I could out of the Online Learning model. Next I would
go back and look at why the Logistic Regression with sklearn and L2 regularization did
not perform better. I observed that this algorithm did outperform the Online Learning
model on at least 25% of the 104 models. Would like to understand which of the 104
models were problematic for regularized Logistic Regression.

Would like to do the same for xgboost.

<br><br>
# Replicating the Submission

### Install Requirements
* python 2.7+,
* pypy 2.2.1,
* pandas 0.14.1

### Run Scripts:

1.  Download the 3 original datafiles (`TrainingData.csv`,`TestData.csv`,and `SubmissionFormat.csv)` to the folder  `origdata/`.

2.  Run `python MakeDatasets.py` - this command produces the training and test files `trainPredictors.csv`, `trainLabels.csv`, and `TestData2.csv`.

3.  Run `pypy Online.py 4 0.5` - note that PyPy is required. This command fits an online logistic regression model, taking 4 passes/epochs over the training data with a 50% chance of using an encountered example in each epoch. (The effect of playing with epochs and `use_example_probability` is small for reasonable values - this is just an example configuration but the one used for the winning submission).  The submission will then be placed in the file `submission1234.csv`. Score should be somewhere around 0.3665 (public leaderboard).

For more details see: `BoxPlots_First_Place_Model.ipynb`
