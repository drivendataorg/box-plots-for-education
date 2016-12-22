![Banner Image](https://s3.amazonaws.com/drivendata/comp_images/4.jpg)
# Box-Plots for Education - 3rd Place
<br><br>
# Entrant Background and Submission Overview

### Mini-bio
I have a MS in Electronic Engineering. In the past 16 years I've been working as an electronic hardware design engineer for big multinationals like XXXXXX and XXXXXX and later as an Automation Engineer in XXXXXX. In 2008 I start learn by myself techniques of machine learning, first turned to improve stock market gains, later I leaned a lot of techniques to improve generic predictions. In 2012 I joined Kaggle competition site and win a couple of competitions and achieved good results in some others, improving my machine learning skills in many areas.

### High Level Summary of Submission
My approach is based in a Gradient Boosted Machine, so all text must be converter to an identification id (number).  First the data was preprocessed to lower case all text, remove punctuation and apply a naive stemmer to all phrases. Then for each feature, as there are many repeated text, each text is converted to a number. Doing that results in a numeric matrix. For the training I used a gbm algorithm that trains and predicts each target from 1 to 9 separately. GBM was choose because it explores the interaction between features and there are many similar texts in the dataset that can be converted to a number (gbm only handle numbers). Also this algorithm can minimize multiclass error classification, according the metric chosen for the competition.

### Omitted Work
I tried to train using different algorithm like Vowpal Wabit, random Forest, Ridge and Logistic Regression, but none seems to fit the competition metric directly (without any further processing). And also these algorithms didn’t perform well in local cross validations.

### Tools Used
Data preparation, analysis and training was done in R.

### Model Evaluation
I evaluated the performance using a 15 fold cross validation in trainset using the provided metric.

### Future Steps
In the end of the competition I found that some testset instances are exactly the same as in the trainset. This characteristic could be better explored in future work.
I could explore ensembling different models also. Training other algorithms like Random Forest and Logistic Regression could potentially improve results, even if those other models have lower performance.

<br><br>
# Replicating the Submission

Instruction for Linux Ubuntu 14.04, probably also runs under Windows

### Install R Language
* Install R Studio
* Open R studio
* Install packages:
    * xgboost
    * data.table

### Run script train1.R    
   This script will load datasets, preprocess trainset and testset, generate extra features, train all 9 targets over 15 folds and using 8 cpus (it will take a loooong time), then saves the results;

### Run script train2.R    
   This script will load datasets, generate extra features, train all 9 targets over 15 folds and using 8 cpus (it will take a loooong time), then saves the results;

### Run script submit.R
   This script will load train1 and train2 predictions, ensemble both and then create a submission file named ens1.csv;

### Change Header for Submission
Now using any text editor (I recommend Notepad++), replace ens1.csv header with original SubmissionFormat.csv header;

DONE!!!
