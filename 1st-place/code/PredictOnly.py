
##############################################################################
# Online.py - an Online Learning Model for Semi-Structured Text Classification
#
# For: DrivenData.org/ERS's BoxPlots for Education Competition
# License: MIT (http://machinelearner.net/boxplots-for-education-1st-place/LICENSE.txt)
#
# This approach is based on tingrtu's Online Learning masterpiece: http://bit.ly/1ItCVcv
# It is customized for specifying flexible interactions between features and
# for decomposing original feature text into "bag of words" tokens.
##############################################################################

from datetime import datetime
from math import log, exp, sqrt
import pickle
import sys
import random
import math
import re

test = sys.argv[1]

# Specify which original features to keep and discard in the model
# Intercept = 0
# Object_Description = 1
# Text_2 = 2
# SubFund_Description = 3
# Job_Title_Description = 4
# Text_3 = 5
# Text_4 = 6
# Sub_Object_Description = 7
# Location_Description = 8
# FTE = 9
# Function_Description = 10
# Facility_or_Department = 11
# Position_Extra = 12
# Total = 13
# Program_Description = 14
# Fund_Description = 15
# Text_1 = 16

originals = range(17)

# Found that removing 5 (Text_3) and 7 (Sub Object Description) generaly helped
originals.remove(5)
originals.remove(7)

# Interaction pairs and triples
pairs = [[1,2,3,4],[6,8],[4,12],[1,4,8,10]]
triples = [[1,4,12]]

print 'pairs',pairs
print 'triples',triples

D = 2 ** 18  # number of weights use for each model, we have 104 of them
alpha = .10   # learning rate for sgd optimization

# utilities ############################################

# Used for assigning the number feat to a categorical level 0 to N
# INPUT:
#     feat: the numerical predictor
#     b: list representing the boundaries for bins
# OUTPUT:
#     a categorical level 0 to N
def boundary(feat,b):
    f = float(feat)
    s = 0
    for step in b:
        if f < step:
            return s
        s += 1
    return s

# Our hashing function
# INPUT:
#     s: the string or number
# OUTPUT:
#     an integer between 0 and D-1
def hash_it(s):
    return abs(hash(s)) % D


# function, generator definitions ############################################

# A. x, y generator
# This is where:
# * All the feature hashes are generated
# * All feature engineering happens
# INPUT:
#     path: path to TrainPredictors.csv or TestData2.csv
#     label_path: (optional) path to TrainLabels.csv
# YIELDS:
#     ID: id of the instance
#     x: list of hashes for predictors
#     y: (if label_path is present) binary label
def data(path, label_path=None):
    # Boundaries for numerical binning of FTE (9) and Total (13)
    b13 = [-706.968,-8.879,
    7.85,41.972,
    73.798,109.55,
    160.786,219.736,
    318.619,461.23,
    646.73,938.36,
    1317.584,2132.933,
    3652.662,6659.524,
    18551.459,39754.287,
    64813.342,129700000]

    b9 = [0.0,0.00431,0.131,0.911,1,50]

    for t, line in enumerate(open(path)):
        # Intcercept term
        x = [0]

        # Skip headers
        if t == 0:
            if label_path:
                label = open(label_path)
                label.readline()  # we don't need the headers
            continue

        # c is an index for the kept original features (15 of them)
        # TODO: drop c and use m for hashing, c was kept for reproducibility
        # m is the index for all the original features (17 of them)
        # feat is the original raw text or value for feature
        c =0
        for m, feat in enumerate(line.rstrip().split(',')):
            # Drop unwanted original features
            if m not in originals:
                continue

            if m == 0:
                ID = int(feat)
            else:
                # convert floats into categorical levels
                # variables 9 (FTE) and 13 (Total) are only numericals
                if m == 13:
                    if feat == "": feat = 0
                    feat = boundary(feat,b13)
                if m == 9:
                    if feat == "": feat = -3
                    feat = boundary(feat,b9)

                # Lowercase and trim so hashes match more often
                feat = str(feat).strip().lower()

                # First we hash the original feature.  For example, if the
                # feature is "special education" and the original feature index is 4, we
                # hash "4_special education"

                original_feature = str(c) + '_' + feat
                x.append( hash_it(original_feature) )

                # Next we break up the original feature value into word parts
                # i.e. create bag-of-word features here
                parts = re.split(' |/|-',feat)

                for i in range(len(parts)):
                    token = parts[i].strip().lower()
                    if token == '': continue

                    # First we hash each token along with its original position index
                    # For example, for the feature value "special education" we hash
                    # its tokens as "4_special" and "4_education" in successive steps of this loop
                    positioned_word = str(c) + '_' + token
                    x.append( hash_it( positioned_word ) )

                    # Next we hash each token by itself, ignoring any information about its position
                    # For example, for "special education" we hash "special" and "education"
                    # regardless of what index position the original feature appeared in.
                    # This views all the feature values in an example as making up a single document
                    x.append( hash_it( token ) )

                c = c + 1

        # Up to this point we've been breaking original features down into smaller features
        # Now we level up and compose original features with each other into larger interction features

        row = line.rstrip().split(',')

        # Start with pairs.  Make pairs from interaction groups defined in pairs variable.
        for interactions in pairs:
            for i in xrange(len(interactions)):
                for j in xrange(i+1,len(interactions)):
                    pair = row[interactions[i]]+"_x_"+row[interactions[j]]
                    x.append( hash_it(pair) )

        # Do the same thing for triples
        for triple in triples:
            trip = row[triple[0]]+"_x_"+row[triple[1]] + '_x_' +row[triple[2]]
            x.append( hash_it(trip) )

        if label_path:
            y = [float(y) for y in label.readline().split(',')[1:]]

        yield (ID, x, y) if label_path else (ID, x)

# C. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def predict(x, w):
    wTx = 0.
    for i in x:  # do wTx
        wTx += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
    return 1. / (1. + exp(-max(min(wTx, 100.), -100.)))  # bounded sigmoid


# training and testing #######################################################
start = datetime.now()
# Number of models.
DIM = 104
K = range(DIM)
w = [[0.] * D for k in range(DIM)]
n = [[0.] * D for k in range(DIM)]

random.seed(1234)

h = ',Function__Aides Compensation,Function__Career & Academic Counseling,Function__Communications,Function__Curriculum Development,Function__Data Processing & Information Services,Function__Development & Fundraising,Function__Enrichment,Function__Extended Time & Tutoring,Function__Facilities & Maintenance,Function__Facilities Planning,"Function__Finance, Budget, Purchasing & Distribution",Function__Food Services,Function__Governance,Function__Human Resources,Function__Instructional Materials & Supplies,Function__Insurance,Function__Legal,Function__Library & Media,Function__NO_LABEL,Function__Other Compensation,Function__Other Non-Compensation,Function__Parent & Community Relations,Function__Physical Health & Services,Function__Professional Development,Function__Recruitment,Function__Research & Accountability,Function__School Administration,Function__School Supervision,Function__Security & Safety,Function__Social & Emotional,Function__Special Population Program Management & Support,Function__Student Assignment,Function__Student Transportation,Function__Substitute Compensation,Function__Teacher Compensation,Function__Untracked Budget Set-Aside,Function__Utilities,Object_Type__Base Salary/Compensation,Object_Type__Benefits,Object_Type__Contracted Services,Object_Type__Equipment & Equipment Lease,Object_Type__NO_LABEL,Object_Type__Other Compensation/Stipend,Object_Type__Other Non-Compensation,Object_Type__Rent/Utilities,Object_Type__Substitute Compensation,Object_Type__Supplies/Materials,Object_Type__Travel & Conferences,Operating_Status__Non-Operating,"Operating_Status__Operating, Not PreK-12",Operating_Status__PreK-12 Operating,Position_Type__(Exec) Director,Position_Type__Area Officers,Position_Type__Club Advisor/Coach,Position_Type__Coordinator/Manager,Position_Type__Custodian,Position_Type__Guidance Counselor,Position_Type__Instructional Coach,Position_Type__Librarian,Position_Type__NO_LABEL,Position_Type__Non-Position,Position_Type__Nurse,Position_Type__Nurse Aide,Position_Type__Occupational Therapist,Position_Type__Other,Position_Type__Physical Therapist,Position_Type__Principal,Position_Type__Psychologist,Position_Type__School Monitor/Security,Position_Type__Sec/Clerk/Other Admin,Position_Type__Social Worker,Position_Type__Speech Therapist,Position_Type__Substitute,Position_Type__TA,Position_Type__Teacher,Position_Type__Vice Principal,Pre_K__NO_LABEL,Pre_K__Non PreK,Pre_K__PreK,Reporting__NO_LABEL,Reporting__Non-School,Reporting__School,Sharing__Leadership & Management,Sharing__NO_LABEL,Sharing__School Reported,Sharing__School on Central Budgets,Sharing__Shared Services,Student_Type__Alternative,Student_Type__At Risk,Student_Type__ELL,Student_Type__Gifted,Student_Type__NO_LABEL,Student_Type__Poverty,Student_Type__PreK,Student_Type__Special Education,Student_Type__Unspecified,Use__Business Services,Use__ISPD,Use__Instruction,Use__Leadership,Use__NO_LABEL,Use__O&M,Use__Pupil Services & Enrichment,Use__Untracked Budget Set-Aside'

# write out weights
print('reading weights')
with open('weights.pkl', 'r') as f:
    w = pickle.load(f)

output = './submission1234.csv'

with open(output, 'w') as outfile:
    outfile.write(h + '\n')
    for ID, x in data(test):
        outfile.write(str(ID))
        for k in K:
            p = predict(x, w[k])
            outfile.write(',%s' % str(p))
        outfile.write('\n')

print('Done, elapsed time: %s' % str(datetime.now() - start))
