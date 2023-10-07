# -------------------------------------------------------------------------
# AUTHOR: Tommy James
# FILENAME: naive_bayes.py
# SPECIFICATION: read the file
# weather_training.csv (training set) and output the classification of each test instance from the file
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 week
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv
db = []
test1 = []

# reading the training data in a csv file
# --> add your Python code here
# reading the data in a csv file
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)

rows = len(db)
cols = len(db[0])-1

# transform the original training features to numbers and add them to the 4D array X.
# For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
# --> add your Python code here
X = [[0 for i in range(1, cols)] for j in range(rows)]

feature_vals = {
    'Sunny': 1,
    'Overcast': 2,
    'Rain': 3,
    'Hot': 1,
    'Mild': 2,
    'Cool': 3,
    'High': 1,
    'Normal': 2,
    'Weak': 1,
    'Strong': 2,
    'Yes': 1,
    'No': 2
}


def trans_feats(db, rows, cols, X, feature_vals):
    for i in range(rows):
        for j in range(cols-1):
            if db[i][j+1] in feature_vals:
                X[i][j] = feature_vals[db[i][j+1]]

    return X


X = trans_feats(db, rows, cols, X, feature_vals)

# transform the original training classes to numbers and add them to the vector Y.
# For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
# --> add your Python code here
Y = [0 for i in range(rows)]

# transforms classes to numbers and adds to vector Y


def trans_class(db, rows, cols, Y, feature_vals):
    for i in range(rows):
        if db[i][cols] in feature_vals:
            Y[i] = feature_vals[db[i][cols]]

    return Y


Y = trans_class(db, rows, cols, Y, feature_vals)

# fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

# reading the test data in a csv file
# --> add your Python code here
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping header
            test1.append(row)

rows = len(test1)
cols = len(test1[0])-1

X_test = [[0 for i in range(1, cols)] for j in range(rows)]
X_test = trans_feats(test1, rows, cols, X_test, feature_vals)

# printing the header as the solution
# --> add your Python code here
print('Day\t\tOutlook\t\tTemperature\t\tHumidity\t\tWind\t\tPlayTennis\t\tConfidence')
print()

# use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
# --> add your Python code here
# Sunny = 1, Overcast = 2, Rain = 3
# Hot = 1, Mild = 2, Cool = 3
# High = 1, Normal = 2
# Weak = 1, Strong = 2
# Yes = 1, No = 2

# print features, results, and confidence >= 0.75
for i in range(rows):

    result = round(max(clf.predict_proba(X_test[i:i+1])[0]), 2)
    if result >= 0.75:
        # print(f'testing: {X_test[i:i+1]}')
        for j in range(cols):
            print(test1[i][j], end='\t\t')

        if clf.predict(X[i:i+1])[0] == 1:
            print('Yes', end='\t\t')
        else:
            print('No', end='\t\t')

        print(result)
