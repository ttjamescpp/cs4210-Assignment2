# -------------------------------------------------------------------------
# AUTHOR: Tommy James
# FILENAME: knn.py
# SPECIFICATION: Complete the Python program (knn.py) that will read the file binary_points.csv
# and output the LOO-CV error rate for 1NN
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 week
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

# reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)

rows = len(db)
cols = len(db[0]) - 1

# loop your data to allow each instance to be your test set
k = 0
wrong_predict = 0

for data in db:
    # add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    # --> add your Python code here
    X = [[0 for i in range(cols)] for j in range(rows)]

    for i in range(rows):
        for j in range(cols):
            X[i][j] = float(db[i][j])

    # transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    # --> add your Python code here
    Y = [0 for i in range(rows)]

    class_vals = {
        '+': 1,
        '-': 2
    }

    for i in range(rows):
        if db[i][cols] in class_vals:
            Y[i] = float(class_vals[db[i][cols]])

    true_label = Y.pop(k)

    # store the test sample of this iteration in the vector testSample
    # --> add your Python code here
    testSample = [[0 for i in range(cols)] for j in range(rows)]
    testSample[k] = X.pop(k)

    # fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    # use your test sample in this iteration to make the class prediction. For instance:
    # class_predicted = clf.predict([[1, 2]])[0]
    # --> add your Python code here
    class_predict = clf.predict(testSample[k:k+1])

    # compare the prediction with the true label of the test instance to start calculating the error rate.
    # --> add your Python code here
    if class_predict != true_label:
        wrong_predict = wrong_predict + 1

    k += 1

# print the error rate
# --> add your Python code here
error = wrong_predict / rows
print(f'wrong predictions: {wrong_predict}/{rows}')
print(f'Error rate = {error}')
