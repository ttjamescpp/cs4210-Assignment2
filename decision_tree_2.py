# -------------------------------------------------------------------------
# AUTHOR: Tommy James
# FILENAME: decision_tree_2.py
# SPECIFICATION: Your goal is to train, test, and output the performance of the 3 models
# created by using each training set on the test set provided
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 week
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv',
            'contact_lens_training_2.csv', 'contact_lens_training_3.csv']


average = 0
for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    # reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    rows = len(dbTraining)
    cols = len(dbTraining[0])-1

    # transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    # --> add your Python code here

    # feature values to convert to
    feature_vals = {
        'Young': 1,
        'Prepresbyopic': 2,
        'Presbyopic': 3,
        'Myope': 1,
        'Hypermetrope': 2,
        'Yes': 1,
        'No': 2,
        'Normal': 1,
        'Reduced': 2,
    }

    # create 2D array
    X = [[0 for i in range(cols)] for j in range(rows)]

    # transform feature values
    def trans_feats(db, rows, cols, X, feature_vals):
        for i in range(rows):
            for j in range(cols):
                if db[i][j] in feature_vals:
                    X[i][j] = feature_vals[db[i][j]]

        return X

    X = trans_feats(dbTraining, rows, cols, X, feature_vals)

    # transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    # --> add your Python code here
    Y = [0 for i in range(rows)]

    # transform classes
    def trans_class(db, rows, cols, feature_vals, Y):
        for i in range(rows):
            if db[i][cols] in feature_vals:
                Y[i] = feature_vals[db[i][cols]]
        return Y

    Y = trans_class(dbTraining, rows, cols, feature_vals, Y)

    accuracy = 0
    # loop your training and test tasks 10 times here
    for i in range(10):

        # fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)

        # read the test data and add this data to dbTest
        # --> add your Python code here
        dbTest = []

        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0:  # skipping the header
                    dbTest.append(row)

        rows_test = len(dbTest)
        cols_test = len(dbTest[0])-1

        # create arrays for test data
        X_test = [[0 for i in range(cols_test)]
                  for j in range(rows_test)]  # features
        Y_test = [0 for i in range(rows)]  # classes

        j = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for data in dbTest:
            # transform the features of the test instances to numbers following the same strategy done during training,
            X_test = trans_feats(
                dbTest, rows_test, cols_test, X_test, feature_vals)
            Y_test = trans_class(
                dbTest, rows_test, cols_test, Y_test, feature_vals)

            # and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            # where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            # --> add your Python code here
            class_predicted = clf.predict(X_test[j:j+1])[0]

            # compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            # --> add your Python code here
            if class_predicted == Y[j] and class_predicted == 1:
                tp += 1
            elif class_predicted == Y[j] and class_predicted == 2:
                tn += 1

            j += 1

        accuracy = accuracy + ((tp + tn) / rows_test)

    # find the average of this model during the 10 runs (training and test set)
    # --> add your Python code here
    average = (accuracy / 10)

    # print the average accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    # --> add your Python code here
    print(f'The average accuracy when training on {ds}: {average}')
