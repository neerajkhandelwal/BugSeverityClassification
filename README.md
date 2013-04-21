# Bug Severity Classification
Software that find out severity of the bugs and classifies them according to it. The totally supervised apprach using various algorithms. It is implemented in Python using the Scikit library.

**Thanks to Mozilla for their bug repository.**


### Main.py
Main file which uses preprocessing, data, classifier. Takes in the command line argument [number of rows], [vector type] and [classifier] in the particular order or just use default for 10000 rows, hashing vector and svm with rbf kernel.

### Preprocessing.py
Preprocess the data to give the training and testing set from the data set. Also creates the vectors or say finds the features from the data set.

### Data.py
Uses th pymysql.py file for the cursor to run the query and creates the data set. It inherits the pymysql's DB class.

### Classifier.py
Uses the variable from the preprocessing file to initialize then perform the uses the specified classification algorithms over it. Also gives the result in form of precision, recall and accuracy.

===

# Learning
- SVC with rbf kernel is too slow for the larger data set also the it classifies the majority of test data to a single class.
- LinearSVC is quite fast even for half a million data set. Performs better than the SVC with rbf kernel.

===

## How to run?
Change the query and connection to your database in the data.py file without changing the variables. Rest all is taken care of.

===

#### Note:
- Currently implemented: Linear SVC and SVC.
- Classifier: linear, svc
- Vectors type: hashing, count, tfidf
- number of rows: depends upon your data set.