from data import Data
from preprocessing import Preprocess
from classifier import Classifier
from pymysql import DB
import sys


class Project:
    def __init__(self, num_rows, wrt_feature='bugs'):
    	self.db = DB(db='major_2')
    	data = Data(self.db)
    	self.data_set = data.getData(num_rows)

    def preprocess(self, vector='hashing'):
    	self.preprocess = Preprocess(self.data_set)
        self.preprocess.vectorize(vector)

    def run_classifier(self, classifier='svc'):
        self.classifier = Classifier(
            self.preprocess.severity.keys(), self.preprocess.X_train,
            self.preprocess.y_train, self.preprocess.X_test,
            self.preprocess.y_test, self.preprocess.train_size,
            self.preprocess.test_size)
        self.classifier.classify(classifier)
	# Classifier add karna hai yaha pe.

def main(num_rows=10000, vector='hashing', classifier='svc'):
    print "##########  Running %s classifier for %d rows of data. ##########\n\n\n" % (classifier, num_rows)
    project = Project(num_rows)
    project.preprocess(vector)
    project.run_classifier(classifier)

if __name__ == '__main__':
    # try:
    if sys.argv[1] == 'default':
        main()
    else:
        num_rows = int(sys.argv[1])
        vector = sys.argv[2]
        classifier = sys.argv[3]
        main(num_rows, vector, classifier)
    # except:
        # print "\nUsage: \npython main.py [number of rows] [vector type] [classification model]\nor\n python main.py default"

