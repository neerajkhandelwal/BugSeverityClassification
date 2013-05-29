from data import Data
from preprocessing import Preprocess
from classifier import Classifier
from classifier import PipeLineClassifier
from pymysql import DB
from copy import deepcopy
import sys


class Project:
    def __init__(self, num_rows, wrt_feature):
    	self.db = DB(db='major_2')
    	data = Data(self.db)
    	self.data_set = data.getData(num_rows, wrt_feature)

    def preprocess(self, new, vector, chi2):
    	self.prepro = Preprocess(self.data_set, new)
        if new == 'True':
            if vector == 'hashing':
                self.prepro.hashVector()
            if vector == 'tfidf':
                self.prepro.tfidfVector()
            # print self.preprocess.y_train
        else:
            self.prepro.vectorize(vector)
        if chi2:
            self.prepro.chisquare()

    def run_classifier(self, method, classifier):
        if method == 'classifier':
            self.classifier = Classifier(
                self.prepro.severity.keys(), self.prepro.X_train,
                self.prepro.y_train, self.prepro.X_test,
                self.prepro.y_test, self.prepro.train_size,
                self.prepro.test_size)
            self.classifier.classify(classifier)
        if method == 'pipeline':
            self.classifier = PipeLineClassifier(
                self.prepro.severity.keys(), self.prepro.train_corpus,
                self.prepro.y_train, self.prepro.X_test,
                self.prepro.y_test, self.prepro.train_size,
                self.prepro.test_size
            )
            self.classifier.setVariables(classifier)
            self.classifier.benchmark()
	# Classifier add karna hai yaha pe.

def main(num_rows=10000, new=False, method='classifier', vector='hashing', classifier='svc', chi2=True, base='all'):
    print "##########  Running %s classifier for %d rows of data. ##########\n\n\n" % (classifier, num_rows)
    project = Project(num_rows, base)
    if base != 'all':
        data = deepcopy(project.data_set)
        for key in data:
            print "##########  Running for %s platform for %d rows of data. ##########\n\n\n" % (key, len(data[key]))
            project.data_set = deepcopy(data[key])
            project.preprocess(new, vector=vector, chi2=chi2)
            project.run_classifier(method, classifier)
    else:
        project.preprocess(new, vector=vector, chi2=chi2)
        project.run_classifier(method, classifier)

if __name__ == '__main__':
    try:
        default = ''
        if len(sys.argv) == 2:
            default = sys.argv[1]
        else:
            base = sys.argv[1]
            method = sys.argv[2]
            new = sys.argv[3]
            num_rows = int(sys.argv[4])
            vector = sys.argv[5]
            classifier = sys.argv[6]
            try:
                chi2 = sys.argv[7]
                chi2 = True
            except:
                chi2 = False
    except:
        print "\nUsage: "
        print "python main.py [base(all/platform/os)] [method(pipeline/classifier)] [not use test train split(True/False)] [number of rows] [vector type(hashing/tfidf/count)] [classification model(linear/svc/gnb/gd)] [chi2(chi2)]"
        print "\nor\n\npython main.py default"
        print "\nNote: All arguments are required. 'false' for 'not use test train split' do not work. chi2 is optional"

        sys.exit(0)

    if default == 'default':
        main()
    else:
        main(num_rows, new, method, vector, classifier, chi2, base)


