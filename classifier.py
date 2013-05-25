from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier

class Classifier:
	# classifier 'svc' is C implementation of SVM.

	def benchmark(self, classification):
		print "##########  Testing and Benchmarking the classifier. ##########\n\n\n"

		if self.classifier == 'gnb':
			prediction = classification.predict(self.X_test.toarray())
		else:
			prediction = classification.predict(self.X_test)

		print "##########  Results. ##########\n\n\n"

		print "f1-score: %0.3f" % metrics.f1_score(self.y_test, prediction)
		print "Accuracy: %0.2f percent" % (metrics.accuracy_score(self.y_test, prediction)*100)
		print "Precision: %0.3f" % metrics.precision_score(self.y_test, prediction)
		print "Recall: %0.3f" % metrics.recall_score(self.y_test, prediction)
		print "Classification report: \n"
		print metrics.classification_report(self.y_test, prediction, target_names=self.classes)

	# Not a multiclass algo.
	# def KNeighbors(self):
	# 	# weights => uniform and distance
	# 	classification = KNeighborsClassifier(n_neighbors=10, weights='distance')
	# 	classification.fit(self.X_train, self.y_train)
	# 	return classification

	def SVCclassifier(self):
		# Classification is not good though result is quite good. Classifying everything as normal.
		classification = svm.SVC(cache_size=500)
		classification.fit(self.X_train, self.y_train)
		return classification

	def LinearSVCclassifier(self):
		classification = svm.LinearSVC(penalty='l1', dual=False)
		classification.fit_transform(self.X_train, self.y_train)
		return classification

	def GaussianNBayes(self):
		classification = GaussianNB()
		classification.fit(self.X_train.toarray(), self.y_train)
		return classification

	def classify(self, classifier):
		self.classifier = classifier
		print "##########  Training using %s classifier. ##########\n\n\n" % classifier
		if classifier == 'svc':
			self.benchmark(self.SVCclassifier())
		if classifier == 'linear':
			self.benchmark(self.LinearSVCclassifier())
		if classifier == 'gnb':
			self.benchmark(self.GaussianNBayes())
		# if classifier == 'kneighbors':
		# 	self.benchmark(self.KNeighbors())

	def __init__(self, classes, X_train, y_train, X_test, y_test, train_size, test_size):
		self.classes = classes
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		self.train_size = train_size
		self.test_size = test_size
		# print self.testing_data

				# print self.vector_matrix
		# Analyzes the text, basically get the features.
		# self.analyzefn = vectorizer.build_analyzer()
	
		# transform method to be used for non existent words, for testing.
		# print vectorizer.get_feature_names()