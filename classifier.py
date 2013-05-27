from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Scaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from pprint import pprint
import numpy as np

class Classifier:
	# classifier 'svc' is C implementation of SVM.

	def crossValidation(self, classification):
		print "##########  Cross Validating and Testing ##########\n\n\n"
		print "##########  Results. ##########\n\n\n"
		scores = cross_validation.cross_val_score(classification, self.X_train, self.y_train, metrics.f1_score, cv=5, n_jobs=1)
		print "F1-score: %.2f (+/- %.2f)" % (scores.mean(), scores.std()/2)
		scores = cross_validation.cross_val_score(classification, self.X_train, self.y_train, metrics.accuracy_score, cv=5, n_jobs=1)
		print "Accuracy: %.2f (+/- %.2f)" % (scores.mean(), scores.std()/2)
		# scores = cross_validation.cross_val_score(classification, self.X_train, self.y_train, metrics.precision_score, cv=5, n_jobs=1)
		# print "Precision: %.2f (+/- %.2f)" % (scores.mean(), scores.std()/2)
		# scores = cross_validation.cross_val_score(classification, self.X_train, self.y_train, metrics.recall_score, cv=5, n_jobs=1)
		# print "Recall: %.2f (+/- %.2f)" % (scores.mean(), scores.std()/2)


	def benchmark(self, classification):
		print "##########  Testing and Benchmarking the classifier. ##########\n\n\n"

		if self.classifier == 'gnb':
			prediction = classification.predict(self.X_test.toarray())
		elif self.classifier == 'gd':
			self.X_test = self.scaler.transform(self.X_test)
			prediction = classification.predict(self.X_test)
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
		classification = svm.SVC(C=1, gamma=.1, cache_size=500)
		# classification.fit(self.X_train, self.y_train)
		return classification

	def LinearSVCclassifier(self):
		classification = svm.LinearSVC(C=10, penalty='l1', dual=False)
		# classification.fit_transform(self.X_train, self.y_train)
		return classification

	def GaussianNBayes(self):
		classification = GaussianNB()
		self.X_train = self.X_train.toarray()
		# classification.fit(self.X_train.toarray(), self.y_train)
		return classification

	def GradientDescent(self):
		# penalty elasticnet and l2
		classification = SGDClassifier(alpha=.000001, loss='hinge', penalty='l2')
		# self.scaler = Scaler(with_mean=False)
		# self.scaler.fit(self.X_train)
		# self.X_train = self.scaler.transform(self.X_train)
		# classification.fit(self.X_train, self.y_train)
		return classification

	def classify(self, classifier):
		# benchmark can be run just by uncommenting the call and uncommeting the fit function in the function.
		self.classifier = classifier
		print "##########  Training using %s classifier. ##########\n\n\n" % classifier
		if classifier == 'svc':
			self.crossValidation(self.SVCclassifier())
			# self.benchmark(self.SVCclassifier())
		if classifier == 'linear':
			self.crossValidation(self.LinearSVCclassifier())
			# self.benchmark(self.LinearSVCclassifier())
		if classifier == 'gnb':
			self.crossValidation(self.GaussianNBayes())
			# self.benchmark(self.GaussianNBayes())
		if classifier == 'gd':
			self.crossValidation(self.GradientDescent())
			# self.benchmark(self.GradientDescent())
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

class PipeLineClassifier:

	def setVariables(self, classifier):
		if classifier == 'svc':
			self.pipeline = Pipeline([
				('vect', HashingVectorizer(stop_words='english', non_negative=True)),
				('feature_select', SelectKBest(chi2)),
				('clf', svm.SVC(cache_size=500))
			])

			c_range = 10.0 ** np.arange(0, 4)
			gamma_range = 10.0 ** np.arange(-1, 1)
			k_range = 100 * np.arange(1, 31)

			self.parameters = {
				'clf__C': c_range,
				'clf__gamma': gamma_range,
				'feature_select__k': k_range
			}

		if classifier == 'linear':
			self.pipeline = Pipeline([
				('vect', HashingVectorizer(stop_words='english', non_negative=True)),
				('feature_select', SelectKBest(chi2)),
				('clf', svm.LinearSVC(dual=False))
			])

			c_range = 10.0 ** np.arange(0, 2)
			k_range = 100 * np.arange(1, 31)

			self.parameters = {
				'clf__C': c_range,
				'feature_select__k': k_range
			}

		if classifier == 'gd':
			self.pipeline = Pipeline([
				('vect', TfidfVectorizer(stop_words='english', min_df=1, max_df=.8)),
				('feature_select', SelectKBest(chi2)),
				('clf', SGDClassifier(loss='hinge'))
			])

			alpha_range = 10.0 ** np.arange(-6, -3)
			k_range = 100 * np.arange(1, 31)
			self.parameters = {
				'vect__max_df': np.array([.5, .75, 1.0]),
				'clf__penalty': np.array(['l2', 'elasticnet']),
				'clf__alpha': alpha_range,
				'feature_select__k': k_range
				# 'clf__ngram_range': np.array([(1,1), (1,2)]) Not possible too many computations and require large memory.
			}

		if classifier == 'gnb':
			self.pipeline = Pipeline([
				('vect', TfidfVectorizer(stop_words='english', min_df=1)),
				('clf', GaussianNB())
			])
			self.parameters = {
				'vect__max_df': np.array([.5, .75, 1.0]),
				# 'clf__ngram_range': np.array([(1,1), (1,2)]) Not possible too many computations and require large memory.
			}

	def benchmark(self):
		gridSearch = GridSearchCV(self.pipeline, self.parameters, metrics.f1_score, cv=5, verbose=1)
		print "Performing grid search..."
		print "pipeline:", [name for name, _ in self.pipeline.steps]
		print "parameters:"
		pprint(self.parameters)
		gridSearch.fit(self.X_train, self.y_train)

		print "Best score: %0.3f" % gridSearch.best_score_
		print "Best parameters set:"
		best_parameters = gridSearch.best_estimator_.get_params()
		for param_name in sorted(self.parameters.keys()):
			print "\t%s: %r" % (param_name, best_parameters[param_name])


	def __init__(self, classes, X_train, y_train, X_test, y_test, train_size, test_size):
		self.classes = classes
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		self.train_size = train_size
		self.test_size = test_size