from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from numpy import array

class  Preprocess:
	"""Contains all the functions for preprocessing the data including tokenization"""
	severity = {
		'blocker': 0,
		'critical': 1,
		'major': 2,
		'normal': 3,
		'minor': 4,
		'trivial': 5,
		'enhancement': 6
	}

	severity_inv = {
		'0': 'blocker',
		'1': 'critical',
		'2': 'major',
		'3': 'normal',
		'4': 'minor',
		'5': 'trivial',
		'6': 'enhancement'
	}

	# Try using binary classification.

	def createCorpus(self, data):
		# Removing the stop words and minimum document frequency is 1.
		corpus = []
		corpus_severity = []
		for bug in data:
			corpus.append(bug['description'])
			corpus_severity.append(self.severity[bug['severity']])
		return corpus, array(corpus_severity)

	def vectorize(self, vector):
		# print self.corpus
		print "##########  Creating %s vector. ##########\n\n\n" % vector
		if vector == 'hashing':
			self.hashingVector()
		if vector == 'count':
			self.countVector()
		if vector == 'tfidf':
			self.tfIdfVector()

	def test_train_split(self):
		print "##########  Creating training and testing data set. ##########\n\n\n"
		self.train_data, self.test_data = train_test_split(self.data_set, test_size=.2, random_state=42)
		print "##########  Creating corpus out of data. ##########\n\n\n"
		self.train_corpus, self.y_train = self.createCorpus(self.train_data)
		self.test_corpus, self.y_test = self.createCorpus(self.test_data)

	def countVector(self):
		vectorizer = CountVectorizer(min_df=1, stop_words='english')
		self.X_train = vectorizer.fit_transform(self.train_corpus)
		self.X_test = vectorizer.transform(self.test_corpus)
		self.train_size = self.X_train.shape[0]
		self.test_size = self.X_test.shape[0]

	def hashingVector(self):
		vectorizer = HashingVectorizer(stop_words='english', non_negative=True) #n_features: default, 1048576
		self.X_train = vectorizer.fit_transform(self.train_corpus)
		self.X_test = vectorizer.transform(self.test_corpus)
		self.train_size = self.X_train.shape[0]
		self.test_size = self.X_test.shape[0]

	def tfIdfVector(self):
		vectorizer = TfidfVectorizer(min_df=1, max_df=.8, stop_words='english') #n_features: default, 1048576
		self.X_train = vectorizer.fit_transform(self.train_corpus)
		self.X_test = vectorizer.transform(self.test_corpus)
		self.train_size = self.X_train.shape[0]
		self.test_size = self.X_test.shape[0]

	def tfidfVector(self):
		vectorizer = TfidfVectorizer(min_df=1, max_df=.8, stop_words='english') #n_features: default, 1048576
		# self.vectorizer = vectorizer
		self.X_train = vectorizer.fit_transform(self.train_corpus)
		self.train_size = self.X_train.shape[0]

	def hashVector(self):
		vectorizer = HashingVectorizer(stop_words='english', non_negative=True)
		self.X_train = vectorizer.fit_transform(self.train_corpus)
		self.train_size = self.X_train.shape[0]

	def chisquare(self):
		print "######## Running chi squared for selecting best features. #########"
		ch2 = SelectKBest(chi2, k=100)
		# features = self.vectorizer.get_feature_names()
		self.X_train = ch2.fit_transform(self.X_train, self.y_train)
		# f = open("words.txt", "w+")
		# for feature in ch2.get_support(indices=True):
			# f.write(features[feature]+"\n")
		self.train_size = self.X_train.shape[0]
		# self.X_test = ch2.transform(self.X_test)
		# self.test_size = self.X_test.shape[0]

	def __init__(self, data_set, new):
		self.data_set = data_set
		# print self.data_set[0]['severity']
		if new == 'True':
			self.train_corpus, self.y_train = self.createCorpus(self.data_set)
			# self.X_train = None
			self.X_test = None
			self.y_test = None
			# self.y_train = None
			self.test_size = 0
		else:
			self.test_train_split()