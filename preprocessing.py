from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split

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

	def createCorpus(self, data):
		# Removing the stop words and minimum document frequency is 1.
		corpus = []
		corpus_severity = []
		for bug in data:
			corpus.append(bug['description'])
			corpus_severity.append(self.severity[bug['severity']])
		return corpus, corpus_severity

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

	def __init__(self, data_set):
		self.data_set = data_set
		# print self.data_set[0]['severity']
		self.test_train_split()