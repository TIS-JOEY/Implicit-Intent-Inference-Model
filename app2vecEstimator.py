from sklearn.base import BaseEstimator, ClassifierMixin
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
import collections


class app2vec_estimator(BaseEstimator, ClassifierMixin):
	'''
	Build an app2vec's estimator for gridsearch.
	sg, window, min_count, iter are parameters of App2Vec. (You can view more on https://radimrehurek.com/gensim/models/word2vec.html.)
	training_data：Training data for App2Vec.
	etree：The estimator of EtraTreesClassifier.
	'''
	def __init__(self, training_data, sg = 1,size = 100, window = 3, min_count = 0, iter = 10, n_estimators = 1000):
		self.size = size
		self.sg = sg
		self.window = window
		self.min_count = min_count
		self.model = None
		self.iter = iter
		self.training_data = training_data
		self.n_estimators = n_estimators
		self.estimator = None
		self.w2v = None
		self.etree = None

	def fit(self, X, y):
		# Train an App2Vec model.
		self.model = Word2Vec(self.training_data,sg = self.sg,size = self.size,window = self.window,seed = 0,min_count = self.min_count,iter = self.iter,compute_loss=True)
		
		# Store the mapping of apps and their corrsponding vector.
		self.w2v = dict(zip(self.model.wv.index2word,self.model.wv.syn0))
		

		etree_w2v = Pipeline([
			("word2vec vectorizer", TfidfEmbeddingVectorizer(self.w2v)),
			("extra trees", ExtraTreesClassifier(n_estimators = self.n_estimators))])

		etree_w2v.fit(X, y)
		self.etree = etree_w2v
		return self

	def predict(self,X):
		return self.etree.predict(X)

	def score(self,X,y):
		return self.etree.score(X,y)

class MeanEmbeddingVectorizer:
	'''
	Average app vectors for all apps in an app sequence and treat this as the vector of that app sequence. 
	'''

	def __init__(self,app2vec):
		self.app2vec = app2vec
		self.dim = len(app2vec)

	def fit(self, X, y):
		return self

	def transform(self, X):
		return np.array([np.mean([self.app2vec[app] for app in apps if app in self.app2vec], axis = 0) for apps in X])

class TfidfEmbeddingVectorizer:
	'''
	Compare to MeanEmbeddingVectorizer, TfidfEmbeddingVectorizer adds weight considerations.
	'''

	def __init__(self, app2vec):
		self.app2vec = app2vec
		self.app2weight = None
		self.dim = len(app2vec)

	def fit(self, X, y):
		tfidf = TfidfVectorizer(analyzer=lambda x: x)
		tfidf.fit(X)
		# if an app was never seen - it must be at least as infrequent
		# as any of the known words - so the default idf is the max of 
		# known idf's


		max_idf = max(tfidf.idf_)
		
		self.app2weight = collections.defaultdict(
			lambda: max_idf,
			[(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])


		return self

	def transform(self, X):
		return np.array([
				np.mean([self.app2vec[app] * self.app2weight[app]
						 for app in apps if app in self.app2vec], axis=0)
				for apps in X
			])