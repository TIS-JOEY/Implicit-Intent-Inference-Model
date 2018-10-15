import app2vec
from sklearn.externals import joblib
import collections
import numpy as np

#View more, go to http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html.
from sklearn.cluster import AffinityPropagation


class AF:
	def __init__(self,app2vec_model_path,training_data):
		App2Vec = app2vec()
		self.training_data = training_data
		self.app2vec_model = App2Vec.load_App2Vec(app2vec_model_path)
		self.label2id = collections.defaultdict(list)

	def affinity_propagation(self,af_model_path,prefer):

		#get the vector of app2vec.
		vector = self.app2vec_model.wv.syn0

		#store the training data of AF.
		af_training_data = []

		#average the vector of each app sequence as a unit
		for app_seq in self.training_data:
			af_training_data.append(np.mean([app2vec_model[app] for app in app_seq],0))

		# train the af model.
		af_model = AffinityPropagation(preference = prefer).fit(af_training_data)

		# save the model
		joblib.dump(af_model, af_model_path)


	def get_label2id(self,af_model_path):
		# load af model
		af = joblib.load(af_model_path)

		# build a label2id dictionary
		for index,label in enumerate(af.labels_):
			self.label2id[label].append(index)