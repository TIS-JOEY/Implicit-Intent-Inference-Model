from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import pickle
import collections


class App2Vec:
	def __init__(self,stop_app_path = []):
		'''
		training_data：Store the training data
		stop_app：Store the stop app. These stop apps won't treat as the training data for App2Vec.
		label2id：Store the mapping with cluster labels and app sequences.
		'''
		self.training_data = []
		self.stop_app = []
		self.label2id = collections.defaultdict(list)

		if stop_app_path:
			with open(stop_app_path,'r') as f:
				self.stop_app = f.read().split('\n')

	# half ignore process mode..
	def ignore_all_get_app(self,each_app_seq):
		each_app_list = each_app_seq.split()
		result = []
		for app in each_app_list:
			if app in self.stop_app:
				return []
			else:
				result.append(app)
		return [result]

	# provide the training data for App2Vec.
	def csv2training_data(self,raw_file_path,ignore_all = True):
		'''
		file_path：The storage location of your raw training data.
		ignore_all(Optional)：Ignore mode，True is Full ignore mode，False is half ignore mode.
		'''

		df = pd.read_csv(raw_file_path,header = None)

		for index,each_app_seq in df.iterrows():

			#Full ignore mode
			if ignore_all:
				for each_app_list in (map(self.ignore_all_get_app, each_app_seq.tolist())):
					self.training_data.extend(each_app_list)
			
			#Half ignore mode
			else:
				self.training_data.append([app for ele_app_list in each_app_seq.tolist() for app in each_app_list.split(' ') if app not in stop_app])
		
			
	#Train the app2vec.
	def training_App2Vec(self,app2vec_model_path):
		'''
		app2vec_model_path：The storage location of the app2vec model.
		'''

		#Views more, https://radimrehurek.com/gensim/models/word2vec.html
		model = Word2Vec(self.training_data,sg=1,size = 64,window = 3,seed = 0,min_count = 0,iter = 10,compute_loss=True)

		#save the model
		model.save(app2vec_model_path)

	#Train the ANN
	def ANN(self,dim,num_tree,app2vec_model_path,ann_model_path):
		'''
		dim = the Dimension of App2Vec.
		num_tree：The number of trees of your ANN forest. More tress more accurate.
		ann_model_path：The storage path of ANN model.
		'''

		#View more, https://github.com/spotify/annoy.
		from annoy import AnnoyIndex

		#load app2vec model.
		app2vec_model = Word2Vec.load(app2vec_model_path)

		#get the vector of app2vec.
		vector = app2vec_model.wv.syn0
		
		t = AnnoyIndex(dim)
		
		for i in app2vec_model.wv.vocab:
			#get the mapping id.
			index = app2vec_model.wv.vocab[str(i)].index

			#add the mapping.
			t.add_item(index,vector[index])

		#train the app2vec. num_tree is the number of your ANN forest.
		t.build(num_tree)

		#save the model
		t.save(ann_model_path)

	def affinity_propagation(self,app2vec_model_path,af_model_path,prefer):
		#View more, go to http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html.
		from sklearn.cluster import AffinityPropagation

		#load app2vec model.
		app2vec_model = Word2Vec.load(app2vec_model_path)

		#get the vector of app2vec.
		vector = app2vec_model.wv.syn0

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

		
		
if __name__ == "__main__":
	app2vec = App2Vec()
	app2vec.csv2training_data(raw_file_path = '/Users/apple/Documents/paper/raw_data.csv')
	app2vec.training_App2Vec(app2vec_model_path = '/Users/apple/Documents/app2vec.model')
	app2vec.ANN(dim = 64,num_tree = 10000,app2vec_model_path = '/Users/apple/Documents/app2vec.model',ann_model_path = '/Users/apple/Documents/ANN.model')
	app2vec.affinity_propagation(app2vec_model_path = '/Users/apple/Documents/app2vec.model',af_model_path = '/Users/apple/Documents/NewAFCluster.pkl',prefer = -30)
	app2vec.get_label2id(af_model_path = '/Users/apple/Documents/AFCluster.pkl')
