import Training.Model
import re
from aip import AipNlp
import os
import numpy as np
from multiprocessing import Process, Manager,Lock
from collections import Counter


def calculateJob(sen,candidate,CanDes,shared_dict):
	score = calDocScore(sen,CanDes)
	shared_dict[candidate] = score

def calDocScore(text1,text2):
	APP_ID = "APP_ID"
	API_KEY = "APP_KEY"
	SECRET_KEY = "SECRET_KEY "
	client = AipNlp(APP_ID,API_KEY,SECRET_KEY)
	return client.simnet(text1,text2)['score']

class IMIP:
	def __init__(self,explicit_intent,intentApp,app2vec_model_path,ann_model_path,af_model_path):

		# The mapping between apps and intents
		self.intentApp = intentApp

		# Store the explicit intents
		self.explicit_intent = explicit_intent

		# Initial App2Vec class
		self.app2vec = Model.App2Vec()

		# Initial ANN class
		self.ann = Model.ANN(app2vec_model_path = app2vec_model_path,ann_model_path = ann_model_path)

		# Initial AF class
		self.af = Model.AF(app2vec_model_path = app2vec_model_path,af_model_path = af_model_path)

		# Initial BILSTM class
		self.bilstm = Model.BILSTM(app2vec_model_path = app2vec_model_path,max_len = 5)

		# Initial processData class
		self.p_data = Model.processData()

		# Set up description
		self.p_data.processDescription()

		# Load App2Vec model
		self.app2vec_model = app2vec.load_App2Vec(app2vec_model_path)

	def query(self,input_sen = None,model = 'ANN',doc = True,lstm = True):
		if model == 'ANN':
			if lstm:
				if doc:
					result = self.BILSTM_ANN_with_doc_process(input_sen)
				else:
					result = self.BILSTM_ANN_without_doc_process()
			else:
				if doc:
					result = self.ANN_with_doc_process(input_sen)
				else:
					result = self.ANN_without_doc_process()

		else:
			if lstm:
				if doc:
					result = self.BILSTM_AF_with_doc_process(input_sen)
				else:
					result = self.BILSTM_AF_without_doc_process()
			else:
				if doc:
					result = self.AF_with_doc_process(input_sen)
				else:
					result = self.AF_without_doc_process()

		if lstm:
				if ranker == 'mv':
					result = self.BILSTM_ANN_with_mv_process()
				elif ranker == 'mf':
					result = self.BILSTM_ANN_with_mf_process()
				elif ranker == 'doc':
					result = self.BILSTM_ANN_with_doc_process(input_sen)
			else:
				if ranker == 'mv':
					result = self.ANN_with_mv_process()
				elif ranker == 'mf':
					result = self.ANN_with_mf_process()
				elif ranker == 'doc':
					result = self.ANN_with_doc_process(input_sen)

		else:
			if lstm:
				if ranker == 'mv':
					result = self.BILSTM_AF_with_mv_process()
				elif ranker == 'mf':
					result = self.BILSTM_AF_with_mf_process()
				elif ranker == 'doc':
					result = self.BILSTM_AF_with_doc_process(input_sen)
			else:
				if ranker == 'mv':
					result = self.AF_with_mv_process()
				elif ranker == 'mf':
					result = self.AF_with_mf_process()
				elif ranker == 'doc':
					result = self.AF_with_doc_process(input_sen)

		return result


	def ANN_with_mv_process(self):

		# transfer to app
		apps = [intentApp[i] for i in self.explicit_intent]

		# Load ANN model
		ann_model = self.ann.load_ANN()

		# Get ids
		indexs = [self.app2vec_model.wv.vocab[app].index for app in apps]

		# Get their neighbor and flat it to 1D.
		nbrs = list(itertools.chain.from_iterable([self.ann_model.get_nns_by_item(index,10) for index in indexs]))
		
		# Transfer to app and avoid duplicate
		nbrs = [self.app2vec_model.wv.index2word[nbr] for nbr in nbrs if self.app2vec_model.wv.index2word[nbr] not in apps]

		counter = collections.Counter(nbrs)

		major_voting_filter = [app_with_count[0] for app_with_count in counter.most_common()]

		# Transfer to class
		result = self.p_data.checkClass(major_voting_filter,5)

		return result

	def ANN_with_mf_process(self,mf_filepath = 'data/model/mf_matrix.txt'):

		mf_matrix = self.load_resources(mf_filepath)

		# transfer to app
		apps = [intentApp[i] for i in self.explicit_intent]

		# Load ANN model
		ann_model = self.ann.load_ANN()

		# Get ids
		indexs = [self.app2vec_model.wv.vocab[app].index for app in apps]

		# Get their neighbor and flat it to 1D.
		nbrs = list(itertools.chain.from_iterable([self.ann_model.get_nns_by_item(index,10) for index in indexs]))

		scoring = {}
		for app in indexs:
			for nbr in nbrs:
				if nbr in scoring:
					scoring[nbr] = max(scoring[nbr],mf_matrix[app-1][nbr])
				else:
					scoring[nbr] = mf_matrix[app-1][nbr]

		# Sort by frequency
		mf_filter = list(map(lambda y:self.app2vec_model.wv.index2word[y],sorted(scoring,key = scoring.get,reverse = True)))

		mf_filter = list(filter(lambda x:x in apps,mf_filter))

		# Compare with true labels
		result = self.p_data.checkClass(mf_filter,5)

		return result

	def ANN_with_doc_process(self,input_sen):

		# transfer to app
		apps = [intentApp[i] for i in self.explicit_intent]

		# Load ANN model
		ann_model = self.ann.load_ANN()

		# Get ids
		indexs = [self.app2vec_model.wv.vocab[app].index for app in apps]

		# Get their neighbor and flat it to 1D.
		nbrs = list(itertools.chain.from_iterable([self.ann_model.get_nns_by_item(index,5) for index in indexs]))
		
		# Transfer to app and avoid duplicate
		nbrs = [self.app2vec_model.wv.index2word[nbr] for nbr in nbrs if self.app2vec_model.wv.index2word[nbr] not in apps]

		pool = multiprocessing.Pool()
		manager = Manager()

		# For recording the semantic score
		shared_dict = manager.dict()

		for nbr_id in range(len(nbr_app)):

			# Calculate the semantic score
			pool.apply_async(calculateJob,args=(input_sen, nbr_app[nbr_id],self.p_data.app2des[nbr_app[nbr_id]],shared_dict))

		pool.close()
		pool.join()
				
		# Sort by semantic score
		semantic_filter = sorted(shared_dict,key = shared_dict.get,reverse = True)

		result = self.p_data.checkClass(semantic_filter,5)

		return result

	def AF_with_mv_process(self):

		# Load AF model
		af_model = self.af.get_af_model()

		# transfer to app
		apps = [intentApp[i] for i in self.explicit_intent]

		# Get the input vector
		vector = np.mean([self.app2vec_model[app] for app in apps],0)

		# The predicted label
		predict_label = af_model.predict([vector])

		# Major voting 
		counter = collections.Counter(self.af.label2app[predict_label[0]])

		# Choose the top k apps with higher voting and avoid duplicate.
		major_voting_filter = [app_with_count[0] for app_with_count in counter.most_common() if app_with_count[0] not in apps]

		result = self.p_data.checkClass(major_voting_filter,5)

	def AF_with_mf_process(self,mf_filepath = 'data/model/mf_matrix.txt'):
		
		mf_matrix = self.load_resources(mf_filepath)

		# Load AF model
		af_model = self.af.get_af_model()

		# transfer to app
		apps = [intentApp[i] for i in self.explicit_intent]

		# Get ids
		indexs = [self.app2vec_model.wv.vocab[app].index for app in apps]

		# Get the input vector
		vector = np.mean([self.app2vec_model[app] for app in apps],0)

		# The predicted label
		predict_label = af_model.predict([vector])

		predict_app = [self.app2vec_model.wv[app].index for app in self.af.label2app[predict_label[0]]]

		scoring = {}
		for app in indexs:
			for nbr in predict_app:
				if nbr in scoring:
					scoring[nbr] = max(scoring[nbr],mf_matrix[app-1][nbr])
				else:
					scoring[nbr] = mf_matrix[app-1][nbr]

		# Sort by frequency
		mf_filter = list(map(lambda y:self.app2vec_model.wv.index2word[y[0]],sorted(scoring,key = scoring.get,reverse = True)))

		mf_filter = list(filter(lambda x:x in apps,mf_filter))

		# Compare with true labels
		result = self.p_data.checkClass(mf_filter,5)

	def AF_with_doc_process(self,input_sen):

		# Load AF model
		af_model = self.af.get_af_model()

		# transfer to app
		apps = [intentApp[i] for i in self.explicit_intent]

		# Get the input vector
		vector = np.mean([self.app2vec_model[app] for app in apps],0)

		# The predicted label
		predict_label = af_model.predict([vector])

		# Get the candididate apps and avoid duplicate
		candidiates = list(filter(lambda x:x not in apps,list(set(self.af.label2app[predict_label[0]]))))

		pool = multiprocessing.Pool()
		manager = Manager()

		# For recording the semantic score
		shared_dict = manager.dict()

		for candidiate_id in range(len(candidiates)):

			# Calculate the semantic score
			pool.apply_async(calculateJob,args=(input_sen,candidiates[candidiate_id],p_data.app2des[candidiates[candidiate_id]]))

		pool.close()
		pool.join()

		# Sort by semantic score
		semantic_filter = sorted(shared_dict,key = shared_dict.get,reverse = True)

		result = self.p_data.checkClass(semantic_filter,5)

		return result

	def BILSTM_ANN_with_mv_process(self,bilstm_filepath = 'data/model/BILSTM_model.h5'):

		# Load BILSTM model
		bilstm_model = self.bilstm.load_BI_LSTM_model(bilstm_filepath)

		# Load ANN model
		ann_model = self.ann.load_ANN()

		# transfer to index
		apps = [self.app2vec_model.wv.vocab[intentApp[i]].index+1 for i in self.explicit_intent]

		# predicted vector
		vector_predict = self.bilstm.predict(apps,bilstm_model)

		# Get their neighbors.
		nbrs = ann_model.get_nns_by_vector(vector_predict,10)

		# Transfer them to apps and avoid duplicate
		nbrs = [self.app2vec_model.wv.index2word[nbr] for nbr in nbrs if self.app2vec_model.wv.index2word[nbr] not in apps]

		counter = collections.Counter(nbrs)

		major_voting_filter = [app_with_count[0] for app_with_count in counter.most_common()]

		result = self.p_data.checkClass(major_voting_filter,5)

	def BILSTM_ANN_with_mf_process(self,mf_filepath = 'data/model/mf_matrix.txt',bilstm_filepath = 'data/model/BILSTM_model.h5'):

		mf_matrix = self.load_resources(mf_filepath)

		# Load BILSTM model
		bilstm_model = self.bilstm.load_BI_LSTM_model(bilstm_filepath)

		# Load ANN model
		ann_model = self.ann.load_ANN()

		# transfer to index
		apps = [self.app2vec_model.wv.vocab[intentApp[i]].index+1 for i in self.explicit_intent]

		# predicted vector
		vector_predict = self.bilstm.predict(apps,bilstm_model)

		# Get their neighbors.
		nbrs = ann_model.get_nns_by_vector(vector_predict,10)

		scoring = {}
		for app in indexs:
			for nbr in nbrs:
				if nbr in scoring:
					scoring[nbr] = max(scoring[nbr],mf_matrix[app-1][nbr])
				else:
					scoring[nbr] = mf_matrix[app-1][nbr]

		# Sort by frequency
		mf_filter = list(map(lambda y:self.app2vec_model.wv.index2word[y],sorted(scoring,key = scoring.get,reverse = True)))

		mf_filter = list(filter(lambda x:x in apps,mf_filter))

		# Compare with true labels
		result = self.p_data.checkClass(mf_filter,5)

		return result

	def BILSTM_ANN_with_doc_process(self,input_sen, bilstm_filepath = 'data/model/BILSTM_model.h5'):

		# Load BILSTM model
		bilstm_model = self.bilstm.load_BI_LSTM_model(bilstm_filepath)

		# Load ANN model
		ann_model = self.ann.load_ANN()

		# transfer to app
		apps = [self.app2vec_model.wv.vocab[intentApp[i]].index+1 for i in self.explicit_intent]

		# predicted vector
		vector_predict = self.bilstm.predict(apps,bilstm_model)

		# Get their neighbor and flat it to 1D.
		nbrs = ann_model.get_nns_by_vector(vector_predict,len(y_test[app_seq_id]))

		# Transfer to app
		nbr_app = [self.app2vec_model.wv.index2word[nbr] for nbr in nbrs if self.app2vec_model.wv.index2word[nbr] not in apps]

		pool = multiprocessing.Pool()
		manager = Manager()

		# For recording the semantic score
		shared_dict = manager.dict()

		for nbr_id in range(len(nbr_app)):

			# Calculate the semantic score
			pool.apply_async(calculateJob,args=(input_sen, nbr_app[nbr_id],self.p_data.app2des[nbr_app[nbr_id]],shared_dict))

		pool.close()
		pool.join()
				
		# Sort by semantic score
		semantic_filter = sorted(shared_dict,key = shared_dict.get,reverse = True)

		result = self.p_data.checkClass(semantic_filter,5)

	def BILSTM_AF_with_mv_process(self,bilstm_filepath = 'data/model/BILSTM_model.h5'):

		# Load BILSTM model
		bilstm_model = self.bilstm.load_BI_LSTM_model(bilstm_filepath)

		# Load AF model
		af_model = self.af.get_af_model()

		# transfer to app
		apps = [self.app2vec_model.wv.vocab[intentApp[i]].index+1 for i in self.explicit_intent]

		# predicted vector
		vector_predict = self.bilstm.predict(apps,bilstm_model)

		# The predicted label
		predict_label = af_model.predict([vector_predict])

		# Major voting 
		counter = collections.Counter(self.af.label2app[predict_label[0]])

		# Choose the top k apps with higher voting and avoid duplicate.
		major_voting_filter = [app_with_count[0] for app_with_count in counter.most_common() if app_with_count[0] not in apps]

		result = self.p_data.checkClass(major_voting_filter,5)

		return result

	def BILSTM_AF_with_mv_process(self,mf_filepath = 'data/model/mf_matrix.txt',bilstm_filepath = 'data/model/BILSTM_model.h5'):

		mf_matrix = self.load_resources(mf_filepath)

		# Load BILSTM model
		bilstm_model = self.bilstm.load_BI_LSTM_model(bilstm_filepath)

		# Load AF model
		af_model = self.af.get_af_model()

		# transfer to app
		apps = [self.app2vec_model.wv.vocab[intentApp[i]].index+1 for i in self.explicit_intent]

		# predicted vector
		vector_predict = self.bilstm.predict(apps,bilstm_model)

		# The predicted label
		predict_label = af_model.predict([vector_predict])

		predict_app = [self.app2vec_model.wv[app].index for app in self.af.label2app[predict_label[0]]]

		scoring = {}
		for app in indexs:
			for nbr in predict_app:
				if nbr in scoring:
					scoring[nbr] = max(scoring[nbr],mf_matrix[app-1][nbr])
				else:
					scoring[nbr] = mf_matrix[app-1][nbr]

		# Sort by frequency
		mf_filter = list(map(lambda y:self.app2vec_model.wv.index2word[y],sorted(scoring,key = scoring.get,reverse = True)))

		mf_filter = list(filter(lambda x:x in apps,mf_filter))

		# Compare with true labels
		result = self.p_data.checkClass(mf_filter,5)

		return result

	def BILSTM_AF_with_doc_process(self,input_sen,bilstm_filepath = 'data/model/BILSTM_model.h5'):

		# Load BILSTM model
		bilstm_model = self.bilstm.load_BI_LSTM_model(bilstm_filepath)

		# Load AF model
		af_model = self.af.get_af_model()

		# transfer to app
		apps = [self.app2vec_model.wv.vocab[intentApp[i]].index+1 for i in self.explicit_intent]

		# predicted vector
		vector_predict = self.bilstm.predict(apps,bilstm_model)

		# The predicted label
		predict_label = af_model.predict([vector_predict])

		# Get the candididate apps and avoid duplicate
		candidiates = list(filter(lambda x:x not in apps,list(set(self.af.label2app[predict_label[0]]))))

		pool = multiprocessing.Pool()
		manager = Manager()

		# For recording the semantic score
		shared_dict = manager.dict()

		for candidiate_id in range(len(candidiates)):

			# Calculate the semantic score
			pool.apply_async(calculateJob,args=(input_sen,candidiates[candidiate_id],p_data.app2des[candidiates[candidiate_id]]))

		pool.close()
		pool.join()

		# Sort by semantic score
		semantic_filter = sorted(shared_dict,key = shared_dict.get,reverse = True)

		result = self.p_data.checkClass(semantic_filter,5)

		return result




