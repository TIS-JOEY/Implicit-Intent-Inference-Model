import Training.Model as Model
import re
from aip import AipNlp
import os
import numpy as np
from collections import Counter
import json
from gensim.models import Word2Vec
from hanziconv import HanziConv
import itertools
import multiprocessing
import collections




def calculateJob(sen,candidate,CanDes,shared_dict):
	score = calDocScore(sen,CanDes)
	print(score)
	shared_dict[candidate] = score

def calDocScore(text1,text2):
	APP_ID = "APP_ID"
	API_KEY = "API_KEY"
	SECRET_KEY = "SECRET_KEY"
	client = AipNlp(APP_ID,API_KEY,SECRET_KEY)
	return client.simnet(text1,text2)['score']

class IMIP:
	def __init__(self,explicit_intent,intentApp,app2vec_model_path,ann_model_path,af_model_path,app2des):

		# The mapping between apps and intents
		self.intentApp = intentApp

		# Store the explicit intents
		self.explicit_intent = explicit_intent

		# Initial App2Vec class
		self.app2vec = Model.App2Vec()

		self.app2des = app2des

		# Initial ANN class
		self.ann = Model.ANN(app2vec_model_path = app2vec_model_path,ann_model_path = ann_model_path)

		# Initial AF class
		self.af = Model.AF(app2vec_model_path = app2vec_model_path,af_model_path = af_model_path)

		# Initial BILSTM class
		self.bilstm = Model.BILSTM(app2vec_model_path = app2vec_model_path,max_len = 5)

		# Initial processData class
		self.p_data = Model.processData()

		self.app2vec_model = Word2Vec.load('data/Model/app2vec.model')

		# Load App2Vec model
		self.app2vec_model = self.app2vec.load_App2Vec(app2vec_model_path)

	def query(self,input_sen = None,model = 'ANN',ranker = 'doc',lstm = True):

		
		if model == 'ANN':
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

		# transfer to app and get their ids
		apps = [j for i in self.explicit_intent for j in self.intentApp[i]]
		indexs = [self.app2vec_model.wv.vocab[app].index for app in apps if app in self.app2vec_model]

		# Load ANN model
		ann_model = self.ann.load_ANN()

		# Get their neighbor and flat it to 1D.
		nbrs = list(itertools.chain.from_iterable([ann_model.get_nns_by_item(index,10) for index in indexs]))
		
		# Transfer to app and avoid duplicate
		nbrs = [self.app2vec_model.wv.index2word[nbr] for nbr in nbrs if self.app2vec_model.wv.index2word[nbr] not in apps]

		counter = collections.Counter(nbrs)

		major_voting_filter = [app_with_count[0] for app_with_count in counter.most_common()]

		result = major_voting_filter[:5]

		return result

	def ANN_with_mf_process(self,mf_filepath = 'data/Model/wmf_matrix.txt'):

		mf_matrix = self.p_data.load_resource(mf_filepath)

		# transfer to app and get their ids
		apps = [j for i in self.explicit_intent for j in self.intentApp[i]]
		indexs = [self.app2vec_model.wv.vocab[app].index for app in apps if app in self.app2vec_model]

		# Load ANN model
		ann_model = self.ann.load_ANN()

		# Get their neighbor and flat it to 1D.
		nbrs = list(itertools.chain.from_iterable([ann_model.get_nns_by_item(index,10) for index in indexs]))

		scoring = {}
		for app in indexs:
			for nbr in nbrs:
				if nbr in scoring:
					scoring[nbr] = max(scoring[nbr],mf_matrix[app-1][nbr])
				else:
					scoring[nbr] = mf_matrix[app-1][nbr]

		# Sort by frequency
		mf_filter = list(map(lambda y:self.app2vec_model.wv.index2word[y],sorted(scoring,key = scoring.get,reverse = True)))

		mf_filter = list(filter(lambda x:x not in apps,mf_filter))

		result = mf_filter[:5]

		return result

	def ANN_with_doc_process(self,input_sen):

		# transfer to app and get their ids
		apps = [j for i in self.explicit_intent for j in self.intentApp[i]]
		indexs = [self.app2vec_model.wv.vocab[app].index for app in apps if app in self.app2vec_model]

		# Load ANN model
		ann_model = self.ann.load_ANN()

		# Get their neighbor and flat it to 1D.
		nbrs = list(itertools.chain.from_iterable([ann_model.get_nns_by_item(index,5) for index in indexs]))
		
		# Transfer to app and avoid duplicate
		nbr_app = [self.app2vec_model.wv.index2word[nbr] for nbr in nbrs if self.app2vec_model.wv.index2word[nbr] not in apps]

		pool = multiprocessing.Pool()
		manager = multiprocessing.Manager()

		# For recording the semantic score
		shared_dict = manager.dict()

		for nbr_id in range(len(nbr_app)):

			# Calculate the semantic score
			pool.apply_async(calculateJob,args=(input_sen, nbr_app[nbr_id],self.app2des[nbr_app[nbr_id]],shared_dict))
			#calculateJob(input_sen, nbr_app[nbr_id],self.app2des[nbr_app[nbr_id]],shared_dict)


		pool.close()
		pool.join()
		
		shared_dict = dict(shared_dict)
		# Sort by semantic score
		semantic_filter = sorted(shared_dict,key = shared_dict.get,reverse = True)

		result = semantic_filter[:5]

		return result

	def AF_with_mv_process(self):

		# Load AF model
		af_model = self.af.get_af_model()

		# transfer to app
		apps = [j for i in self.explicit_intent for j in self.intentApp[i]]

		# Get the input vector
		vector = np.mean([self.app2vec_model[app] for app in apps if app in self.app2vec_model],0)

		# The predicted label
		predict_label = af_model.predict([vector])

		# Major voting 
		counter = collections.Counter(self.af.label2app[predict_label[0]])

		# Choose the top k apps with higher voting and avoid duplicate.
		major_voting_filter = [app_with_count[0] for app_with_count in counter.most_common() if app_with_count[0] not in apps]

		result = major_voting_filter[:5]

		return result

	def AF_with_mf_process(self,mf_filepath = 'data/Model/wmf_matrix.txt'):
		
		mf_matrix = self.p_data.load_resource(mf_filepath)

		# Load AF model
		af_model = self.af.get_af_model()

		apps = [j for i in self.explicit_intent for j in self.intentApp[i]]

		indexs = [self.app2vec_model.wv.vocab[app].index for app in apps if app in self.app2vec_model]

		# Get the input vector
		vector = np.mean([self.app2vec_model[app] for app in apps if app in self.app2vec_model],0)

		# The predicted label
		predict_label = af_model.predict([vector])

		predict_app = [self.app2vec_model.wv.vocab[app].index for app in self.af.label2app[predict_label[0]]]

		scoring = {}
		for app in indexs:
			for nbr in predict_app:
				if nbr in scoring:
					scoring[nbr] = max(scoring[nbr],mf_matrix[app-1][nbr])
				else:
					scoring[nbr] = mf_matrix[app-1][nbr]

		# Sort by frequency
		mf_filter = list(map(lambda y:self.app2vec_model.wv.index2word[y],sorted(scoring,key = scoring.get,reverse = True)))

		mf_filter = list(filter(lambda x:x not in apps,mf_filter))

		result = mf_filter[:5]

		return result

	def AF_with_doc_process(self,input_sen):

		# Load AF model
		af_model = self.af.get_af_model()

		apps = [j for i in self.explicit_intent for j in self.intentApp[i]]

		# Get the input vector
		vector = np.mean([self.app2vec_model[app] for app in apps if app in self.app2vec_model],0)

		# The predicted label
		predict_label = af_model.predict([vector])


		# Get the candididate apps and avoid duplicate
		candidiates = list(filter(lambda x:x in apps,list(set(self.af.label2app[predict_label[0]]))))


		pool = multiprocessing.Pool()
		manager = multiprocessing.Manager()

		# For recording the semantic score
		shared_dict = manager.dict()

		for candidiate_id in range(len(candidiates)):

			# Calculate the semantic score
			pool.apply_async(calculateJob,args=(input_sen,candidiates[candidiate_id],self.app2des[candidiates[candidiate_id]],shared_dict))

		pool.close()
		pool.join()

		shared_dict = dict(shared_dict)
		
		# Sort by semantic score
		semantic_filter = sorted(shared_dict,key = shared_dict.get,reverse = True)

		result = semantic_filter[:5]

		return result

	def BILSTM_ANN_with_mv_process(self,bilstm_filepath = 'data/Model/BILSTM_model.h5'):

		# Load BILSTM model
		bilstm_model = self.bilstm.load_BI_LSTM_model(bilstm_filepath)

		# Load ANN model
		ann_model = self.ann.load_ANN()

		# transfer to index
		apps = [self.app2vec_model.wv.vocab[j].index+1 for i in self.explicit_intent for j in self.intentApp[i] if j in self.app2vec_model]

		# predicted vector
		vector_predict = self.bilstm.BI_LSTM_predict([apps],bilstm_model)

		# Get their neighbors.
		nbrs = ann_model.get_nns_by_vector(vector_predict,10)

		# Transfer them to apps and avoid duplicate
		nbrs = [self.app2vec_model.wv.index2word[nbr] for nbr in nbrs if self.app2vec_model.wv.index2word[nbr] not in apps]

		counter = collections.Counter(nbrs)

		major_voting_filter = [app_with_count[0] for app_with_count in counter.most_common()]

		result = major_voting_filter[:5]

		return result

	def BILSTM_ANN_with_mf_process(self,mf_filepath = 'data/Model/wmf_matrix.txt',bilstm_filepath = 'data/Model/BILSTM_model.h5'):

		mf_matrix = self.p_data.load_resource(mf_filepath)

		# Load BILSTM model
		bilstm_model = self.bilstm.load_BI_LSTM_model(bilstm_filepath)

		# Load ANN model
		ann_model = self.ann.load_ANN()

		# transfer to index
		apps = [self.app2vec_model.wv.vocab[j].index+1 for i in self.explicit_intent for j in self.intentApp[i] if j in self.app2vec_model]

		# predicted vector
		vector_predict = self.bilstm.BI_LSTM_predict([apps],bilstm_model)

		# Get their neighbors.
		nbrs = ann_model.get_nns_by_vector(vector_predict,10)

		scoring = {}
		for app in apps:
			for nbr in nbrs:
				if nbr in scoring:
					scoring[nbr] = max(scoring[nbr],mf_matrix[app-1][nbr])
				else:
					scoring[nbr] = mf_matrix[app-1][nbr]

		# Sort by frequency
		mf_filter = list(map(lambda y:self.app2vec_model.wv.index2word[y],sorted(scoring,key = scoring.get,reverse = True)))

		mf_filter = list(filter(lambda x:x not in apps,mf_filter))

		result = mf_filter[:5]

		return result
	
	def BILSTM_ANN_with_doc_process(self,input_sen, bilstm_filepath = 'data/Model/BILSTM_model.h5'):
		
		# Load BILSTM model
		bilstm_model = self.bilstm.load_BI_LSTM_model(bilstm_filepath)

		# Load ANN model
		ann_model = self.ann.load_ANN()

		# transfer to app
		apps = [self.app2vec_model.wv.vocab[j].index+1 for i in self.explicit_intent for j in self.intentApp[i] if j in self.app2vec_model]

		# predicted vector
		vector_predict = self.bilstm.BI_LSTM_predict([apps],bilstm_model)

		# Get their neighbor and flat it to 1D.
		nbrs = ann_model.get_nns_by_vector(vector_predict,10)

		# Transfer to app
		nbr_app = [self.app2vec_model.wv.index2word[nbr] for nbr in nbrs if self.app2vec_model.wv.index2word[nbr] not in apps]

		pool = multiprocessing.Pool()
		manager = multiprocessing.Manager()

		# For recording the semantic score
		shared_dict = manager.dict()

		for nbr_id in range(len(nbr_app)):
			#calculateJob(input_sen, nbr_app[nbr_id],self.app2des[nbr_app[nbr_id]],shared_dict)
			# Calculate the semantic score
			pool.apply_async(calculateJob,args=(input_sen, nbr_app[nbr_id],self.app2des[nbr_app[nbr_id]],shared_dict))

		pool.close()
		pool.join()
		
		shared_dict = dict(shared_dict)	
		# Sort by semantic score
		semantic_filter = sorted(shared_dict,key = shared_dict.get,reverse = True)

		result = semantic_filter[:5]
	
	def BILSTM_AF_with_mv_process(self,bilstm_filepath = 'data/Model/BILSTM_model.h5'):

		# Load BILSTM model
		bilstm_model = self.bilstm.load_BI_LSTM_model(bilstm_filepath)

		# Load AF model
		af_model = self.af.get_af_model()

		# transfer to app
		apps = [self.app2vec_model.wv.vocab[j].index+1 for i in self.explicit_intent for j in self.intentApp[i] if j in self.app2vec_model]

		# predicted vector
		vector_predict = self.bilstm.BI_LSTM_predict([apps],bilstm_model)

		# The predicted label
		predict_label = af_model.predict([vector_predict])

		# Major voting 
		counter = collections.Counter(self.af.label2app[predict_label[0]])

		# Choose the top k apps with higher voting and avoid duplicate.
		major_voting_filter = [app_with_count[0] for app_with_count in counter.most_common() if app_with_count[0] not in apps]

		result = major_voting_filter[:5]

		return result

	def BILSTM_AF_with_mf_process(self,mf_filepath = 'data/Model/wmf_matrix.txt',bilstm_filepath = 'data/Model/BILSTM_model.h5'):

		mf_matrix = self.p_data.load_resource(mf_filepath)

		# Load BILSTM model
		bilstm_model = self.bilstm.load_BI_LSTM_model(bilstm_filepath)

		# Load AF model
		af_model = self.af.get_af_model()

		# transfer to app
		apps = [j for i in self.explicit_intent for j in self.intentApp[i]]

		indexs = [self.app2vec_model.wv.vocab[app].index+1 for app in apps if app in self.app2vec_model]

		# predicted vector
		vector_predict = self.bilstm.BI_LSTM_predict([indexs],bilstm_model)

		# The predicted label
		predict_label = af_model.predict([vector_predict])

		predict_app = [self.app2vec_model.wv.vocab[app].index for app in self.af.label2app[predict_label[0]]]

		scoring = {}
		for app in indexs:
			for nbr in predict_app:
				if nbr in scoring:
					scoring[nbr] = max(scoring[nbr],mf_matrix[app-1][nbr])
				else:
					scoring[nbr] = mf_matrix[app-1][nbr]

		# Sort by frequency
		mf_filter = list(map(lambda y:self.app2vec_model.wv.index2word[y],sorted(scoring,key = scoring.get,reverse = True)))

		mf_filter = list(filter(lambda x:x not in apps,mf_filter))

		result = mf_filter[:5]

		return result

	def BILSTM_AF_with_doc_process(self,input_sen,bilstm_filepath = r'data/Model/BILSTM_model.h5'):

		# Load BILSTM model
		bilstm_model = self.bilstm.load_BI_LSTM_model(bilstm_filepath)

		# Load AF model
		af_model = self.af.get_af_model()

		# transfer to app
		apps = [j for i in self.explicit_intent for j in self.intentApp[i] if j in self.app2vec_model]

		indexs = [self.app2vec_model.wv.vocab[app].index+1 for app in apps if app in self.app2vec_model]

		# predicted vector
		vector_predict = self.bilstm.BI_LSTM_predict([indexs],bilstm_model)

		# The predicted label
		predict_label = af_model.predict([vector_predict])

		# Get the candididate apps and avoid duplicate
		candidiates = list(filter(lambda x:x in apps,list(set(self.af.label2app[predict_label[0]]))))
		
		pool = multiprocessing.Pool()
		manager = multiprocessing.Manager()

		# For recording the semantic score
		shared_dict = manager.dict()

		for candidiate_id in range(len(candidiates)):

			# Calculate the semantic score
			pool.apply_async(calculateJob,args=(input_sen,candidiates[candidiate_id],self.app2des[candidiates[candidiate_id]],shared_dict))

		pool.close()
		pool.join()

		shared_dict = dict(shared_dict)
		
		# Sort by semantic score
		semantic_filter = sorted(shared_dict,key = shared_dict.get,reverse = True)

		result = semantic_filter[:5]

		return result

