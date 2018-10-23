# Scientific computing
import pandas as pd
import numpy as np
from scipy import spatial

# Model
from gensim.models import Word2Vec
import gensim
from annoy import AnnoyIndex
from sklearn.cluster import AffinityPropagation

# Saver
from sklearn.externals import joblib
import pickle
import json

# Plot
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Evaluating
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import app2vecEstimator
from sklearn.model_selection import GridSearchCV, KFold

# Others
import collections
import os
import itertools
import random

# Preprocessing
from nltk.tokenize import word_tokenize

# Keras
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.models import Sequential,load_model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model

class processData:
	def __init__(self, goal_app_path = 'data/goal_app.txt', mapping_path = 'data/training_data/apps.csv', ignore_all = True,app2vec_model_path = None):
		'''
		mapping：Store the mapping of id and app_name
		training_data：Store the training data
		stop_app：Store the stop app. These stop apps won't treat as the training data for App2Vec.
		ignore_all：True is full cut mode, False is select cut mode.
		'''

		self.app2id = {}
		self.goal_app = []
		self.training_data = []
		self.ignore_all = ignore_all
		self.app2des = {}
		self.text = []
		self.class2app = collections.defaultdict(list)
		self.app2class = {}
		self.limit_length = 0

		if app2vec_model_path:
			self.app2vec_model = Word2Vec.load(app2vec_model_path)
	
		# load apps which need to be ignored.
		if goal_app_path:
			with open(goal_app_path,'r') as f:
				self.goal_app = f.read().split('\n')

		# load the mapping of id and app_name.
		if mapping_path:
			df = pd.read_csv(mapping_path,header = None)
			self.app2id = dict([row.tolist()[0].split(';') for index,row in df.iterrows()])
			self.id2app = {v: k for k, v in self.app2id.items()}

	# load the training data for App2vec.
	def load_training_data(self,raw_data_path):
		rf = open(raw_data_path,'rb')
		self.training_data = pickle.load(rf)
		rf.close()

	def load_resource(self,raw_data_path):
		rf = open(raw_data_path,'rb')
		result = pickle.load(rf)
		rf.close()
		return result

	def load_text(self,raw_data_path):
		rf = open(raw_data_path,'rb')
		self.text = pickle.load(rf)
		rf.close()

	# save the training data for app2vec.
	def save(self,data,write_file_path):
		wf = open(write_file_path,'wb')
		pickle.dump(data,wf)
		wf.close()

	def saveToApp(self,data,write_file_path):
		wf = open(write_file_path,'wb')
		data = [[self.id2app[app] for app in row] for row in data]
		pickle.dump(data,wf)
		wf.close()

	def saveText(self,data,write_file_path):
		wf = open(write_file_path,'wb')
		pickle.dump(data,wf)
		wf.close()

	# load R1 resource
	def processR1(self,R1_path,save = False):
		if os.path.exists('data/training_data/R1_data.txt'):
			self.training_data.extend(self.load_resource('data/training_data/R1_data.txt'))
		else:
			df = pd.read_csv(R1_path,header = None)

			for index,each_app_seq in df.iterrows():

				#Full cut mode
				if self.ignore_all:
					for each_app_list in (map(self._processR1, each_app_seq.tolist())):
						self.training_data.extend(each_app_list)
					
				#Select cut mode
				else:
					self.training_data.append([app for ele_app_list in each_app_seq.tolist() for app in each_app_list.split(' ') if app not in stop_app])

			if save:
				self.save(self.training_data,'data/training_data/R1_data.txt')

	def _processR1(self,each_app_seq):
		each_app_list = each_app_seq.split()
		result = []
		for app in each_app_list:
			app = self.id2app[app]
			if app not in self.goal_app:
				return []
			else:
				result.append(app)
		return [result]

	# load R2 resource.
	def processR2(self,R2_path,save = False):
		if os.path.exists('data/training_data/R2_data.txt'):
			self.training_data.extend(self.load_resource('data/training_data/R2_data.txt'))
			self.text.extend(self.load_resource('data/training_data/R2_text.txt'))
		else:
			df = pd.read_csv(R2_path,header = None)

			raw_data = [row.tolist()[0].split('\t') for index,row in df.iterrows()]

			all_data = [(row[1],row[2]) for row in raw_data if ';' in row[1]]

			data,text = map(list, zip(*all_data))

			self.text.extend(text)

			#Full cut mode
			if self.ignore_all:
				for each_app_list in map(self._processR2,data):
					self.training_data.extend(each_app_list)
				
			#Select cut mode
			else:
				self.training_data.append([app for ele_app_list in [row.split(' ')[1] for index,row in df.iterrows()] for app in each_app_list.split(';') if app not in self.stop_app])

			if save:
				self.save(self.training_data,'data/training_data/R2_data.txt')
				self.saveText(self.text,'data/training_data/R2_text.txt')

		self.limit_length = len(self.training_data)

	def _processR2(self,each_app_seq):
		each_app_list = each_app_seq.split(';')
		result = []
		for app in each_app_list:
			if app not in self.goal_app:
				return []
			else:
				result.append(app)
		return [result]

	def processR3(self,R3_path,save = False):

		if os.path.exists('data/training_data/R3_data.txt'):
			self.training_data.extend(self.load_resource('data/training_data/R3_data.txt'))
			self.text.extend(self.load_resource('data/training_data/R3_text.txt'))

		else:
			df = pd.read_csv(R3_path,header = None)

			raw_data = [row.tolist()[0].split('\t') for index,row in df.iterrows()]

			all_data = [(row[1],row[2]) for row in raw_data if ';' in row[1]]

			data,text = map(list, zip(*all_data))

			self.text.extend(text)

			#Full cut mode
			if self.ignore_all:
				for each_app_list in map(self._processR2,data):
					self.training_data.extend(each_app_list)
				
			#Select cut mode
			else:
				self.training_data.append([app for ele_app_list in [row.split(' ')[1] for index,row in df.iterrows()] for app in each_app_list.split(';') if app not in self.stop_app])

			if save:
				self.save(self.training_data,'data/training_data/R3_data.txt')
				self.saveText(self.text,'data/training_data/R3_text.txt')

	# for classifying.
	def classify_data(self,filepath):
		df = pd.read_excel(filepath)
		X,y = [],[]
		for index,row in df.iterrows():

			each_X = []

			for i in row['data'].split():
				if i not in self.goal_app:
					each_X = []
					break
				each_X.append(self.id2app[i])

			if each_X:
				X.append(each_X)
				y.append(row['relevant'])

		return X,y

	def training_data_without_doc(self):

		if not self.training_data:
			self.setup_training_data()

		X,y = [],[]

		for i in self.training_data:

			# Transfer to Index
			data = list(map(lambda x:self.app2vec_model.wv.vocab[x].index+1,i))

			
			half = len(i)//2
			X.append(data[:half])
			y.append(i[half:])
			
			'''
			for j in range(len(i)):
				X.append([data[j]])
				y.append(i[:j]+i[j+1:])
			'''
		return X,y

	def training_data_with_doc(self):

		if not self.training_data:
			self.setup_training_data()

		X,y = [],[]

		app_with_text_data = self.training_data[self.limit_length:]

		for index,data in enumerate(app_with_text_data):

			# Transfer to Index
			t_index = list(map(lambda x:self.app2vec_model.wv.vocab[x].index+1,data))
			'''
			half = len(data)//2
			X.append([t_index[:half],self.text[index]])
			y.append(data[half:])
			'''
			'''
			for j in range(len(data)):
				X.append([t_index[j],self.text[index]])
				y.append(data[:j]+data[j+1:])
				#text.append(self.des[str(i[j])])
			'''
		return X,y

	def generateClass(self,file_path,save):
		if os.path.exists('data/training_data/class2app.txt'):
			self.class2app = self.load_resource('data/training_data/class2app.txt')
			self.app2class = self.load_resource('data/training_data/app2class.txt')
			self.app2des = self.load_resource('data/training_data/app2des.txt')
		else:
			df = pd.read_excel(file_path)
			
			for index,row in df.iterrows():
				self.class2app[row['class']].append(str(row['package_name'])) 	
				self.app2class[row['package_name']] = row['class']
				self.app2des[row['package_name']] = row['description']

			if save:
				self.save(self.class2app,'data/training_data/class2app.txt')
				self.save(self.app2class,'data/training_data/app2class.txt')
				self.save(self.app2des,'data/training_data/app2des.txt')
		
	def checkClass(self,filter_result,length):
		'''
		result_count = 0
		run_count = 0
		result = []

		while(result_count!=length and run_count<len(filter_result)):
			if(self.app2class[filter_result[run_count]] not in result):
				result.append(self.app2class[filter_result[run_count]])
				result_count+=1
			run_count+=1
		'''
		#print(len(filter_result),length)
		return list(set([self.app2class[i] for i in filter_result]))[:5]

	def prepare_BI_LSTM_training_data(self,app2vec_model,max_len = 5,test_size = 0.1):
		'''
		max_len = number of apps in a sequence
		vector_dim = number of dimension of the vector
		'''
		vector_dim = len(app2vec_model.wv.syn0[0])

		X,y = self.training_data_without_doc()

		X_data = pad_sequences(maxlen = max_len,sequences = X,padding = 'post',value = 0)
		#y_data = pad_sequences(maxlen = max_len,sequences = y,padding = 'post',value = 0)

		X_vector = [[app2vec_model.wv.syn0[app_index-1] for app_index in each_app_seq] for each_app_seq in X]
		X_vector = pad_sequences(maxlen = max_len,sequences = X_vector,padding = 'post',value = 0)
		y_vector = [np.mean([app2vec_model[each_y] for each_y in seq_y],0) for seq_y in y]
		

		#initiation the training set
		X_training_data = np.zeros((len(X_data), max_len, vector_dim), dtype=np.float)
		#y_train = np.zeros((len(y), vector_dim), dtype=np.float)
		y_training_data = np.zeros((len(y_vector), vector_dim), dtype=np.float)

		for i in range(len(X_data)):

			for k in range(max_len):
				vector = X_vector[i][k]

				for j in range(len(vector)):
					X_training_data[i,k,j] = vector[j]

			vector = y_vector[i]
			for j in range(len(vector)):
				y_training_data[i,j] = vector[j]

		
		combined = list(zip(X_training_data, y_training_data))
		random.shuffle(combined)
		X_training_data[:], y_training_data[:] = zip(*combined)
		

		if test_size:

			size = int(len(X_training_data)*(1-test_size))
			X_train = X_training_data[:size]
			y_train = y_training_data[:size]

			X_test = X_training_data[size:]
			y_test = y[size:]

		else:

			X_train,X_test,y_train,y_test = X_training_data,y_training_data,None,None

		return X_train,X_test,y_train,y_test

	def prepare_BI_LSTM_training_doc_data(self,app2vec_model,max_len = 5,test_size = 0.1):
		'''
		max_len = number of apps in a sequence
		vector_dim = number of dimension of the vector
		'''
		vector_dim = len(app2vec_model.wv.syn0[0])

		# Load Testing data
		X,y = self.training_data_with_doc()

		combined = list(zip(X, y))
		random.shuffle(combined)
		X[:], y[:] = zip(*combined)

		X,X_text = zip(*X)

		X_data = pad_sequences(maxlen = max_len,sequences = X,padding = 'post',value = 0)
		#y_data = pad_sequences(maxlen = max_len,sequences = y,padding = 'post',value = 0)

		X_vector = [[app2vec_model.wv.syn0[app_index-1] for app_index in each_app_seq] for each_app_seq in X]
		X_vector = pad_sequences(maxlen = max_len,sequences = X_vector,padding = 'post',value = 0)
		y_vector = [np.mean([app2vec_model[each_y] for each_y in seq_y],0) for seq_y in y]
		

		#initiation the training set
		X_training_data = np.zeros((len(X_data), max_len, vector_dim), dtype=np.float)
		#y_train = np.zeros((len(y), vector_dim), dtype=np.float)
		y_training_data = np.zeros((len(y_vector), vector_dim), dtype=np.float)

		for i in range(len(X_data)):

			for k in range(max_len):
				vector = X_vector[i][k]

				for j in range(len(vector)):
					X_training_data[i,k,j] = vector[j]

			vector = y_vector[i]
			for j in range(len(vector)):
				y_training_data[i,j] = vector[j]

		
		
		

		if test_size:

			size = int(len(X_training_data)*(1-test_size))
			X_train = X_training_data[:size]
			y_train = y_training_data[:size]

			X_test = X_training_data[size:]
			y_test = y[size:]
			X_text = X_text[size:]

		else:

			X_train,X_test,y_train,y_test,X_text = X_training_data,y_training_data,None,None,None

		return X_train,X_test,y_train,y_test,X_text

	def setup_training_data(self,save = False):
		self.generateClass('data/training_data/class_map.xlsx',save)
		self.processR1('data/resources/R1.csv',save)
		self.processR2('data/resources/R2.csv',save)
		self.processR3('data/resources/R3.csv',save)

		if not os.path.exists('data/training_data/app2vec_training_data.txt'):
			self.save(self.training_data,'data/training_data/app2vec_training_data.txt')

class App2Vec(processData):
	def __init__(self):
		'''
		training_data：Store the training data.
		label2id：Store the mapping with cluster labels and app sequences.
		'''
		super().__init__()

	#----App2Vec----
	def training_App2Vec(self,app2vec_model_path,sg=1,size = 95,window = 3,seed = 0,min_count = 0,iter = 20000,compute_loss=True):
		'''
		app2vec_model_path：The storage location of the app2vec model.
		'''

		if not self.training_data:
			self.load_training_data('data/training_data/app2vec_training_data.txt')
		#Views more, https://radimrehurek.com/gensim/models/word2vec.html
		model = Word2Vec(self.training_data,sg=sg,size = size,window = window,seed = seed,min_count = min_count,iter = iter,compute_loss=compute_loss,workers = 5)

		#save the model
		model.save(app2vec_model_path)

	def load_App2Vec(self,app2vec_model_path):

		model = Word2Vec.load(app2vec_model_path)
		return model

	def show_app2vec(self,app2vec_model_path):
		app2vec_model = self.load_App2Vec(app2vec_model_path)
		X = app2vec_model[app2vec_model.wv.vocab]

		word_labels = [app2vec_model.wv.index2word[i] for i in range(len(X))]


		tsne = TSNE(n_components = 3)
		X_tsne = tsne.fit_transform(X)

		plt.scatter(X_tsne[:,0], X_tsne[:,1])

		for label,x,y in zip(word_labels,X_tsne[:,0], X_tsne[:,1]):
			plt.annotate(label,xy = (x, y), xytext = (0,0), textcoords = 'offset points')
		plt.xlim(X_tsne[:,0].min()+0.00005, X_tsne[:,0].max()+0.00005)
		plt.ylim(X_tsne[:,1].min()+0.00005, X_tsne[:,1].max()+0.00005)
		plt.show()

	def grid_app2vec(self,**param):
		'''
		Find the best paramters for app2vec model.
		'''

		# get R4 Resource which is for classifying.
		X,y = self.classify_data('data/training_data/data_relevant.xlsx')

		if not self.training_data:
			self.load_training_data('data/training_data/app2vec_training_data.txt')

		inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

		gsearch = GridSearchCV(app2vecEstimator.app2vec_estimator(self.training_data,size = 90,window = 3,min_count = 0,iter = 1000), param_grid = param, scoring = 'accuracy', cv = inner_cv)
		gsearch.fit(X,y)
		print('done')

		self.plot_grid_search(gsearch.cv_results_,param['iter'],param['size'],'iteration','size')
		print("CV_Result：")
		print("="*10)
		print(gsearch.cv_results_)
		print("="*10)
		print()
		print("Best_params：")
		print("="*10)
		print(gsearch.best_params_)
		print()
		print("Best score：")
		print("="*10)
		print(gsearch.best_score_)

	def plot_grid_search(self,cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
		# Get Test Scores Mean and std for each grid search
		plt.style.use('seaborn-white')
		scores_mean = cv_results['mean_test_score']
		scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

		scores_sd = cv_results['std_test_score']
		scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

		# Plot Grid search scores
		_, ax = plt.subplots(1,1)

		# Param1 is the X-axis, Param 2 is represented as a different curve (color line)
		for idx, val in enumerate(grid_param_2):
			ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

		ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
		ax.set_xlabel(name_param_1, fontsize=16)
		ax.set_ylabel('CV Average Score', fontsize=16)
		ax.legend(loc="best", fontsize=15)
		ax.grid('on')
		plt.show()

class BILSTM:
	def __init__(self,app2vec_model_path,max_len):
		self.app2vec_model = Word2Vec.load(app2vec_model_path)
		self.vector_dim = len(self.app2vec_model.wv.syn0[0])
		self.max_len = max_len

	def get_model(self,epochs = 50,batch_size = 30):
		p_data = processData(app2vec_model_path = 'data/Model/app2vec.model')
		X,y,_,_ = p_data.prepare_BI_LSTM_training_data(self.app2vec_model,test_size = 0)

		self.train_BILSTM(X_train = X,y_train = y,epochs = epochs,batch_size = batch_size,for_evaluate = False,store_path = 'data/Model/BILSTM_model.h5')

	def train_BILSTM(self,X_train,y_train,epochs = 50,batch_size = 30,max_len = 5,for_evaluate = False,store_path = None):

		rnn_size = 32
		learning_rate = 0.0001

		print('Building LSTM model...')
		model = Sequential()
		model.add(Bidirectional(LSTM(rnn_size, activation="relu"),input_shape=(self.max_len, self.vector_dim)))
		model.add(Dropout(0.5))
		model.add(Dense(self.vector_dim))
		
		optimizer = Adam(lr=learning_rate)
		callbacks=[EarlyStopping(patience=4, monitor='val_loss')]

		model_loss = self.compute_loss()

		model.compile(loss= model_loss, optimizer='rmsprop', metrics=[])
		print('LSTM model built.')
		# 'cosine_proximity'

		callbacks=[EarlyStopping(patience=2, monitor='val_loss'),
				   ModelCheckpoint(filepath='data/log' + "/" + 'my_model_sequence_lstm.{epoch:02d}.hdf5',\
								   monitor='val_loss', verbose=1, mode='auto', period=5)]

		history = model.fit(X_train, y_train,
				 batch_size=batch_size,
				 shuffle=True,
				 epochs=epochs,
				 callbacks=callbacks,
				 validation_split=0.1)

		#self.make_BI_LSTM_plot(history)
		if not for_evaluate:
			model.save(store_path)

		return model

	def make_BI_LSTM_plot(self,history):
		hist = pd.DataFrame(history.history)
		plt.style.use("ggplot")
		plt.figure(figsize=(12,12))
		plt.plot(hist["loss"])
		plt.plot(hist["val_loss"])
		plt.show()
		
	def load_BI_LSTM_model(self,filename,app2vec_model):

		model = load_model(filename,custom_objects={'compute_loss':self.compute_loss,'cos_distance':self.cos_distance})
		
		return model

	def compute_loss(self):
		return self.cos_distance

	def cos_distance(self,y_true, y_pred):
		y_pred = K.l2_normalize(y_pred, axis = -1)
		y_true = K.l2_normalize(y_true, axis = -1)
			
		return -K.mean(y_true * y_pred, axis=-1)

	def l2_normalize(self,x, axis):
		norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
		return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())

	def BI_LSTM_predict(self, X,BI_LSTM_model):

		X_vector = [[self.app2vec_model.wv.syn0[app_index-1] for app_index in each_app_seq] for each_app_seq in X]
		X_vector = pad_sequences(maxlen = self.max_len,sequences = X_vector,padding = 'post',value = 0)

		X_test = np.zeros((1, self.max_len, self.vector_dim), dtype = np.float)

		for k in range(max_len):
			vector = X_vector[0][k]

			for j in range(len(vector)):
				X_test[0,k,j] = vector[j]

		result = BI_LSTM_model.predict(X_test)[0]

		return result

class WordSemantic:
	def __init__(self):
		self.shared_dict = {}
		self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('data/model/GoogleNews-vectors-negative300.bin',binary = True)

	def calculateJob(self,sen,candidate,candidate_description):

		# Calculate the semantic score.
		score = self.calDocScore(sen,candidate_description)

		# Recording...
		if candidate in self.shared_dict:
			self.shared_dict[candidate] = max(score,self.shared_dict[candidate])
		else:
			self.shared_dict[candidate] = score

	def calDocScore(self,text1,text2):

		# Get their doc2vec
		text1 = np.mean([self.word2vec_model[word] for word in word_tokenize(text1) if word in self.word2vec_model],axis = 0)
		text2 = np.mean([self.word2vec_model[word] for word in word_tokenize(text2) if word in self.word2vec_model],axis = 0)
		
		# Calculate the cosine similarity
		score = 1 - spatial.distance.cosine(text1, text2)
		print(score)

		return score

class AF(processData,BILSTM,WordSemantic):
	def __init__(self,app2vec_model_path,max_len = 5,af_model_path = None):
		self.label2app = collections.defaultdict(list)
		self.af_model_path = af_model_path

		processData.__init__(self)
		BILSTM.__init__(self,app2vec_model_path,max_len)
		WordSemantic.__init__(self)

	def AF(self,max_iter = 3000,preference = -30,for_evaluate = False,lstm = False,doc = False):
		
		# Evaluating
		if for_evaluate:

			if doc:
				if lstm:
					self.evaluate_AF_BILSTM_doc(max_iter = max_iter,preference = preference,for_evaluate = True)
				else:
					self.evaluate_af_doc(max_iter = max_iter,preference = preference)
			else:
				if lstm:
					self.evaluate_AF_BILSTM(max_iter = max_iter,preference = preference,for_evaluate = True)
				else:
					self.evaluate_af(max_iter = max_iter,preference = preference)
				
		# Train AF model
		else:

			#store the training data of AF.
			af_training_data = []

			if not self.training_data:
				self.load_training_data('data/training_data/app2vec_training_data.txt')

			#Average the vector of each app sequence as a unit
			for app_seq in self.training_data:
				af_training_data.append(np.mean([self.app2vec_model[app] for app in app_seq],0))

			af_model = AffinityPropagation(max_iter = max_iter,preference = preference).fit(af_training_data)
			
			# save the model
			joblib.dump(af_model, af_model_path)

	def evaluate_af(self,max_iter,preference):

		# Load Testing data
		X,y = self.training_data_without_doc()

		#store the training data of AF.
		af_training_data = []

		#Average the vector of each app sequence as a unit
		for app_seq in self.training_data:
			af_training_data.append(np.mean([self.app2vec_model[app] for app in app_seq],0))


		# Get Testing data
		X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.9, random_state=0, shuffle = True)
		
		# Recording the performance
		cv_result = collections.defaultdict(list)

		for max_iter_param in max_iter:
			for preference_param in preference:
				
				# Train AF model
				af_model = AffinityPropagation(max_iter = max_iter_param,preference = preference_param).fit(af_training_data)

				# Get Label to their corrsponding app seqs.
				self._get_label2app(af_model)

				# For calculating the accuracy
				sum = 0
				total_num = 0
				
				for app_seq_id in range(len(X_test)):
					
					# Get the input vector
					vector = np.mean([self.app2vec_model.wv.syn0[app_index - 1] for app_index in X_test[app_seq_id]],0)

					# The predicted label
					predict_label = af_model.predict([vector])

					# transfer to app and count the frequency
					counter = collections.Counter(self.label2app[predict_label[0]])

					# Sort by frequency
					major_voting_filter = [app_with_count[0] for app_with_count in counter.most_common()]

					y = set([self.app2class[i] for i in y_test[app_seq_id]] )

					# Compare with true labels
					result = self.checkClass(major_voting_filter,len(y))

					# Count the correct records
					sum+=len(set(result).intersection(y))

					# Count the total number
					total_num+=len(y)
				

				# Record the accuracy.
				cv_result[max_iter_param].append((preference_param,sum/(total_num)))
					
		# Make Plot
		self.plot_af(cv_result)

	def evaluate_af_doc(self,max_iter,preference):

		# Load Testing data
		X,y = self.training_data_without_doc()

		#store the training data of AF.
		af_training_data = []

		#Average the vector of each app sequence as a unit
		for app_seq in self.training_data:
			af_training_data.append(np.mean([self.app2vec_model[app] for app in app_seq],0))

		# Get Testing data
		X_train,tmp_test,y_train,y_test = train_test_split(X,y, test_size=0.9, random_state=0, shuffle = True)

		# Get App seqs and their corrsponding sentence.
		X_test_data,X_test_text = zip(*tmp_test)

		# Recording the performance
		cv_result = collections.defaultdict(list)

		for max_iter_param in max_iter:
			for preference_param in preference:

				# Train AF model
				af_model = AffinityPropagation(max_iter = max_iter_param,preference = preference_param).fit(af_training_data)

				# Get Label to their corrsponding app seqs.
				self._get_label2app(af_model)

				# For calculating the accuracy
				sum = 0
				total_num = 0

				for app_seq_id in range(len(X_test_data)):
					# For recording the semantic score
					self.shared_dict = dict()

					# Get the input vector
					vector = np.mean([self.app2vec_model.wv.syn0[app_index - 1] for app_index in X_test[app_seq_id]],0)

					# The predicted label
					predict_label = af_model.predict([vector])

					# Get the candididate apps
					candidiates = list(set(self.label2app[predict_label[0]]))

					for candidiate_id in range(len(candidiates)):
						# Calculate the semantic score
						self.calculateJob(X_test_text[app_seq_id],candidiates[candidiate_id],self.app2des[candidiates[candidiate_id]])

					# Sort by semantic score
					semantic_filter = sorted(self.shared_dict,key=self.shared_dict.get,reverse = True)

					y = set([self.app2class[i] for i in y_test[app_seq_id]] )

					# Compare with true labels
					result = self.checkClass(semantic_filter,len(y))

					# Count the correct records
					sum+=len(set(result).intersection(y))

					# Count the total number
					total_num+=len(y)

				# Record the accuracy.
				cv_result[max_iter_param].append((preference_param,sum/(total_num)))
					
		# Make Plot
		self.plot_af(cv_result)
		
	def evaluate_AF_BILSTM(self,max_iter,preference,for_evaluate):
		
		# Prepare the training and testing data
		X_train,X_test,y_train,y_test = self.prepare_BI_LSTM_training_data(self.app2vec_model,test_size = 0.9)

		#store the training data of AF.
		af_training_data = []

		#Average the vector of each app sequence as a unit
		for app_seq in self.training_data:
			af_training_data.append(np.mean([self.app2vec_model[app] for app in app_seq],0))

		# Train BILSTM model
		BI_LSTM_model = self.train_BILSTM(X_train,y_train,for_evaluate = for_evaluate)
		
		# Recording the performance
		cv_result = collections.defaultdict(list)

		for max_iter_param in max_iter:
			for preference_param in preference:

				# Train AF model
				af_model = AffinityPropagation(max_iter = max_iter_param,preference = preference_param).fit(af_training_data)

				# Get Label to their corrsponding app seqs.
				self._get_label2app(af_model)

				# For calculating the accuracy
				sum = 0
				total_num = 0
				
				for app_seq_id in range(len(X_test)):

					# For recording the semantic score
					self.shared_dict = dict()
					
					X = np.array([X_test[app_seq_id]])

					# Get the predicted vector
					vector_predict = BI_LSTM_model.predict(X)

					# The predicted label
					predict_label = af_model.predict(vector_predict)

					# Most voting 
					counter = collections.Counter(self.label2app[predict_label[0]])

					# Sort by frequency
					major_voting_filter = [app_with_count[0] for app_with_count in counter.most_common()]

					y = set([self.app2class[i] for i in y_test[app_seq_id]] )

					# Compare with true labels
					result = self.checkClass(major_voting_filter,len(y))

					# Count the correct records
					sum+=len(set(result).intersection(y))

					# Count the total number
					total_num+=len(y)
					

				# Record the accuracy.
				cv_result[max_iter_param].append((preference_param,sum/(total_num)))
					
		# Make Plot
		self.plot_af(cv_result)
		
	def evaluate_AF_BILSTM_doc(self,max_iter,preference,for_evaluate):

		# Prepare the training and testing data
		X_train,X_test,y_train,y_test,X_text = self.prepare_BI_LSTM_training_doc_data(self.app2vec_model,test_size = 0.9)

		# get the vector of app2vec.
		vector = self.app2vec_model.wv.syn0

		#store the training data of AF.
		af_training_data = []

		#Average the vector of each app sequence as a unit
		for app_seq in self.training_data:
			af_training_data.append(np.mean([self.app2vec_model[app] for app in app_seq],0))

		# Train BILSTM model
		BI_LSTM_model = self.train_BILSTM(X_train,y_train,for_evaluate = for_evaluate)
		
		# Recording the performance
		cv_result = collections.defaultdict(list)

		for max_iter_param in max_iter:
			for preference_param in preference:

				# Train AF model
				af_model = AffinityPropagation(max_iter = max_iter_param,preference = preference_param).fit(af_training_data)

				# Get Label to their corrsponding app seqs.
				self._get_label2app(af_model)

				# For calculating the accuracy
				sum = 0
				total_num = 0
				
				for app_seq_id in range(len(X_test)):
					# For recording the semantic score
					self.shared_dict = dict()

					X = np.array([X_test[app_seq_id]])

					# Get the predicted vector
					vector_predict = BI_LSTM_model.predict(X)

					# The predicted label
					predict_label = af_model.predict(vector)

					# Get the candididate apps
					candidiates = list(set(self.label2app[predict_label[0]]))

					for candidiate_id in range(len(candidiates)):
						# Calculate the semantic score
						self.calculateJob(X_text[app_seq_id],candidiates[candidiate_id],self.app2des[candidiates[candidiate_id]])

					# Sort by semantic score
					semantic_filter = sorted(self.shared_dict,key=self.shared_dict.get,reverse = True)

					y = set([self.app2class[i] for i in y_test[app_seq_id]] )

					# Compare with true labels
					result = self.checkClass(semantic_filter,len(y))

					# Count the correct records
					sum+=len(set(result).intersection(y))

					# Count the total number
					total_num+=len(y)

				# Record the accuracy.
				cv_result[max_iter_param].append((preference_param,sum/(total_num)))
					
		# Make Plot
		self.plot_af(cv_result)

	def plot_af(self,cv_result):
		# Get Test Scores Mean and std for each grid search
		plt.style.use('seaborn-white')
		
		# Plot Grid search scores
		_, ax = plt.subplots(1,1)

		for key,value in cv_result.items():
			ax.plot([v[0] for v in value], [v[1] for v in value], '-o', label = 'Max_iter '+ str(key))

		ax.set_title("AffinityPropagation", fontsize=20, fontweight='bold')
		ax.set_xlabel('Preference', fontsize=16)
		ax.set_ylabel('Accuracy', fontsize=16)
		ax.legend(loc="best", fontsize=15)
		ax.grid('on')
		plt.show()

	def _get_label2app(self,af_model):
		# Get Label to their corrsponding app seqs.

		# Renewing..
		self.label2app = collections.defaultdict(list)

		# build a label2id dictionary
		for index,label in enumerate(af_model.labels_):
			self.label2app[label].extend(self.training_data[index])

	def get_af_model(self):
		# load af model
		af = joblib.load(self.af_model_path)

		# build a label2id dictionary
		for index,label in enumerate(af.labels_):
			self.label2app[label].extend(self.training_data[index])

		return af

class ANN(processData,BILSTM,WordSemantic):
	def __init__(self,app2vec_model_path,ann_model_path = 'data/Model/ann_model.ann',max_len = None):
		self.ann_accuracy = []
		self.ann_num_trees = []
		self.ann_model_path = ann_model_path

		processData.__init__(self)
		BILSTM.__init__(self,app2vec_model_path,max_len)
		WordSemantic.__init__(self)
		

	def load_ANN(self):

		ann = AnnoyIndex(self.vector_dim)
		ann.load(self.ann_model_path)
		return ann

	def _ANN_builder(self,num_tree):

		ann_model = AnnoyIndex(self.vector_dim)
		
		vector = self.app2vec_model.wv.syn0

		for i in self.app2vec_model.wv.vocab:
			#get the mapping id.
			index = self.app2vec_model.wv.vocab[i].index + 1

			#add the mapping.
			ann_model.add_item(index,vector[index-1])

		#train the app2vec. num_tree is the number of your ANN forest.
		ann_model.build(num_tree)

		#save the model
		ann_model.save(self.ann_model_path)
		
	def ANN(self,num_tree,for_evaluate = False,doc = False,lstm = False):
		'''
		dim = the Dimension of App2Vec.
		num_tree：The number of trees of your ANN forest. More tress more accurate.
		ann_model_path：The storage path of ANN model.
		'''	

		if for_evaluate:
			if doc:
				if lstm:
					self.evaluate_ANN_BILSTM_doc(num_trees = num_tree,for_evaluate = for_evaluate)
				else:
					self.evaluate_ann_doc(num_trees = num_tree)
			else:
				if lstm:
					self.evaluate_ANN_BILSTM(num_trees = num_tree,for_evaluate = for_evaluate)
				else:
					self.evaluate_ann(num_trees = num_tree)
		else:
			self._ANN_builder(num_tree)

	def evaluate_ann(self,num_trees):
		'''
		Without Doc2Vec
		'''

		# Load Testing data
		X,y = self.training_data_without_doc()

		# Get Testing data
		X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.9, random_state=0, shuffle = True)

		for num_tree in num_trees:
			
			# Build ANN
			self._ANN_builder(num_tree)

			# Load ANN model
			ann_model = self.load_ANN()

			# For calculating the accuracy
			sum = 0
			total_num = 0

			for app_seq_id in range(len(X_test)):

				# Get their neighbor and flat it to 1D.
				nbrs = list(itertools.chain.from_iterable([ann_model.get_nns_by_item(index,10) for index in X_test[app_seq_id]]))
				
				# transfer to app
				nbrs = [self.app2vec_model.wv.index2word[nbr-1] for nbr in nbrs]

				counter = collections.Counter(nbrs)

				# Sort by frequency
				major_voting_filter = [app_with_count[0] for app_with_count in counter.most_common()]

				y = set([self.app2class[i] for i in y_test[app_seq_id]] )

				# Compare with true labels
				result = self.checkClass(major_voting_filter,len(y))

				# Count the correct records
				sum+=len(set(result).intersection(y))

				# Count the total number
				total_num+=len(y)

			print(sum/total_num)
				

			# Record the accuracy
			self.ann_accuracy.append(sum/total_num)

			# Record the parameters
			self.ann_num_trees.append(num_tree)

		# Make Plot
		self.evaluate_ann_make_plot()

	def evaluate_ann_doc(self,num_trees):
		'''
		With Doc2Vec
		'''

		# Load Testing data
		X,y = self.training_data_with_doc()

		# Get Testing data
		X_train,tmp_test,y_train,y_test = train_test_split(X,y, test_size=0.9, random_state=0, shuffle = True)

		# Get App seqs and their corrsponding sentence.
		X_test_data,X_test_text = zip(*tmp_test)


		for num_tree in num_trees:
			
			# Build ANN
			self._ANN_builder(num_tree)

			# Load ANN model
			ann_model = self.load_ANN()

			# For calculating the accuracy
			sum = 0
			total_num = 0

			for app_seq_id in range(len(X_test_data)):
				
				# For recording the semantic score
				self.shared_dict = dict()

				# Get their neighbor and flat it to 1D.
				nbrs = list(itertools.chain.from_iterable([ann_model.get_nns_by_item(index,10) for index in X_test_data[app_seq_id]]))

				# Transfer to app
				nbr_app = [self.app2vec_model.wv.index2word[nbr-1] for nbr in nbrs]

				for nbr_id in range(len(nbr_app)):
					# Calculate the semantic score
					self.calculateJob(X_test_text[app_seq_id],nbr_app[nbr_id],self.app2des[nbr_app[nbr_id]])
				
				# Sort by semantic score
				semantic_filter = sorted(self.shared_dict,key=self.shared_dict.get,reverse = True)

				y = set([self.app2class[i] for i in y_test[app_seq_id]] )

				# Compare with true labels
				result = self.checkClass(semantic_filter,len(y))

				# Count the correct records
				sum+=len(set(result).intersection(y))

				# Count the total number
				total_num+=len(y)
				

			# Record the accuracy
			self.ann_accuracy.append(sum/total_num)

			# Record the parameters
			self.ann_num_trees.append(num_tree)

		# Make Plot
		self.evaluate_ann_make_plot()

	def evaluate_ANN_BILSTM(self,num_trees,for_evaluate):

		# Prepare the training and testing data
		X_train,X_test,y_train,y_test = self.prepare_BI_LSTM_training_data(self.app2vec_model,test_size = 0.9)

		# Train BILSTM model
		BI_LSTM_model = self.train_BILSTM(X_train,y_train,for_evaluate = for_evaluate)

		for num_tree in num_trees:
			
			# Build ANN
			self._ANN_builder(num_tree)

			# Load ANN model
			ann_model = self.load_ANN()

			# For calculating the accuracy
			sum = 0
			total_num = 0

			for app_seq_id in range(len(X_test)):
				
				X = np.array([X_test[app_seq_id]])

				vector_predict = BI_LSTM_model.predict(X)[0]

				# Get their neighbor.
				nbrs = ann_model.get_nns_by_vector(vector_predict,10)

				# Transfer to apps
				nbrs = [self.app2vec_model.wv.index2word[nbr-1] for nbr in nbrs]

				counter = collections.Counter(nbrs)

				# Sort by frequency
				major_voting_filter = [app_with_count[0] for app_with_count in counter.most_common()]

				y = set([self.app2class[i] for i in y_test[app_seq_id]] )

				# Compare with true labels
				result = self.checkClass(major_voting_filter,len(y))

				# Count the correct records
				sum+=len(set(result).intersection(y))

				# Count the total number
				total_num+=len(y)
				

			# Record the accuracy
			self.ann_accuracy.append(sum/total_num)

			# Record the parameters
			self.ann_num_trees.append(num_tree)

		# Make Plot
		self.evaluate_ann_make_plot()

	def evaluate_ANN_BILSTM_doc(self,num_trees,for_evaluate):

		# Prepare the training and testing data
		X_train,X_test,y_train,y_test,X_text = self.prepare_BI_LSTM_training_doc_data(self.app2vec_model,test_size = 0.9)

		# Train BILSTM model
		BI_LSTM_model = self.train_BILSTM(X_train,y_train,for_evaluate = for_evaluate)

		for num_tree in num_trees:
			
			# Build ANN
			self._ANN_builder(num_tree)

			# Load ANN model
			ann_model = self.load_ANN()

			# For calculating the accuracy
			sum = 0
			total_num = 0

			for app_seq_id in range(len(X_test)):

				# For recording the semantic score
				self.shared_dict = dict()
				
				X = np.array([X_test[app_seq_id]])

				vector_predict = BI_LSTM_model.predict(X)[0]

				# Get their neighbor and flat it to 1D.
				nbrs = ann_model.get_nns_by_vector(vector_predict,10)

				# Transfer to app
				nbr_app = [self.app2vec_model.wv.index2word[nbr-1] for nbr in nbrs]

				for nbr_id in range(len(nbr_app)):
					# Calculate the semantic score
					self.calculateJob(X_text[app_seq_id],nbr_app[nbr_id],self.app2des[nbr_app[nbr_id]])
				
				# Sort by semantic score
				semantic_filter = sorted(self.shared_dict,key=self.shared_dict.get,reverse = True)

				y = set([self.app2class[i] for i in y_test[app_seq_id]] )

				# Compare with true labels
				result = self.checkClass(semantic_filter,len(y))

				# Count the correct records
				sum+=len(set(result).intersection(y))

				# Count the total number
				total_num+=len(y)
				

			# Record the accuracy
			self.ann_accuracy.append(sum/total_num)

			# Record the parameters
			self.ann_num_trees.append(num_tree)

		# Make Plot
		self.evaluate_ann_make_plot()

	def evaluate_ann_make_plot(self):
		'''
		Make plot for ANN.
		'''
		plt.plot(self.ann_num_trees,self.ann_accuracy)
		plt.title('num_trees vs. accuracy')
		plt.ylabel('% accuracy')
		plt.xlabel('num_tress')
		plt.show()












if __name__ == "__main__":
	# -*- Train Affinity Propagation
	af = AF(app2vec_model_path = 'data/Model/app2vec.model',max_len = 5,af_model_path = 'data/Model/af_model.pkl')

	# Goal: Find the best parameters
	# With Doc2Vec, set doc to True.
	# With BILSTM, set lstm to True.
	# and vice versa.
	af.AF(max_iter = [10000],
	      preference = range(-20,-31,-10), 
	      for_evaluate = True,
	      lstm = False, 
	      doc = False)
	 
	# After finding, we can train our AF model.
	#af.AF(max_iter = 4000,preference = -30,for_evaluate = False)

