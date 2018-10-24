import pandas as pd
import numpy as np
import codecs
from hanziconv import HanziConv
import re
import jieba
from sklearn.model_selection import train_test_split
from keras_contrib.layers import CRF
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
import matplotlib.pyplot as plt
from keras_contrib.utils import save_load_utils
import json
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


class processData:
	def __init__(self):
		self.count = 1
		self.training_data = []
		self.re_words = re.compile(u"[\u4e00-\u9fa5]+")
		self.word2id = {}
		self.id2word = {}
		self.mapping_count = 1
		self.tag2id = {'O':0,'COMMENT':1,'PERIOD':2}
		self.id2tag = {0:'O',1:'COMMENT',2:'PERIOD'}

	def _processAns(self,cut_sen):
		if(cut_sen != ''):
			sen = list(jieba.cut(cut_sen))
			result = []
			for i in range(len(sen)):
				if(sen[i]=='，' and result!=[]):
					result[-1][1] = 'COMMENT'
				elif(self.re_words.search(sen[i])):
					if(sen[i] not in self.word2id):
						self.word2id[sen[i]] = self.mapping_count
						self.id2word[self.mapping_count] = sen[i]
						self.mapping_count+=1
					result.append([sen[i],'O'])

			if(result!=[]):
				result[-1][1] = 'PERIOD'
				result = list(map(lambda x:tuple(x),result))
			else:
				result = None
			return result

	def processAns(self):
		with codecs.open('corpus.txt','r') as f:
			for i in f.readlines():
				if(i.startswith(str(self.count))):
					each_par = HanziConv.toTraditional(i.split('++$++')[1].split()[0])
					each_lists = each_par.replace('！','。').split('。')

					train_unit = list(map(self._processAns,each_lists))

					if train_unit!=None:		
						self.training_data.extend(list(filter(lambda x: x != None, train_unit)))
						self.count+=1

		self.saveParameters()

	def getParameters(self):
		return self.training_data,self.word2id,self.id2word,self.tag2id,self.id2tag

	def saveParameters(self):
		with open('CRF_data.txt','w') as f:
			f.write(json.dumps(self.training_data))

		with open('word2id.txt','w') as f:
			f.write(json.dumps(self.word2id))

		with open('id2word.txt','w') as f:
			f.write(json.dumps(self.id2word))

		with open('tag2id.txt','w') as f:
			f.write(json.dumps(self.tag2id))

		with open('id2tag.txt','w') as f:
			f.write(json.dumps(self.id2tag))

	def loadParameters(self):
		with open('CRF_data.txt','r') as f:
			self.training_data = json.load(f)

		with open('word2id.txt','r') as f:
			self.word2id = json.load(f)

		with open('id2word.txt','r') as f:
			self.id2word = json.load(f)

		with open('tag2id.txt','r') as f:
			self.tag2id = json.load(f)

		with open('id2tag.txt','r') as f:
			self.id2tag = json.load(f)


class BI_LSTM_CRF:
	def __init__(self,training_data,word2id,id2word,tag2id,id2tag):
		self.training_data = training_data
		self.word2id = word2id
		self.id2word = id2word
		self.tag2id = tag2id
		self.id2tag = id2tag
		self.max_len = 50
		self.num_word = len(word2id)
		self.num_tag = len(tag2id)
		self.X = None
		self.y = None

	def pad_process(self):
		X = [[int(self.word2id[w[0]]) for w in s] for s in self.training_data]
		self.X = pad_sequences(maxlen = self.max_len,sequences = X,padding = 'post',value = 0)

		y = [[int(self.tag2id[w[1]]) for w in s] for s in self.training_data]
		y = pad_sequences(maxlen = self.max_len,sequences = y,padding = "post",value = self.tag2id['O'])
		self.y = [to_categorical(i, num_classes = self.num_tag) for i in y]

	def train_test_split(self):
		return train_test_split(self.X, self.y, test_size=0.1)

	def train(self):
		input = Input(shape = (self.max_len,))
		model = Embedding(input_dim = self.num_word + 1, output_dim = 20, input_length = self.max_len, mask_zero = True)(input)
		model = Dropout(0.1)(model)
		model = Bidirectional(LSTM(units = 50, return_sequences = True, recurrent_dropout = 0.1))(model)
		model = TimeDistributed(Dense(50, activation="relu"))(model)  # softmax output layer
		crf = CRF(self.num_tag)
		out = crf(model)
		model = Model(input, out)
		model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

		self.pad_process()
		train_X,test_X, train_y, test_y = self.train_test_split()

		history = model.fit(train_X, np.array(train_y), batch_size=32, epochs=1,validation_split=0.1, verbose=1)

		self.make_plot(history)
		return model

	def evaluate(self,model):
		self.pad_process()
		train_X,test_X, train_y, test_y = self.train_test_split()

		test_pred = model.predict(test_X,verbose = 1)

		pred_labels = self._evaluate(test_pred)
		test_labels = self._evaluate(test_y)
		print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
		print(classification_report(test_labels, pred_labels))

	def _evaluate(self,pred):
		out = []
		for pred_i in pred:
			out_i = []
			for p in pred_i:
				p_i = np.argmax(p)
				out_i.append(self.id2tag[p_i].replace("PAD","O"))
			out.append(out_i)
		return out

	def make_plot(self,history):
		hist = pd.DataFrame(history.history)
		plt.style.use("ggplot")
		plt.figure(figsize=(12,12))
		plt.plot(hist["acc"])
		plt.plot(hist["val_acc"])
		plt.show()

	def save_embedding_bilstm2_crf_model(self,model, filename):
		save_load_utils.save_all_weights(model,filename)

	def get_model(self,filename):
		model = self.train()
		save_load_utils.load_all_weights(model, filename)
		return model

	def load_model(self,filename):
		input = Input(shape = (self.max_len,))
		model = Embedding(input_dim = self.num_word + 1, output_dim = 20, input_length = self.max_len, mask_zero = True)(input)
		model = Dropout(0.1)(model)
		model = Bidirectional(LSTM(units = 50, return_sequences = True, recurrent_dropout = 0.1))(model)
		model = TimeDistributed(Dense(50, activation="relu"))(model)  # softmax output layer
		crf = CRF(self.num_tag)
		out = crf(model)
		model = Model(input, out)
		model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

		save_load_utils.load_all_weights(model, filename,include_optimizer=False)
		return model

	def predict(self):
		self.pad_process()
		train_X,test_X, train_y, test_y = self.train_test_split()


		model = self.load_model('BI_LSTM_CRF_model.h5')
		i = 1927
		p = model.predict(np.array([test_X[i]]))
		p = np.argmax(p, axis=-1)
		true = np.argmax(test_y[i], -1)
		print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
		print(30 * "=")
		for w, t, pred in zip(test_X[i], true, p[0]):
		    if w != 0:
		        print("{:15}: {:5} {}".format(self.id2word[w-1], self.id2tag[t], self.id2tag[pred]))

