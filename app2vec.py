from gensim.models import Word2Vec
import pandas as pd

class App2Vec:
	def __init__(self,stop_app_path):
		'''
		training_data：Store the training data
		stop_app：Store the stop app. These stop apps won't treat as the training data for App2Vec.
		'''

		self.training_data = []
		with open(stop_app,'r') as f:
			self.stop_app = f.read().split('\n')

	# half ignore process mode..
	def ignore_all_get_app(self,each_row):
		each_row = each_row.split()
		res = []
		for ele in each_row:
			if ele in self.stop_app:
				return []
			else:
				res.append(ele)
		return [res]

	# provide the training data for App2Vec.
	def csv2training_data(self,raw_file_path,ignore_all = True):
		'''
		file_path：The storage location of your raw training data.
		ignore_all(Optional)：Ignore mode，True is Full ignore mode，False is half ignore mode.
		'''

		df = pd.read_csv(file_path,header = None)

		for i,j in df.iterrows():

			if ignore_all:
				for each_row in (map(self.ignore_all_get_app,j.tolist())):
					self.training_data.extend(each_row)
			else:
				self.training_data.append([k for ele in j.tolist() for k in ele.split(' ') if k not in stop_app])

			
	#Train the app2vec.
	def training_App2Vec(self,app2vec_model_path):
		'''
		app2vec_model_path：The storage location of the app2vec model.
		'''

		#Views more, go to https://radimrehurek.com/gensim/models/word2vec.html
		model = Word2Vec(self.training_data,sg=1,size = 128,window = 3,seed = 0,min_count = 0,iter = 10,compute_loss=True)
		model.save(app2vec_model_path)

	#Train the ANN
	def ANN(self,dim,num_tree,app2vec_model_path,ann_model_path):
		'''
		dim = the Dimension of App2Vec.
		num_tree：The number of trees of your ANN forest. More tress more accurate.
		ann_model_path：The storage path of ANN model.
		'''
		
		#View more, go to https://github.com/spotify/annoy.
		from annoy import AnnoyIndex

		#load app2vec model.
		model = Word2Vec.load(model_path)

		#get the vector of app2vec.
		vector = model.wv.syn0
		
		t = AnnoyIndex(dim)
		
		
		for i in model.wv.vocab:
			#get the mapping id.
			index = model.wv.vocab[str(i)].index

			#add the mapping.
			t.add_item(index,vector[index])

		#train the app2vec. num_tree is the number of your ANN forest.
		t.build(num_tree)

		#save the model
		t.save(ann_model_path)
		
		
if __name__ == "__main__":
	app2vec = App2Vec()
	app2vec.csv2training_data(raw_file_path = '/Users/apple/Documents/raw_data.csv')
	app2vec.training_App2Vec(app2vec_model_path = '/Users/apple/Documents/app2vec.model')
	app2vec.ANN(dim = 64,num_tree = 10000,app2vec_model_path = '/Users/apple/Documents/app2vec.model',ann_model_path = '/Users/apple/Documents/ANN.model')