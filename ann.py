import app2vec
from sklearn.model_selection import train_test_split

#View more, https://github.com/spotify/annoy.
from annoy import AnnoyIndex

class ANN:
	def __init__(self,app2vec_model_path):
		App2Vec = app2vec()
		self.app2vec_model = App2Vec.load_App2Vec(app2vec_model_path)

	#Train the ANN
	def train_ANN(self,dim,num_tree,ann_model_path):
		'''
		dim = the Dimension of App2Vec.
		num_tree：The number of trees of your ANN forest. More tress more accurate.
		ann_model_path：The storage path of ANN model.
		'''

		#get the vector of app2vec.
		vector = self.app2vec_model.wv.syn0
		
		t = AnnoyIndex(dim)
		
		for i in self.app2vec_model.wv.vocab:
			#get the mapping id.
			index = self.app2vec_model.wv.vocab[str(i)].index

			#add the mapping.
			t.add_item(index,vector[index])

		#train the app2vec. num_tree is the number of your ANN forest.
		t.build(num_tree)

		#save the model
		t.save(ann_model_path)

	def load_ANN(self,model_path,dim):
		'''
		Load the ANN model.
		'''
		ann = AnnoyIndex(dim)
		ann.load(model_path)
		return ann

	def evaluate_ann(self,X,y,dim,app2vec_model_path,ann_model_path):
		'''
		Evalue the ANN model.
		'''
		t = self.load_ANN(app2vec_model_path,dim)
		X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.8, random_state=0, shuffle = True)

		sum = 0
		for i in range(len(X_test)):
			index = app2vec_model.wv.vocab[str(X_test[i][0])].index
			nbr = t.get_nns_by_item(index, len(y_test[i]))
			true_y = list(map(lambda each_y:self.app2vec_model.wv.vocab[str(each_y)].index,y_test[i]))

			
			sum+=len(set(nbr).intersection(set(true_y)))

		return sum/len(y_test)