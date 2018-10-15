from gensim.models import Word2Vec
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import app2vecEstimator
from sklearn.model_selection import GridSearchCV, KFold

class App2Vec:
	def __init__(self):
		'''
		training_data：Store the training data.
		label2id：Store the mapping with cluster labels and app sequences.
		'''

		self.training_data = []

	# load the training data for App2vec.
	def load_training_data(self,raw_data_path):
		rf = open(raw_data_path,'rb')
		self.training_data = pickle.load(rf)
		rf.close()	
			
	#Train the app2vec.
	def training_App2Vec(self,app2vec_model_path):
		'''
		app2vec_model_path：The storage location of the app2vec model.
		'''

		#Views more, https://radimrehurek.com/gensim/models/word2vec.html
		model = Word2Vec(self.training_data,sg=1,size = 95,window = 3,seed = 0,min_count = 0,iter = 1000000,compute_loss=True)

		#save the model
		model.save(app2vec_model_path)

	def load_App2Vec(self,app2vec_model_path):
		'''
		Load the app2vec model.
		'''
		model = Word2Vec.load(app2vec_model_path)
		return model

	def show_app2vec(self,app2vec_model_path):
		'''
		make a plot of the app2vec model.
		'''

		app2vec_model = Word2Vec.load(app2vec_model_path)
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

	def grid_app2vec(self,X,y,app2vec_model_path):
		'''
		Find the best paramters for app2vec model.
		'''

		
		param = {
			'size' : [i for i in range(50,200,10)],
			'window' : [5,10],
		}

		inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

		gsearch = GridSearchCV(app2vecEstimator.app2vec_estimator(X,size = 90,window = 3,min_count = 0), param_grid = param, scoring = 'accuracy', cv = inner_cv)
		gsearch.fit(X,y)

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