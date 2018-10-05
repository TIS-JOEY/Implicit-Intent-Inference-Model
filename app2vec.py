from gensim.models import Word2Vec
import pandas as pd

class App2Vec:
	def __init__(self,stop_app_path):
		'''
		training_data：儲存訓練資料
		stop_app：儲存停用app
		'''

		self.training_data = []
		with open(stop_app,'r') as f:
			self.stop_app = f.read().split('\n')

	#半忽略模式過濾
	def ignore_all_get_app(self,each_row):
		each_row = each_row.split()
		res = []
		for ele in each_row:
			if ele in self.stop_app:
				return []
			else:
				res.append(ele)
		return [res]

	#準備Word2Vec訓練資料，我們要將每一段的app使用紀錄存成[[A,B],[B,C,D],[D,E,A]...]，每一個子串列為一段使用紀錄，Word2Vec會將其看作是一個句子，以學習他們之中的關聯性。
	def csv2training_data(self,raw_file_path,ignore_all = True):
		'''
		file_path：raw_data(csv檔)
		stop_app：要忽略的app
		ignore_all(Optional)：忽略模式，True為全忽略模式，False為半忽略模式
		'''

		#讀取app使用記錄檔
		df = pd.read_csv(file_path,header = None)

		#逐列讀取
		for i,j in df.iterrows():

			if ignore_all:
				for each_row in (map(self.ignore_all_get_app,j.tolist())):
					self.training_data.extend(each_row)
			else:
				self.training_data.append([k for ele in j.tolist() for k in ele.split(' ') if k not in stop_app])

			
	#進行App2Vec訓練
	def training_App2Vec(self,model_path):
		'''
		model_path：app2vec模型儲存位置
		'''

		#此train	ing_data為prepare_training_data方法所創建出來的
		#此參數可變動，詳情請參閱https://radimrehurek.com/gensim/models/word2vec.html
		model = Word2Vec(self.training_data,sg=1,size = 128,window = 3,seed = 0,min_count = 0,iter = 10,compute_loss=True)
		model.save(model_path)

	#投入ANN進行訓練
	def ANN(self,dim,num_tree,app2vec_model_path,ann_model_path):
		'''
		dim：app2vec訓練時的維度
		num_tree：要訓練幾棵樹，Annoy官方文檔指出，建立越多數，其準確率越好
		ann_model_path：ANN模型儲存位置
		'''

		from annoy import AnnoyIndex

		#載入模型
		model = Word2Vec.load(model_path)

		#取得app2vec向量
		vector = model.wv.syn0
		
		t = AnnoyIndex(dim)
		
		
		for i in model.wv.vocab:
			#得到對應的id
			index = model.wv.vocab[str(i)].index

			#將id與vector對應，並投入ANN進行訓練
			t.add_item(index,vector[index])

		#訓練...
		t.build(num_tree)

		#儲存模型
		t.save(ann_model_path)
		
		
if __name__ == "__main__":
	app2vec = App2Vec()
	app2vec.csv2training_data(raw_file_path = '/Users/apple/Documents/raw_data.csv')
	app2vec.training_App2Vec()
	app2vec.ANN(dim = 64,num_tree = 10000,app2vec_model_path = '/Users/apple/Documents/app2vec.model',ann_model_path = '/Users/apple/Documents/ANN.model')
