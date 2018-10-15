import pickle
import pandas

class processData:
	def __init__(self, stop_app_path = None, mapping_path = None, ignore_all = True):
		'''
		mapping：Store the mapping of id and app_name
		training_data：Store the training data
		stop_app：Store the stop app. These stop apps won't treat as the training data for App2Vec.
		ignore_all：True is full cut mode, False is select cut mode.
		'''

		self.mapping = {}
		self.stop_app = []
		self.training_data = []
		self.ignore_all = ignore_all

		# load apps which need to be ignored.
		if stop_app_path:
			with open(stop_app_path,'r') as f:
				self.stop_app = f.read().split('\n')

		# load the mapping of id and app_name.
		if mapping_path:
			df = pd.read_csv(mapping_path,header = None)
			self.mapping = dict([row.tolist()[0].split(';') for index,row in df.iterrows()])

	def _csv2training_data(self,each_app_seq):
		each_app_list = each_app_seq.split()
		result = []
		for app in each_app_list:
			if app in self.stop_app:
				return []
			else:
				result.append(app)
		return [result]

	#create App2Vec training data
	def csv2App2Vec_training_data(self,raw_file_path):
		df = pd.read_csv(raw_file_path,header = None)

		for index,each_app_seq in df.iterrows():

			#Full cut mode
			if self.ignore_all:
				for each_app_list in (map(self._csv2training_data, each_app_seq.tolist())):
					self.training_data.extend(each_app_list)
				
			#Select cut mode
			else:
				self.training_data.append([app for ele_app_list in each_app_seq.tolist() for app in each_app_list.split(' ') if app not in stop_app])

	# For evaluating App2Vec model.
	def csv2evaluate_App2Vec_training_data(self,raw_file_path):
		df = pd.read_csv(raw_file_path,header = None)
		X,y = [],[]
		for index,row in df.iterrows():

				app_seq,label = zip(row.tolist())

				X.append(app_seq[0].split(' '))
				y.append(label[0])
		return X,y

	# For evaluating ANN model.
	def csv2evaluate_ANN_training_data(self,raw_file_path):
		X,y = [],[]
		self.csv2training_data(raw_file_path)
		for i in self.training_data:
			for j in range(len(i)):
				X.append([i[j]])
				y.append(i[:j]+i[j+1:])
		return X,y

	# save the training data.
	def save(self,write_file_path):
		wf = open(write_file_path,'wb')
		pickle.dump(self.training_data,wf)
		wf.close()