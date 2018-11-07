# Implicit-Intent-Inference-Model
This interface are for processing multi-implicit-intent. It allows you to easily build eight different models (including ANN-based model, Affinity Propagation-based model, ANN-based model with Doc2Vec, Affinity Propagation-based with Doc2Vec, BILSTM-ANN-based model, BILSTM-ANN-based model with Doc2Vec, BILSTM-Affinity Propagation based model, BILSTM-Affinity Propagation based model with Doc2Vec). Moreover, we also provide the evaluating function for optimizing these model.

# Require
1. Download GoogleNews word2vec --> https://github.com/mmihaltz/word2vec-GoogleNews-vectors. After unzipping it, put it to 'data/Model/' directory.
2. create the data/log folder
3. pip install -r requirements.txt


# Usage
## Training
```
Execute Training/Model.py
```

## App2Vec
```
# -*- Train App2Vec -*-
ap = App2Vec()

# Currently we only can show the plot of iter and size.
# Find the best parameters for App2Vec.
ap.grid_app2vec(size = range(95,101,5),
		iter = range(1,6,5),
		window = range(4,5,1))

# After finding, we can train out App2Vec model.
ap.training_App2Vec(app2vec_model_path = 'data/Model/app2vec.model')

# Make plot for our App2Vec model.
ap.show_app2vec(app2vec_model_path = 'data/Model/app2vec.model')
```

## BILSTM
``` 
# -*- Train BILSTM
bilstm = BILSTM(app2vec_model_path = 'data/Model/app2vec.model',max_len = 5)
bilstm.get_model(epochs = 50,batch_size = 30)
```

## Weighted Matrix Factorization
```
p_data = processData()

app2vec_model = Word2Vec.load('data/Model/app2vec.model')

# Train the Matrix Factorization model.
p_data.wmf_model(app2vec_model = app2vec_model)
```

## ANN
```
# -*- Train ANN
ann = ANN(app2vec_model_path = 'data/Model/app2vec.model',ann_model_path = 'data/Model/ann_model.ann',max_len = 5)

# Goal: Find the best parameters
# With BILSTM, set lstm to True, vice versa.
# You can set ranker to 'mv'(major voting filter), 'doc'(semantic filter), 'mf'(matrix factorization filter)
ann.ANN(num_tree = range(10000,20001,10000),
        for_evaluate = True,
        lstm = True,
	ranker = 'mv')

# After finding, we can train our ANN model.
ann.ANN(num_tree = 18000,for_evaluate = False)
```
## Affinity Propagation
``` 
# -*- Train Affinity Propagation
af = AF(app2vec_model_path = 'data/Model/app2vec.model',max_len = 5,af_model_path = 'data/Model/af_model.pkl')

# Goal: Find the best parameters
# With BILSTM, set lstm to True, vice versa.
# You can set ranker to 'mv'(major voting filter), 'doc'(semantic filter), 'mf'(matrix factorization filter)
af.AF(max_iter = range(1000,4001,1000),
      preference = range(-10,-41,-10), 
      for_evaluate = True,
      lstm = False, 
      ranker = 'doc')
 
# After finding, we can train our AF model.
af.AF(max_iter = 4000,preference = -30,for_evaluate = False)
```

## Predict
```
Execute main.py
```
```
# Load the app's description
data = {}
with open(r'Training/data/Model/app2des.json','r',encoding = 'utf-8') as f:
	data = json.load(f)

# Load the mapping of explict intent and apps
mapping = {}
with open(r'Training/data/Model/app_mapping.json','r',encoding = 'utf-8') as f:
	mapping = json.load(f)

# If someone's intent is 問路
imip = IMIP(explicit_intent = ['問路'],intentApp = mapping,app2vec_model_path = r'Training/data/Model/app2vec.model',ann_model_path = r'Training/data/Model/ann_model.ann',af_model_path = r'Training/data/Model/af_model.pkl',app2des = data)

# If someone says 我想要去公園吃飯和玩
# The parameter:
# model : ANN or AF
# ranker : mv, mf or doc
# lstm : True or False
print(imip.query(HanziConv.toSimplified('我想要去公園吃飯和玩'),model = 'ANN',ranker = 'doc',lstm = False))

```
