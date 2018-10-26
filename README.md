# Implicit-Intent-Inference-Model
This interface are for processing multi-implicit-intent. It allows you to easily build eight different models (including ANN-based model, Affinity Propagation-based model, ANN-based model with Doc2Vec, Affinity Propagation-based with Doc2Vec, BILSTM-ANN-based model, BILSTM-ANN-based model with Doc2Vec, BILSTM-Affinity Propagation based model, BILSTM-Affinity Propagation based model with Doc2Vec). Moreover, we also provide the evaluating function for optimizing these model.

# Background
## App2Vec
App2Vec is an unsupervised learning method to embed words into a dense vector space. In this dense vector space, semantically and syntactically related words are close to each other. App2Vec uses a shallow neural network that is trained to learn the distribution representation of words. In basic, Word2Vec is a single layer neural network with one hidden layer. Both input and output are represented as the One-Hot encoding. The learned vector of words is stored as input-hidden layer weight matrix.

There are two modes of Word2Vec model: (1) Continuous Bags of Word (CBOW) and (2) Continuous Skip-Gram.
### CBOW
CBOW model is given the context words to predict the center word in the window. In order to represent the vector of context words, CBOW averages or sums the vector of each context word. Given context words of the current words, the objective of CBOW is to maximize the log probability of current word.

### Continuous Skip-Gram
Contrary to CBOW, Continuous Skip Gram model is given the current word to predict various individual context words in the window. Given the current word, the goal of Continuous Skip-Gram model is to maximize the log probability of context words.

![image](image/skip-gram.png)

In this interface, we use gensim library to achieve it (https://radimrehurek.com/gensim/models/word2vec.html).

## ANN
The objective of the nearest neighbor search is to find objects similar to the query point from a collection of objects. However, the processing cost is very high when the nearest neighbor search is applied to a high-dimensional data. For this reason, Approximate Nearest Neighbor(ANN) search is proposed to tackle this problem. ANN reduces the cost of processing greatly by sacrificing a little accuracy but get similar results to nearest neighbor search. ANN can be roughly categorized into three groups (Fu and Cai, 2016): Data Structure-based, Hash-based.

In this interface, we use AnnoyIndex library to achieve it. AnnoyIndex is a hash-based ANN (https://github.com/spotify/annoy).

## Affinity Propagation
Affinity Propagation is a unsupervised learning method which does not require the pre-defined number of clusters. It can automatically find a collection of objects which are representative of clusters and discover the number of clusters. In order to find the exemplars for each cluster, Affinity Propagation takes a set of pairwise similarities as input and passes the messages between these pairwise data objects. In this training stage, Affinity Propagation updates two matrices  and .  represent the responsibility of each object. A higher value for the  of object in cluster  means that object would be a better exemplar for cluster .  represent the availability of each object. A higher value for the  of object in cluster  means that object would be likely to belong to cluster . This updating is executed iteratively until convergence. Once convergence is achieved, exemplars of each cluster are generated. Affinity Propagation outputs the final clusters.

In this interface, we use  Scikit-Learn library to achieve it (http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html).

# Require
Download GoogleNews word2vec --> https://github.com/mmihaltz/word2vec-GoogleNews-vectors.
After unzipping it, put it to 'data/Model/' directory.

# Usage
## Execute
```
Execute Training/Model.py
```

## Prepare Training data
```
p_data = processData()

# just one line, then all set.
p_data.setup_training_data(save = True)
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

## Matrix Factorization
```
p_data = processData()

app2vec_model = Word2Vec.load('data/Model/app2vec.model')

# Train the Matrix Factorization model.
p_data.mf_model(app2vec_model = app2vec_model, 
		K = 2, alpha = 0.1, 
		beta = 0.01, 
		iterations = 1000, 
		retrain = True)
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
