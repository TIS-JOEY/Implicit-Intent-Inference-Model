# App2Vec_Python
This interface allows you to easily build the App2Vec model and other related advanced models(including ANN, Affinity Propagation).

## App2Vec
App2Vec is an unsupervised learning method to embed words into a dense vector space. In this dense vector space, semantically and syntactically related words are close to each other. App2Vec uses a shallow neural network that is trained to learn the distribution representation of words. In basic, Word2Vec is a single layer neural network with one hidden layer. Both input and output are represented as the One-Hot encoding. The learned vector of words is stored as input-hidden layer weight matrix.

There are two modes of Word2Vec model: (1) Continuous Bags of Word (CBOW) and (2) Continuous Skip-Gram. 
### CBOW
CBOW model is given the context words to predict the center word in the window. In order to represent the vector of context words, CBOW averages or sums the vector of each context word. Given context words of the current words, the objective of CBOW is to maximize the log probability of current word.

### Continuous Skip-Gram
Contrary to CBOW, Continuous Skip Gram model is given the current word to predict various individual context words in the window. Given the current word, the goal of Continuous Skip-Gram model is to maximize the log probability of context words.

The figure of Continuous Skip-Gram mode is shown as below.
![image](image/skip-gram.png)

In this interface, we use gensim library to achieve it.

## Training Data
App2Vec treats each app as a unit. And we use daily app usage data as our training data.
Of course, it's impossible to train the raw data directly.
So we provide the below function：

### parameter
`raw_file_path` = The storage location of your raw training data (Currently, we only support the csv file).

`app2vec_model_path` = The storage location of App2Vec model.

Raw data should be like...
Each row has a app sequence which contains several apps.

| app sequence1 |
| :--- |
| app sequence2 |
| app sequence3 |
| app sequence4 |
| app sequence5 |
| app sequence6 |
| ... |


### Usage
```text
import app2vec.App2Vec

app2vec = App2Vec()
app2vec.csv2training_data(raw_file_path = '/Users/apple/Documents/raw_data.csv')
app2vec.training_App2Vec(model_path = '/Users/apple/Documents/app2vec.model')
```
In this case, we can get the app2vec which name is app2vec.model.

# ANN
The objective of the nearest neighbor search is to find objects similar to the query point from a collection of objects. However, the processing cost is very high when the nearest neighbor search is applied to a high-dimensional data. For this reason, Approximate Nearest Neighbor(ANN) search is proposed to tackle this problem. ANN reduces the cost of processing greatly by sacrificing a little accuracy but get similar results to nearest neighbor search. ANN can be roughly categorized into two groups: Data Structure-based, Hash-based.

In this interface, we use AnnoyIndex to achieve it. AnnoyIndex is a hash-based ANN.

### parameter
`dim` = the Dimension of App2Vec.

`num_tree` = The number of trees of your ANN forest. More tress more accurate.

`ann_model_path` = The storage path of ANN model.

### Usage
```text
import app2vec.App2Vec

app2vec = App2Vec()
app2vec.csv2training_data(raw_file_path = '/Users/apple/Documents/raw_data.csv')
app2vec.training_App2Vec(model_path = '/Users/apple/Documents/app2vec.model')
app2vec.ANN(dim = 64,num_tree = 10000,app2vec_model_path = '/Users/apple/Documents/app2vec.model',ann_model_path = '/Users/apple/Documents/ANN.model')
```








