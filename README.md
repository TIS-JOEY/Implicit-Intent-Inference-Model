# App2Vec_Python
This interface allows you to easily build the App2Vec model and other related advanced models(including ANN, Affinity Propagation)

## App2Vec
App2Vec is an unsupervised learning method to embed words into a dense vector space. In this dense vector space, semantically and syntactically related words are close to each other. App2Vec uses a shallow neural network that is trained to learn the distribution representation of words. In basic, Word2Vec is a single layer neural network with one hidden layer. Both input and output are represented as the One-Hot encoding. The learned vector of words is stored as input-hidden layer weight matrix

There are two modes of Word2Vec model: (1) Continuous Bags of Word (CBOW) and (2) Continuous Skip-Gram. 
### CBOW
CBOW model is given the context words to predict the center word in the window. In order to represent the vector of context words, CBOW averages or sums the vector of each context word. Given context words of the current words, the objective of CBOW is to maximize the log probability of current word.

### Continuous Skip-Gram
Contrary to CBOW, Continuous Skip Gram model is given the current word to predict various individual context words in the window. Given the current word, the goal of Continuous Skip-Gram model is to maximize the log probability of context words.

In this interface, we use gensim library to achieve it.

## Training Data
App2Vec treats each app as a unit. And we use daily app usage data as our training data.
Of course, it's impossible to train the raw data directly.
So we provide the below function
```text
import app2vec

```




