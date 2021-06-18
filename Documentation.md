# Documentation 

## Data Retrieval

For every `pld` in the training set I retrieved all positive emotions, negative emotions, hashtags,
mentioned accounts and mentioned entities available in the TweetsCOV19 knowledge graph via its sparql 
endpoint. The result is a raw dataset, a dictionary that is composed of five lists per `pld`.    

## Data Processing 

For every `pld` entry in the raw data set I aggregate the positive and negative emotions by 
the computing average, the standard deviation and the amount of emotions. 

The text lists are processed by tokenizing them before using `Fasttext` to get a word embedding 
and sum up the embedding for every word in the word lists.   

## Model

I chose a two lane neural networks that processes the emotions and embeddings in two different
smaller networks before being concatenated and fed to a combined network, and the evaluated by 
a softmax layer. 

The network is based the `pytorch` framework and trained with their ADAM optimizer.  





