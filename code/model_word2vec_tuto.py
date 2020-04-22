import gensim 
from gensim.models import Word2Vec 

"""------------------------------------------------------------------------------------------
    (1) https://spacy.io/models 
    (2) https://spacy.io/usage/vectors-similarity 
    (3) https://radimrehurek.com/gensim/
    (4) https://code.google.com/archive/p/word2vec/ 

	Différentes approches sont possibles pour construire un model: 
		- utiliser les sentences directement pour construire le model
		- utiliser un corpus plus volumineux pour trained le model (voir gensim)
		- load un model pré-existant
			Google (3.4 GB) & GLove () models : https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

			utiliser api gensim : https://github.com/RaRe-Technologies/gensim-data
			pre-trained model en français (2.24GB): https://github.com/Kyubyong/wordvectors

			https://fasttext.cc/docs/en/crawl-vectors.html

	=> ce sont des solutions coûteuses en mémoire

----------------------------------------------------------------------------------------------"""


# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]


# construire un model
model = Word2Vec(sentences, size=10, window=1, min_count=1, workers=1) 

# lister le vocabulaire connu du model
vocab=list(model.wv.vocab)
print('\n',vocab)

# obtenir le vecteur d'un mot
vect=model.wv['sentence']
print('\n',vect)

# obtenir les mots les plus similaires à un mot donné
similar=model.wv.most_similar('sentence')
print('\n',similar) 

# save model
model.save('model.bin') 

# load model
new_model = Word2Vec.load('model.bin') 
print('\n',new_model)

# similarité entre deux mots
print('\n similarity:',new_model.similarity('yet', 'one'))
print('\n similarity:',new_model.similarity('second', 'first'))


# comment continuer d'entrainer un model déjà existant ?
more_sentences=[['I', 'am', 'still', 'trying', 'to', 'understand'], ['is','it','possible', 'french']]
new_model.build_vocab(more_sentences, update=True)
new_model.train(more_sentences, total_examples=len(more_sentences) , epochs=model.epochs)
print(new_model)
print('\n',list(new_model.wv.vocab))

# Attention, il est aussi possible d'utiliser Phrase (phase) pour prendes en compte des bigrams

# pour obtenir le vecteur d'une sentence, utiliser la méthode average pour la moyenne des vecteurs

# utiliser cosine similarity pour la similarité
