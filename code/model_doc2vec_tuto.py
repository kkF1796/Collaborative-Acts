from gensim.models import doc2vec
#from gensim.test.utils import common_texts    # Existence de set de données déjà préparés
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Load data

doc = ["This is a sentence", "This is another sentence"]

# Il faut créer des sets pour cette méthode
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(doc)]#enumerate(common_texts)]
print(documents)

# création des models
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# Train model (set min_count = 1, if you want the model to work with the provided example data set)

#model = doc2vec.Doc2Vec(documents, size = 100, window = 300, min_count = 1, workers = 4)

# Get the vectors
print(model)
print(model.docvecs[0])
model.docvecs[1]

vec1 = model.infer_vector(doc[0].split())
vec2 = model.infer_vector(doc[1].split())
print(vec1)

from scipy import spatial

print(spatial.distance.cosine(vec1, vec2))
test=["I am not happy"]
print(model.n_similarity(doc[0],doc[1]))
print(model.n_similarity(doc[0],test.split(" ")))


model.save('model.bin')
new_model = Doc2Vec.load('model.bin') # load model

print(new_model[0])

print(len(new_model.docvecs))

""""
# comment continuer d'entrainer un model déjà existant ?
add=["I am still not happy", "It is working well"]
n=len(model.docvecs)
adds = [TaggedDocument(add, [i+n]) for i, add in enumerate(add)]
print(adds)

"""

"""
vect=new_model.infer_vector(["I am still not happy", "It is working well"])
print(vect)
#new_model.build_vocab(adds, update=True)   #new_model.update_vocab(adds)
print(len(model.docvecs))

#new_model.train(vect, total_examples=len(vect), epochs=model.iter)
#new_model.train(more_sentences, total_examples=len(more_sentences) , epochs=model.epochs)
#print("\n",model.docvecs[2])
"""

#new_sentence = "I opened a new mailbox".split(" ")  
#model.docvecs.most_similar(positive=[model.infer_vector(new_sentence)],topn=5)
