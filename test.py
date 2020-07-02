import spacy
from spacy import displacy

spacy.prefer_gpu()

nlp = spacy.load("fr_core_news_sm")


text = 'Ceci est un travail intéressant.'
doc = nlp(text)
print(doc)


norm = text.lower()
print(norm)


tokens=[token for token in doc]
print(tokens)


stopwords=[token for token in tokens if not token.is_stop]
print(stopwords)


lemma=[token.lemma_ for token in tokens]
print(lemma)

punct = [token for token in tokens if not token.is_punct]
print(punct)

print(doc.vector)
