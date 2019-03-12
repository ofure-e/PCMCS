# author 'Ebhomielen Ofure'


# load data
filename = 'case_play_ansi.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

import nltk

# split into words
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

sent_text = sent_tokenize(text)
tokenized_text = [word_tokenize(sentence) for sentence in sent_text]
#print(tokenized_text)
#print('\n\n\n')

# convert to lower case
no_punctuation = []
for sentence in tokenized_text:
    tokens = [w.lower() for w in sentence]
    no_punctuation.append(tokens)

# create a list of stopwords and words to be ignored
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
remove_words = list()
remove_words.append('A')
remove_words.extend(('a', '.', ',', '?', ';', '(', ')', '\"', ))

# filter out stopwords
clean_tokens = []
for sentence in no_punctuation:
    tokens_no_stopwords = [w for w in sentence if w not in stop_words]
    tokens_no_stopwords = [w for w in tokens_no_stopwords if w not in remove_words]
    clean_tokens.append(tokens_no_stopwords)
#print(clean_tokens)
#print('\n\n\n')

#tagged = nltk.pos_tag(clean_tokens)
#print(tagged)

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

lemmatized_tokens = []
for sentence in clean_tokens:
    tokens_lem = [lem.lemmatize(lemma, pos='v') for lemma in sentence]
    lemmatized_tokens.append(tokens_lem)
#print(lemmatized_tokens)

# training word embeddings
from gensim.models import Word2Vec

model = Word2Vec(lemmatized_tokens, min_count=3)
#print(model)
print(model.wv.most_similar(positive=['atovaquone-proguanil'], topn = 10))
#words = list(model.wv.vocab)
#print(words)
#print(model['atovaquone-proguanil'])
model.save('model.bin')
new_model=Word2Vec.load('model.bin')
'''print(new_model)
print(len(new_model['atovaquone-proguanil']))
print(new_model.wv.similarity('atovaquone-proguanil', 'treated') > 0.3)'''

'''from nltk.data import find
from gensim.models import KeyedVectors
word2vec_sample = str(find('word2vecTools/types.txt'))
model = KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
print(len(model.vocab))'''


'''#fit a 2d PCA model to the vectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as pyplot
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()'''

'''from nltk.probability import FreqDist
fdist = FreqDist(tokenized_text)'''
#print(fdist)
#print(fdist.most_common(2))

#import matplotlib.pyplot as plt
#fdist.plot(77,cumulative=False)
#plt.show()




