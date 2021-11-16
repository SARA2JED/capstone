
#%%
import math
from gensim.corpora import dictionary
from nltk.tokenize import word_tokenize 
import pandas as pd
import spacy


#%% load data 
song_data = pd.read_csv("clean_file.csv")
df = pd.DataFrame(song_data)

# %% test 
df['processed_text'].iloc[0]
# %%
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import dict_from_corpus, simple_preprocess
from gensim.models import CoherenceModel 
#%%
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(df['processed_text'].iloc[:20]))

data_words
# %%
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])
 
 
 
# %%
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
nlp = spacy.load("en_core_web_sm")
#%%
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

#%%
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
#%%
# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

#%%
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
#%%
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
#%%
print(data_lemmatized[:1])
# %%
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])
# %% word assigned to 0 
id2word[0]
# %% print words and thier assigned number 
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[9:10]]


# %% topic mod;ing 
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
# %%
# Print the Keyword in the 20 topics
print(lda_model.print_topics())
doc_lda = lda_model[corpus]
