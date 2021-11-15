# capstone
Capstone Project for Misk Data Science Immersive Course 2021
this project was created to test demonstrate my data since knowledge and nlp skills i've learned in this course


## goals of this project 
develop a mode that can analyze and vectors words so they can be used to measure similarity between song lyrics and clusters of word that later can be mapped to emojis 


## features 
- data cleaning (remove stop words / lemmatizing / vectorizing )

- clustering similar words together using k-mean clustering  

- measure similarity using sklearn libraries 

### future work to be added 
- map emojis to the clusters 
- add a recommendation system that return a song based on emojis inputted by the user

### extra work 
- topic modeling using gensim 

## Data
this data set was scraped from Vagalume by Anderson Neisse 
for this project we only used the lyrics dataset that contained : 
[songs with lyrics ](https://www.kaggle.com/neisse/scrapped-lyrics-from-6-genres?select=lyrics-data.csv)

- Sname : name for song 
- lyrics : one string containing all the lyrics in the song 
- Idiom : language of the song

### example 
| SName  | lyrics |
| ------------- | ------------- |
| More Than This  | I could feel at the time. There was no way of ...	  |
| Because The Night	Take me now  | baby, here as I am. Hold me close...	  |


## how to run the files 
run the files in the following order to get the best result from the model 
 - text cleaning 
 - k mean clustering 
 - similarity 

## Limitations & problems
this code was written for a device with limited memory space to run it , but a little tweaking and it well be able to run bigger datasets 

## Refrences : 
[topic modeling gensim](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#10removestopwordsmakebigramsandlemmatize)

[cleaning preprocessing text data by building nlp pipeline](https://towardsdatascience.com/cleaning-preprocessing-text-data-by-building-nlp-pipeline-853148add68a)

[unsupervised text clustering using natural language processing nlp](https://medium.com/@rohithramesh1991/unsupervised-text-clustering-using-natural-language-processing-nlp-1a8bc18b048d)

[Gensim word vector visualization](https://web.stanford.edu/class/cs224n/materials/Gensim%20word%20vector%20visualization.html)

[Detecting Text Similarity over Short Passages: Exploring Linguistic Feature Combinations via Machine Learning](https://aclanthology.org/W99-0625.pdf)

[Word Embeddings and Song Embeddings](https://towardsdatascience.com/lyric-based-song-recommendation-with-doc2vec-embeddings-and-spotifys-api-5a61c39f1ce2)

[What Songs Tell Us About: Text Mining with Lyrics](https://towardsdatascience.com/what-songs-tell-us-about-text-mining-with-lyrics-ca80f98b3829) 

[Topic Modeling with LSA, PSLA, LDA & lda2Vec](https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05) 

[Calculating Document Similarities using BERT, word2vec, and other models](https://towardsdatascience.com/calculating-document-similarities-using-bert-and-other-models-b2c1a29c9630)


