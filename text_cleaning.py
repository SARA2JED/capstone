#%% imports 
import pandas as pd
import spacy

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import DefaultTagger
from nltk.corpus import treebank

#%% Load Text Cleaning Pkgs
import neattext.functions as nfx

#
from tqdm import tqdm
tqdm.pandas()

#%%  installing the nltk python package
nltk.download()
#%% import data 
song_data = pd.read_csv("data/csv_formated_files/lyrics-data.csv")
df = pd.DataFrame(song_data, columns =[ 'SName', 'Lyric' , 'Idiom'])

# remove non english songs from the datset 
df = df.drop(df[df.Idiom != 'ENGLISH'].index)
# drop Idiom column 
df = df.drop(columns="Idiom")

#%%
english_stop_words = set(stopwords.words("english"))

# Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
df['lyrics_without_stopwords'] = df['Lyric'].apply(lambda x: ' '.join([word for word in x.split() if word not in (english_stop_words)]))

#%%
# Data Cleaning
dir(nfx)
# remove punctuations
df['Clean_Text'] = df['lyrics_without_stopwords'].apply(nfx.remove_punctuations)
# remove hashtags
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_hashtags)
# remove numbers
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_numbers)
# remove characters
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_special_characters)

#%%
df.head(5)
#%% Stemming and Lemmatization

stemmer = PorterStemmer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
nltk.download('wordnet')

#%%(takes 30 min )
df['lyrics_stem/lem'] = df['Clean_Text'].str.split().apply(lambda x: [stemmer.stem(y) for y in x]).apply(lambda x: [lemmatizer.lemmatize(y) for y in x])

#%%
df.head(5)

#%% save new clean data 
df.to_csv('clean_file.csv', index=False)


# %%
df['processed_text'] = df['lyrics_stem/lem'].apply(' '.join)

#%%
df.head(5)

#%% save new clean data 
df.to_csv('clean_file.csv', index=False)



#%%
emoji_data = pd.read_csv("data/csv_formated_files/capstone-emoji-labeling.csv")
emoji_df = pd.DataFrame(emoji_data)

#%%
emoji_df.columns

# %%
emoji_df.drop(['Score', 'Timestamp'], axis='columns', inplace=True)

# %%
emoji_df.columns

# %%

str = ' '.join(emoji_df['heart'].iloc[1:])
# %%
emoji_df['cake'].to_string
# %%
