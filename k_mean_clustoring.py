#%%K-Means Clustering with Python

#%%
import pandas as pd 

#%%
#%% load data 
song_data = pd.read_csv("clean_file.csv")
df = pd.DataFrame(song_data)
df2 = df.iloc[:1000]
#%%
df.head(5)
# %%
# Step 3: Data preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

#%%
documents = df2['processed_text'].values.astype("U")

vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(documents)
#%%
k = 20
model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
model.fit(features)

#%%
df2['cluster'] = model.labels_

df2.head(5)

#%% output the result to a text file.

clusters = df2.groupby('cluster')    

#%%
for cluster in clusters.groups:
    f = open('clusters/cluster'+str(cluster)+ '.csv', 'w') # create csv file
    data = clusters.get_group(cluster)[['processed_text']] # get title and overview columns
    f.write(data.to_csv()) # set index to id
    f.close()

print("Cluster centroids: \n")
#%%
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(k):
    print("Cluster %d:" % i)
    for j in order_centroids[i, :10]: #print out 10 feature terms of each cluster
        print (' %s' % terms[j])
    print('------------')
# %%
