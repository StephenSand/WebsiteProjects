# data from https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score



### Cleaning dataset

# Reading the file
df = pd.read_csv('Mall_Customers.csv')

### vis 0: describe ###
pd.options.display.max_columns = None
df.describe().to_html().replace('\n','')

# Dropping NaN values
df.dropna(inplace=True)
df.drop(['CustomerID'],axis=1, inplace=True)

# Handling text features
dums = pd.get_dummies(df['Gender'])
df = pd.concat([df,dums],axis=1)
df.drop('Gender',axis=1,inplace=True)

# Visualizing feature distribution to see if a scaler is necessary
pear = sns.pairplot(df)
#no need for normalization

### vis 1: pairplot ###
pear.savefig('cluster_pairplot.png')

## Data visualization

# PCA
pca = PCA(n_components=2).fit(df)
X_pca = pca.transform(df)
df_pca = pd.DataFrame(X_pca)
df_pca.plot.scatter(x=[0],y=[1])
#plt.show()

### vis 2: pca ###
plt.savefig('pca_scatter.png')

# MDS
mds = MDS(n_components=2)
X_mds = mds.fit_transform(df)
df_mds = pd.DataFrame(X_mds)
df_mds.plot.scatter(x=[0],y=[1])
#plt.show()
#appears to be the cleanest representation

### vis 3: mds ###
plt.savefig('mds_scatter.png')

# t-SNE
tsne = TSNE(random_state=0,perplexity=30)
X_tsne = tsne.fit_transform(df)
df_tsne = pd.DataFrame(X_tsne)
df_tsne.plot.scatter(x=[0],y=[1])
#plt.show()

### vis 4: tsne ###
plt.savefig('tsne_scatter.png')


## Clustering models

# DBSCAN clustering
dbscan = DBSCAN(eps=2,min_samples=2)
predict_dbscan = dbscan.fit_predict(df)
# Evaluation
silh = silhouette_score(df,predict_dbscan)
ch = calinski_harabasz_score(df,predict_dbscan)
db = davies_bouldin_score(df,predict_dbscan)
clusters = len(np.unique(predict_dbscan))
#Guessing 5 clusters
# Visualizing predicted classes using the mds transformed data
classes = np.unique(predict_dbscan)
for x in classes:
    plt.scatter(X_mds[predict_dbscan == x , 0], X_mds[predict_dbscan == x , 1], label=x)

#plt.legend()
#plt.show()

### vis 5: dbscan cluster ###
plt.savefig('dbscan.png')


# KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=0)
predict_kmeans = kmeans.fit_predict(df)
# Evaluation
silh = silhouette_score(df,predict_kmeans)
ch = calinski_harabasz_score(df,predict_kmeans)
db = davies_bouldin_score(df,predict_kmeans)
# Visualizing predicted classes using the mds transformed data
centroids = kmeans.cluster_centers_
classes = np.unique(predict_kmeans)
for x in classes:
    plt.scatter(X_mds[predict_kmeans == x , 0], X_mds[predict_kmeans == x , 1], label=x)

#plt.legend()
#plt.show()


### vis 6: kmeans ###
plt.savefig('kmeans.png')


## Agglomerative Clustering
aggl = AgglomerativeClustering(n_clusters=5)
predict_aggl= aggl.fit_predict(df)
# Evaluation
silh = silhouette_score(df,predict_aggl)
ch = calinski_harabasz_score(df,predict_aggl)
db = davies_bouldin_score(df,predict_aggl)
# Visualizing predicted classes using the mds transformed data
classes = np.unique(predict_aggl)
for x in classes:
    plt.scatter(X_mds[predict_aggl == x , 0], X_mds[predict_aggl == x , 1], label=x)

#plt.legend()
#plt.show()

### vis 7: agglomerative clustering ###
plt.savefig('agg.png')


## Evaluation ##
#Based on instrinsic qualities, the kmeans model is actually slightly better than the agglomerative clustering model


