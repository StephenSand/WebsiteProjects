# data from https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score



## Cleaning dataset

# Reading the file
df = pd.read_csv('Mall_Customers.csv')

## Looking at the features
#pd.options.display.max_columns = None
df.describe()

# Dropping NaN values
df.dropna(inplace=True)
df.drop(['CustomerID'],axis=1, inplace=True)


# Changing alphabetic feature to binary feature
df['Gender'] = df['Gender'].replace({'Male':False, 'Female':True})


## Pairplot - Visualizing feature distribution to see if a scaler is necessary
sns.set_theme(palette='inferno')
pear = sns.pairplot(df)
pear.savefig('cluster_pairplot.png')
plt.clf()
#no need for normalization

## Data visualization

# PCA
pca = PCA(n_components=2).fit(df)
X_pca = pca.transform(df)
df_pca = pd.DataFrame(X_pca)
df_pca.plot.scatter(x=[0],y=[1])
plt.title('PCA')
#plt.show()
plt.savefig('pca_scatter.png')
plt.clf()

# MDS
mds = MDS(n_components=2)
X_mds = mds.fit_transform(df)
df_mds = pd.DataFrame(X_mds)
df_mds.plot.scatter(x=[0],y=[1])
plt.title('MDS')
#plt.show()
plt.savefig('mds_scatter.png')
plt.clf()

# t-SNE
tsne = TSNE(random_state=0,perplexity=30)
X_tsne = tsne.fit_transform(df)
df_tsne = pd.DataFrame(X_tsne)
df_tsne.plot.scatter(x=[0],y=[1])
plt.title('tSNE')
#plt.show()
plt.savefig('tsne_scatter.png')
plt.clf()


## Clustering models

# DBSCAN clustering
dbscan = DBSCAN(eps=10,min_samples=7)
predict_dbscan = dbscan.fit_predict(df)
# Evaluation
silh = silhouette_score(df,predict_dbscan)
print(silh)
#0.08101764288512514
ch = calinski_harabasz_score(df,predict_dbscan)
print(ch)
#18.900799015324814
db = davies_bouldin_score(df,predict_dbscan)
print(db)
#1.8983395402028869
clusters = len(np.unique(predict_dbscan))
#Guessing 5 clusters
# Visualizing predicted classes using the t-sne transformed data
classes = np.unique(predict_dbscan)
for x in classes:
    plt.scatter(X_tsne[predict_dbscan == x , 0], X_tsne[predict_dbscan == x , 1], label=x)

plt.title('DBSCAN')
plt.legend(labels = ['Group 1', 'Group 2', 'Group 3','Group 4','Group 5','Group 6'], loc='upper right', ncols=2)
#plt.show()
plt.savefig('dbscan.png')
plt.clf()


# KMeans clustering
kmeans = KMeans(n_clusters=6, random_state=0)
predict_kmeans = kmeans.fit_predict(df)
# Evaluation
silh = silhouette_score(df,predict_kmeans)
print(silh)
#0.4506609653808789
ch = calinski_harabasz_score(df,predict_kmeans)
print(ch)
#166.44782249295693
db = davies_bouldin_score(df,predict_kmeans)
print(db)
#0.7520902834855221
# Visualizing predicted classes using the t-sne transformed data
centroids = kmeans.cluster_centers_
classes = np.unique(predict_kmeans)
for x in classes:
    plt.scatter(X_tsne[predict_kmeans == x , 0], X_tsne[predict_kmeans == x , 1], label=x)

plt.title('K-Means')
plt.legend(labels = ['Group 1', 'Group 2', 'Group 3','Group 4','Group 5','Group 6'], loc='upper right', ncols=2)
#plt.show()
plt.savefig('kmeans.png')
plt.clf()


## Agglomerative Clustering
aggl = AgglomerativeClustering(n_clusters=6)
predict_aggl= aggl.fit_predict(df)
# Evaluation
silh = silhouette_score(df,predict_aggl)
print(silh)
#0.4428008535928764
ch = calinski_harabasz_score(df,predict_aggl)
print(ch)
#159.3286285014588
db = davies_bouldin_score(df,predict_aggl)
print(db)
#0.7690392732314946
# Visualizing predicted classes using the t-sne transformed data
classes = np.unique(predict_aggl)
for x in classes:
    plt.scatter(X_tsne[predict_aggl == x , 0], X_tsne[predict_aggl == x , 1], label=x)

plt.title('Agglomerative Clustering')
plt.legend(labels = ['Group 1', 'Group 2', 'Group 3','Group 4','Group 5','Group 6'], loc='upper right', ncols=2)
#plt.show()
plt.savefig('agg.png')


## Evaluation ##
#Based on instrinsic qualities, the kmeans model is actually slightly better than the agglomerative clustering model



