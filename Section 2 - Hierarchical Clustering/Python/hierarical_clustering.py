#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the data set
dataset = pd.read_csv('Mall_Customers.csv') 
x = dataset.iloc[: , 3:5].values

#using the Dendogram to find out the optimal no. of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x , method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidian Distance')
plt.show()

#training the Hirearical model on data set
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 3 , affinity = 'euclidean' , linkage = 'ward')
y_hc = cluster.fit_predict(x);

#visualsing cluster
plt.scatter(x[y_hc == 0 , 0], x[y_hc == 0 , 1] , s=100 , c= 'red' , label = 'Cluster 1')
plt.scatter(x[y_hc == 1 , 0], x[y_hc == 1 , 1] , s=100 , c= 'blue' , label = 'Cluster 2')
plt.scatter(x[y_hc == 2 , 0], x[y_hc == 2 , 1] , s=100 , c= 'green' , label = 'Cluster 3')
#plt.scatter(x[y_hc == 3 , 0], x[y_hc == 3 , 1] , s=100 , c= 'cyan' , label = 'Cluster 4')
#plt.scatter(x[y_hc == 4 , 0], x[y_hc == 4 , 1] , s=100 , c= 'magenta' , label = 'Cluster 5')              
plt.title('Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()

#Tried with both 3 and 5 (no of clusters) and 5 is better!!