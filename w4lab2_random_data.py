import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

"""
Input these parameters into make_blobs:
    n_samples: The total number of points equally divided among clusters.
        Choose a number from 10-1500
    centers: The number of centers to generate, or the fixed center locations.
        Choose arrays of x,y coordinates for generating the centers. Have 1-10 centers (ex. centers=[[1,1], [2,5]])
    cluster_std: The standard deviation of the clusters. The larger the number, the further apart the clusters
        Choose a number between 0.5-1.5
"""

# generating random data
X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)
# Plot the scatter plot of the randomly generated data
plt.scatter(X1[:, 0], X1[:, 1], marker='o')
plt.show()

# Agglomerative Clustering starts
"""
The Agglomerative Clustering class will require two inputs:

    n_clusters: The number of clusters to form as well as the number of centroids to generate.
        Value will be: 4
    linkage: Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.
        Value will be: 'complete'
        Note: It is recommended you try everything with 'average' as well
"""
agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
# fit the model
agglom.fit(X1,y1)

# to show the clustering, this code goes
# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(6,4))

# These two lines of code are used to scale the data points down,
# Or else the data points will be scattered very far apart.

# Create a minimum and maximum range of X1.
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

# Get the average distance for X1.
X1 = (X1 - x_min) / (x_max - x_min)

# This loop displays all of the datapoints.
for i in range(X1.shape[0]):
    # Replace the data points with their respective cluster value 
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})

# Remove the x ticks, y ticks, x and y axis
plt.xticks([])
plt.yticks([])
#plt.axis('off')a

# Display the plot of the original data before clustering
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
# Display the plot
plt.show()

# the distance matrix
dist_matrix = distance_matrix(X1,X1) 
print(dist_matrix)

# complete-linkage class from hierarchy
Z = hierarchy.linkage(dist_matrix, 'complete')
dendro = hierarchy.dendrogram(Z)
# print(dendro)