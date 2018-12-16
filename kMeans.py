import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load dataset
#change this line to the path of fer2018.csv
dataset = pd.read_csv(&quot;C:/Users/Rayan/Downloads/DATASET_ALL.csv&quot;)

#knn
array = dataset.values
x = array[:, 0:2304]
y = array[:, 2304]

#split the data
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42, stratify =y)

#standardizing the data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#PCA
#make instance of the model
pca = PCA(.95)
pca.fit(x_train)

#to see how many components were selected
total_no_comp = pca.n_components_
print(total_no_comp)

#update train and test dataset
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

#for kmeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

#drop emotion column
dataset.drop(&#39;emotion&#39;, axis=1,inplace=True)
#kmean with 7 cluesters
kmeans = kmeans = KMeans(n_clusters=7, max_iter=600, algorithm = &#39;auto&#39;)
kmeans.fit(x_train)

#find the optimal k
for k in range (1, 11):

  # Create a kmeans model on our data, using k clusters. random_state helps ensure that the
  # algorithm returns the same results each time.
  kmeans_model = KMeans(n_clusters=k, random_state=1).fit(x_train[:, :])

  # These are our fitted labels for clusters -- the first cluster has labe l 0, and the second has label 1.
  labels = kmeans_model.labels_

  # Sum of distances of samples to their closest cluster center
  interia = kmeans_model.inertia_
  print (&quot;k:&quot;,k, &quot; cost:&quot;, interia)

#this to plot the k
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
  km = KMeans(n_clusters=k)
  km = km.fit(x_train)
  Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, &#39;bx-&#39;)
plt.xlabel(&#39;k&#39;)
plt.ylabel(&#39;Sum_of_squared_distances&#39;)
plt.title(&#39;Elbow Method For Optimal k&#39;)
plt.show()
