import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Wisconsin Breast Cancer dataset.csv")
df = data.drop(["id"], axis = 1)
df["diagnosis"] = df["diagnosis"].map({'M':1, 'B':0})
x = df.iloc[:,1:]

kmeans = KMeans(n_clusters = 2,random_state = 42, init = "k-means++")
kmeans.fit(x)
pred = kmeans.predict(x)
print(x.iloc[:,0])
print(x.iloc[:,1])

plt.figure(figsize = (13,10))
plt.scatter(x.iloc[:,0], x.iloc[:,1], c = pred, cmap = 'viridis', marker = 'o')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 100, c = 'red', marker ='^')
plt.xlabel("radius mean")
plt.ylabel("texture mean")
plt.show()
