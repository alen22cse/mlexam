import ssl
import certifi
import urllib.request

ssl_context = ssl.create_default_context(cafile = certifi.where())
urllib.request.urlopen = lambda *args, **kwargs : urllib.request.build_opener(urllib.request.HTTPSHandler(context = ssl_context)).open(*args, **kwargs) 

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
faces = fetch_olivetti_faces()
x = faces.data
y = faces.target

x_train,x_test,y_train,y_test = train_test_split(x, y ,test_size = 0.5, random_state = 42)

gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy : {accuracy *100 :.2f} %")

n_samples = 10
plt.figure(figsize =(10,8))
for i in range(n_samples):
    ax = plt.subplot(1, n_samples, i+1)
    ax.imshow(x_test[i].reshape(64,64), cmap = 'grey')
    ax.set_title(f"Pred : {y_pred[i]}\nTrue : {y_test[i]}")
plt.show()
