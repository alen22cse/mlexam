import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
values = np.random.rand(100)

x_train = values[:50].reshape(-1,1)
y_train = np.array(["Class 1" if x <= 0.5 else "Class 2" for x in values[:50]])
x_test = values[50:].reshape(-1,1)
y_test = np.array(["Class 1" if x <= 0.5 else "Class 2" for x in values[50:]])

k_values = [1,2,3,4,5,20,30]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy for {k} : {accuracy*100:.2f}%")

    print(predictions)
