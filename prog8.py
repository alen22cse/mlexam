import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from  sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = load_breast_cancer()
x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(x, y ,test_size = 0.2, random_state = 42)

clf = DecisionTreeClassifier(random_state = 42)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy : {accuracy *100 :.2f} %")

new_sample = np.array([x_test[0]])
predictions = clf.predict(new_sample)

prediction_class = "Benign" if predictions == 1 else "Malignant"
print(f"Prediction class for the new sample : {prediction_class}")

plt.figure(figsize = (10,8))
plot_tree(clf, filled = True, feature_names = data.feature_names, class_names = data.target_names)
plt.title("Decision Tree Classifier ")
plt.show()
