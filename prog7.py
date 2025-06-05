import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\housing.csv")
x = data[["median_income"]]
y = data["median_house_value"]

x_train,x_test,y_train,y_test = train_test_split(x, y ,test_size = 0.2, random_state = 42)

linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)
y_pred = linear_reg.predict(x_test)

poly = make_pipeline(PolynomialFeatures(degree = 3), LinearRegression())
poly.fit(x_train, y_train)
poly_pred = poly.predict(x_test)

plt.scatter(x_test,y_test, color = "blue", label ="Actual")
plt.plot(x_test,y_pred, color ="red", label = "Predicted")
plt.title("Linear Regression")
plt.xlabel("medinc")
plt.ylabel("target")
plt.legend()
plt.show()

plt.scatter(x_test,y_test, color = "blue", label ="Actual")
plt.scatter(x_test,poly_pred, color ="red", label = "Predicted")
plt.title("Polynomial Regression")
plt.xlabel("medinc")
plt.ylabel("target")
plt.legend()
plt.show()
