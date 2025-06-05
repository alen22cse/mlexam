import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def gaussian_kernel(x,x_query,tau):
    return np.exp(-(x - x_query)**2/ (2*tau **2))

def locally_weighted_regression(x,y,x_query,tau):
    n_points = len(x)
    x_b = np.c_[np.ones(len(x)),x]
    x_query_b = np.array([1,x_query])
    w = np.diag(gaussian_kernel(x,x_query,tau))
    theta = np.linalg.inv(x_b.T @ w @ x_b)@ x_b.T @ w @ y
    return x_query_b @ theta

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,2,4,3.5,5,6,7,6.5,8])
x_query = np.linspace(1,10,100)
tau = 1.0
y_lwr = np.array([locally_weighted_regression(x,y,q,tau) for q in x_query])

plt.figure(figsize=(10,8))
plt.scatter(x,y,color = "blue", label = "Data Points")
plt.plot(x_query, y_lwr, color = "red", label = "Locally Weighted Regression")
plt.title("Locally Weighted Regression")
plt.xlabel("X")
plt.legend()
plt.show()
