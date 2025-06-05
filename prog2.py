import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
housing_df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\housing.csv")
numeric_data = housing_df.select_dtypes(include = [np.number])

sns.heatmap(numeric_data.corr(), annot = True, cmap = "coolwarm", fmt = ".2f", linewidths = 0.5)
plt.title("Correlation matrix")
plt.show()

sns.pairplot(numeric_data, diag_kind = "kde", plot_kws = {"alpha": 0.5})
plt.suptitle("Pairplot", y = 1.04)
plt.show()
