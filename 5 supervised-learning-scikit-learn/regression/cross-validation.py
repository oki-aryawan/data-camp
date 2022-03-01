import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import pandas as pd

data = pd.read_csv("boston.csv")
X = data.drop('MEDV', axis=1).values
y = data['MEDV'].values

reg = LinearRegression()
cv_result = cross_val_score(reg, X, y, cv=5)
print(cv_result)
print(np.mean(cv_result))