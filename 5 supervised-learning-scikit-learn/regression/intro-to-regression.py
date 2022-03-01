import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv('boston.csv')
print(data.head())
reg = LinearRegression()


X = data.drop('MEDV', axis=1).values
y = data['MEDV'].values

X_room = X[:, 5]

y = y.reshape(-1,1)
X_room = X_room.reshape(-1,1)


reg.fit(X_room, y)
prediction_space = np.linspace(min(X_room), max(X_room)).reshape(-1,1)
plt.scatter(X_room, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space), color='black', linewidth=3)

plt.show()