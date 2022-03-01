"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import pandas as pd

data = pd.read_csv('boston.csv')
X = data.drop('MEDV', axis=1).values
y = data['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
rid_pred = ridge.predict(X_test)
rid_score = ridge.score(X_test, y_test)
print(rid_score)

"""

from sklearn.linear_model import Lasso
import pandas as pd
import matplotlib.pyplot as plt

boston = pd.read_csv('boston.csv')
X = boston.drop('MEDV', axis=1).values
y = boston['MEDV'].values
names = boston.drop('MEDV', axis=1).columns
lass = Lasso(alpha=0.1)
lasso_coef = lass.fit(X, y).coef_
_ = plt.plot(range(len(names)), lasso_coef )
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficient')
plt.show()





