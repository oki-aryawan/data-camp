from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd


data = pd.read_csv('boston.csv')
X = data.drop('MEDV', axis=1).values
y = data['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
print(reg_all.score(X_test,y_test))