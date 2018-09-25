
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score


dataset = pd.read_csv("kc_house_data.csv")
#print(dataset.head())
X = dataset.iloc[:, 3: ].values
y = dataset.iloc[:, 2].values


from sklearn.cross_validation import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)
score = r2_score(y_test, y_pred)
print(score)


plt.scatter(dataset.bedrooms,dataset.price)
plt.title("Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()

