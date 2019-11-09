# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#mengimport library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Posisi_gaji.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#membuat model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

#memprediksi hasil model
y_pred = regressor.predict([[6.5]])

#visualisasi hasil model
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color ='blue')
plt.title('sesuai atau tidak(decision tee)')
plt.xlabel('Level Posisi')
plt.ylabel("Gaji")
plt.show()

