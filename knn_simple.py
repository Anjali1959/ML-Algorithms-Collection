import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

X = [[1,2], [2,3], [3,3], [6,5], [7,7]]
y = [0,0,0,1,1]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

new_point = [[5,5]]
prediction = model.predict(new_point)
print("KNN Prediction for [5,5]:", prediction)
