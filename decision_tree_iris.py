from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion='gini', max_depth=10)
model.fit(X_train, y_train)

print("Decision Tree Accuracy:", model.score(X_test, y_test))
