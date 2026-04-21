from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0)
rf_model.fit(X_train, y_train)

print("Random Forest Accuracy:", rf_model.score(X_test, y_test))
