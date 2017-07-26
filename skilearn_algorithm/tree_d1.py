#!-*-encoding=utf-8-*-
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
print iris
print cross_val_score(clf, iris.data, iris.target, cv=10)