from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
type(iris)

x = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x, y)

x_new = [[5, 3, 6, 1], [3, 5, 4, 2]]

print (knn.predict(x_new))
