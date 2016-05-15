# Import the iris dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = .5)

def measure(algo, outputLabel):
    classifier = algo()
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    print(outputLabel + " Accuracy: " + str(metrics.accuracy_score(y_test, predictions)))

from sklearn import metrics, tree, neighbors
algos = { 'Tree': tree.DecisionTreeClassifier,
          'K-Nearest Neighbor': neighbors.KNeighborsClassifier }

for outputLabel, algo in algos.items():
    measure(algo, outputLabel)
