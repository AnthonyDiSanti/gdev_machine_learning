# Import the iris dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = .5)

def measure(classifier, outputLabel):
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    print(outputLabel + " Accuracy: " + str(metrics.accuracy_score(y_test, predictions)))

import random;
class RandomClassifier():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            predictions.append(
                random.choice(self.y_train)
            )
        return predictions

from scipy import spatial
from collections import deque, Counter
class KNNClassifier(RandomClassifier):
    def __init__(self, k):
        self.k = k

    @staticmethod
    def distance(a, b):
        return spatial.distance.euclidean(a, b)

    def _closest(self, row, k=1):
        distances = {};
        for i in range(0, len(self.X_train)):
            dist = KNNClassifier.distance(row, self.X_train[i])
            label = self.y_train[i]
            try:
                distances[dist].append(label)
            except KeyError:
                distances[dist] = [label]
        min_dists = deque(sorted(distances.keys()))
        min_dist_labels = []
        while (len(min_dist_labels) < k):
            min_dist_labels.extend(distances[min_dists.popleft()])
        frequencies = Counter(min_dist_labels).most_common()
        min_labels = []
        max_frequency = max([c[1] for c in frequencies])
        for f in (f for f in frequencies if f[1] == max_frequency):
            min_labels.append(f[0])
        return random.choice(min_labels)
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            predictions.append(
                self._closest(row, self.k)
            )
        return predictions


class NNClassifier(KNNClassifier):
    def __init__(self):
        self.k = 1

from sklearn import metrics, tree, neighbors
algos = {
            'Tree': tree.DecisionTreeClassifier(),
            'SKLearn K-Nearest Neighbor': neighbors.KNeighborsClassifier(),
            'Random': RandomClassifier(),
            'Nearest Neighbor': NNClassifier(),
            '3-Nearest Neighbor': KNNClassifier(3),
            '7-Nearest Neighbor': KNNClassifier(7),
            '15-Nearest Neighbor': KNNClassifier(15),
        }

for outputLabel, algo in algos.items():
    measure(algo, outputLabel)
