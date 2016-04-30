from sklearn.datasets import load_iris
import numpy
from sklearn import tree
from StringIO import StringIO
from graphviz import Source

IRIS = load_iris()

# Print the structure of the data
# print IRIS.feature_names
# print IRIS.target_names

# Print the first data point
# print IRIS.data[0]
# print IRIS.target[0]

# Print all data points
# for i in range(len(IRIS.data)):
    # print "Example %d: label %s features %s" % (i, IRIS.target[i], IRIS.data[i])

test_cohort = [0, 50, 100]
train_target = numpy.delete(IRIS.target, test_cohort)
train_data = numpy.delete(IRIS.data, test_cohort, axis=0)
test_target = IRIS.target[test_cohort]
test_data = IRIS.data[test_cohort]
clf = tree.DecisionTreeClassifier().fit(train_data, train_target)

# Print the expected labels and the predicted labels
print test_target
print clf.predict(test_data)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                          feature_names=IRIS.feature_names,
                          class_names=IRIS.target_names,
                          filled=True,
                          rounded=True,
                          special_characters=True)
src = Source(dot_data.getvalue())
src.render(filename='2-tree_viz.gv')
