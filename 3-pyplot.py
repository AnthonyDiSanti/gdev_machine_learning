import numpy
from matplotlib import pyplot

GREYHOUNDS = 500
LABS = 500

GREY_HEIGHT = 28 + 4 * numpy.random.randn(GREYHOUNDS)
LAB_HEIGHT = 24 + 4 * numpy.random.randn(LABS)

pyplot.hist([GREY_HEIGHT, LAB_HEIGHT], stacked=False, color=['r', 'b'])
pyplot.show()
