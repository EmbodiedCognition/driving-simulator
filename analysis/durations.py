import itertools
import numpy as np
import operator
import sys

for f in sys.argv[1:]:
    d = np.loadtxt(f)
    for m, fs in itertools.groupby(d, key=operator.itemgetter(5)):
        print f, m, fs.next()[0], '%.2f' % (len(list(fs)) / 60.)
