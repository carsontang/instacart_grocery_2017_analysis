import sys

try:
    import pybind_simple as C
except ImportError as e:
    print "Couldn't import pybind_simple"
    sys.exit(1)

import numpy as np

a1 = np.array([i for i in range(10000)])
a2 = np.array([i for i in range(10000)])

result = C.add_arrays(a1, a2)
print result