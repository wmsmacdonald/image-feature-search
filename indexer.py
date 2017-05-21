#!/usr/bin/env python

import sys
from functools import partial as p
import os
import pickle
from get_descriptors import get_descriptors


files = sorted(os.listdir(sys.argv[1]))

file_paths = list(map(p(os.path.join, sys.argv[1]), files))

for f in file_paths:
    if not os.path.isfile(f):
        raise IOError('Cannot open file %s' % f)

descriptors_by_file = list(map(get_descriptors, file_paths))

database = [(file, kp) for file, kp in zip(files, descriptors_by_file)]

pickle.dump(database, open(sys.argv[2], 'wb'))


