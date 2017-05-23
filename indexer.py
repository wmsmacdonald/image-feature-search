#!/usr/bin/env python

import sys
from functools import partial as p
import os
import pickle
from get_descriptors import get_descriptors
from partitioned import Partitioned
import numpy as np
import operator as op

files = sorted(os.listdir(sys.argv[1]))

file_paths = list(map(p(os.path.join, sys.argv[1]), files))

for f in file_paths:
    if not os.path.isfile(f):
        raise IOError('Cannot open file %s' % f)


descriptors_by_file = list(map(get_descriptors, file_paths))

valid_descriptors_by_file, valid_files = zip(*[(descriptors, file)
                                               for descriptors, file
                                               in zip(descriptors_by_file, files)
                                               if descriptors is not None])
partitions = Partitioned()

for descriptors, filename in zip(valid_descriptors_by_file, valid_files):
    partitions.add_partition(len(descriptors), filename)

all_descriptors = np.concatenate(valid_descriptors_by_file, axis=0)

database = (all_descriptors, partitions)

pickle.dump(database, open(sys.argv[2], 'wb'))

