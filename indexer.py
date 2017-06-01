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
#files = ['frame003-004.jpg']

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

print('num_descriptors %d' % len(all_descriptors))

#with open('./indexes/full_index', 'wb') as f:
#    print(f.write(all_descriptors.tobytes()))

#with open('./indexes/values', 'w') as f:
#    f.write(str(partitions))

database = (all_descriptors, partitions)

pickle.dump(database, open('./indexes/full_index.p', 'wb'))

