
import numpy as np
import pickle
import sys

descriptorsFile = open('/home/bill/Downloads/image.descriptors', 'rb')

flat = np.fromfile(descriptorsFile, dtype=np.uint8)

descriptors = flat.reshape((500, 32))

database = [('image.descriptors', descriptors)]

pickle.dump(database, open('./indexes/js_index.p', 'wb'))
