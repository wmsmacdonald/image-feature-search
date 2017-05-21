
import numpy as np
from search import search

descriptorsFile = open('/home/bill/Downloads/image.descriptors', 'rb')

flat = np.fromfile(descriptorsFile, dtype=np.uint8)

descriptors = flat.reshape((500, 32))
results = search(descriptors, './indexes/js_index.p')
for r in results:
    print(r)



