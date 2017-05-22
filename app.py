from flask import Flask, request
from flask_cors import CORS
import numpy as np
import pickle
import os
import search as feature_search
from partitioned import Partitioned

app = Flask(__name__)
CORS(app)

index_file = './indexes/js_index.p'


@app.route('/deleteIndex', methods=['DELETE'])
def delete():
    if os.path.isfile(index_file):
        os.remove(index_file)
    return ''


@app.route('/search', methods=['POST'])
def search():
    filename, data = list(request.files.items())[0]
    buf = data.read()
    flat = np.frombuffer(buf, dtype=np.uint8)
    descriptors = flat.reshape((-1, 16))
    try:
        results = feature_search.search(descriptors, index_file)
        print(filename, results)
        return '' if len(results) == 0 else results[0][0]
    except IOError:
        return 'No indexes'


@app.route('/index', methods=['POST'])
def index():
    filename, data = list(request.files.items())[0]
    buf = data.read()
    flat = np.frombuffer(buf, dtype=np.uint8)
    descriptors = flat.reshape((-1, 16))

    if os.path.isfile(index_file):
        database = pickle.load(open(index_file, 'rb'))
    else:
        database = (np.empty((0, 16), dtype=np.uint8), Partitioned())
    all_descriptors, partitions = database

    updated_all_descriptors = np.append(all_descriptors, descriptors, axis=0)

    partitions.add_partition(len(descriptors), filename)

    updated_database = (updated_all_descriptors, partitions)

    pickle.dump(updated_database, open(index_file, 'wb'))
    return ''


if __name__ == "__main__":
    app.run()

