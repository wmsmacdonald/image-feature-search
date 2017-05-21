from flask import Flask, request
from flask_cors import CORS, cross_origin
import numpy as np
import pickle
import os
import search as feature_search

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
        results = feature_search.search(descriptors, index_file, feature_search.flannKnn)
        print(filename, results)
        return results[0][0]
    except IOError:
        return 'No indexes'


@app.route('/index', methods=['POST'])
def index():
    filename, data = list(request.files.items())[0]
    print(filename)
    buf = data.read()
    flat = np.frombuffer(buf, dtype=np.uint8)
    descriptors = flat.reshape((-1, 16))
    if os.path.isfile(index_file):
        database = pickle.load(open(index_file, 'rb'))
    else:
        database = {}
    database[filename] = descriptors
    pickle.dump(database, open(index_file, 'wb'))
    return ''


if __name__ == "__main__":
    app.run()

