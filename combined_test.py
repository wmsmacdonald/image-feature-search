from search import search_file
import os
import sys
from functools import partial as p, reduce
from get_time import get_time
import json

query_dir = './public/always_sunny_sample_frames/combined_cropped/'

index_file = './indexes/full_index.p'


with open('./frames_to_thumbs.json', 'r') as f:
    frames_to_thumbs = json.load(f)

with open('./public/always_sunny_sample_frames/medq_keyframes.txt', 'r') as f:
    thumb_times = list(map(int, map(float, f.readlines())))

#files = sorted(os.listdir(query_dir))[0:10]
files = ['frame001-002.jpg']

file_paths = list(map(p(os.path.join, query_dir), files))

for f in file_paths:
    if not os.path.isfile(f):
        raise IOError('Cannot open file %s' % f)

for file, file_path in zip(files, file_paths):

    results = search_file(file_path, index_file)
    print(file, results)


