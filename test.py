from search import search
import os
import sys
from functools import partial as p

query_dir = './always_sunny_sample_frames/cropped/'

index_file = './indexes/full_index.p'

files = sorted(os.listdir('./always_sunny_sample_frames/cropped/'))

file_paths = list(map(p(os.path.join, query_dir), files))

for f in file_paths:
    if not os.path.isfile(f):
        raise IOError('Cannot open file %s' % f)

for file, file_path in list(zip(files, file_paths)):
    results = search(file_path, index_file)
    match_position = next(i for i, (matched_file, _) in enumerate(results) if matched_file == file)
    if match_position > 0:
        print('file %s matches at results position %d' % (file_path, match_position))

