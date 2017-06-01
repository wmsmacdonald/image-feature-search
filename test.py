from search import search_file
import os
import sys
from functools import partial as p

query_dir = './public/always_sunny_sample_frames/cropped/'

index_file = './indexes/full_index.p'

files = sorted(os.listdir(query_dir))

file_paths = list(map(p(os.path.join, query_dir), files))

for f in file_paths:
    if not os.path.isfile(f):
        raise IOError('Cannot open file %s' % f)

incorrect = 0

for file, file_path in list(zip(files, file_paths)):
    results = search_file(file_path, index_file)
    print(file, results)
    try:
        match_position = next(i for i, (matched_file, _) in enumerate(results) if matched_file == file)
        if match_position > 0:
            #print('file %s matches at results position %d' % (file_path, match_position))
            incorrect += 1
    except StopIteration:
        #print('file %s has no match' % file_path)
        incorrect += 1

print('%d errors out of %d' % (incorrect, len(files)))
