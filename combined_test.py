from search import search_file
import os
import sys
from functools import partial as p

query_dir = './public/always_sunny_sample_frames/combined_cropped/'

index_file = './indexes/full_index.p'

files = sorted(os.listdir(query_dir))
#files = ['frame001-002.jpg']

file_paths = list(map(p(os.path.join, query_dir), files))

for f in file_paths:
    if not os.path.isfile(f):
        raise IOError('Cannot open file %s' % f)

incorrect = 0

for file, file_path in list(zip(files, file_paths)):
    frame1, frame2 = file[5:][:-4].split('-')
    file1 = 'frame%s.jpg' % frame1
    file2 = 'frame%s.jpg' % frame2
    results = search_file(file_path, index_file)
    if len(results) < 2 or not (results[0][0] == file1 and results[1][0] == file2 or
                                results[0][0] == file2 and results[1][0] == file1):

        print('file %s not correctly matched: %s' % (file, results))
        incorrect += 1

print('%d errors out of %d' % (incorrect, len(files)))
