from search import search_file
import os
import sys
from functools import partial as p, reduce
from get_time import get_time
import json

query_dir = './public/always_sunny_sample_frames/combined_cropped/'

index_file = './indexes/full_index.p'


def get_time_from_thumb(filename, times):
    index = int(filename[5:9]) - 1
    return times[index]

with open('./frames_to_thumbs.json', 'r') as f:
    frames_to_thumbs = json.load(f)

with open('./public/always_sunny_sample_frames/medq_keyframes.txt', 'r') as f:
    thumb_times = list(map(int, map(float, f.readlines())))

files = sorted(os.listdir(query_dir))
files = ['frame001-002.jpg']

file_paths = list(map(p(os.path.join, query_dir), files))

for f in file_paths:
    if not os.path.isfile(f):
        raise IOError('Cannot open file %s' % f)

incorrect = 0

for file, file_path in zip(files, file_paths):
    ranges = frames_to_thumbs[file]
    target_time_ranges = list(map(
        lambda r: range(thumb_times[r[0] - 1], thumb_times[r[1] - 1] + 1),
        ranges))

    results = search_file(file_path, index_file)
    times = [get_time_from_thumb(fn, thumb_times) for fn, votes in results]
    #print(results)
    time = get_time(times, list(zip(*results))[1])
    in_range = reduce(lambda result, r: result or (time in r),
                      target_time_ranges, False)
    if not in_range:
        print('file %s not correctly matched: %s, %s' % (file, time, target_time_ranges))

#print('%d errors out of %d' % (incorrect, len(files)))


