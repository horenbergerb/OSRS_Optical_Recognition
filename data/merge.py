import os
import sys

if len(sys.argv) != 3:
    print('Please provide two directories to merge when running this script.')
    print('Ex) python merge.py "images/cow\\ calves/" "images/vow\\ valves/"')
    exit


label_count = max(
    [int(x[:-4]) for x in os.listdir(sys.argv[1])])
for cur_file in os.listdir(sys.argv[2]):
    label_count += 1
    os.replace(
        sys.argv[2] + cur_file,
        sys.argv[1] + str(label_count) + '.png')
