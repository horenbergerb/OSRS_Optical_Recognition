import os
import sys
import getopt

w = 20
h = 20

try:
    opts, args = getopt.getopt(sys.argv[1:], "hw")
except getopt.GetoptError:
    print("Error while getting arguments")

for opt, arg in opts:
    if opt == '-h':
        h = int(arg)
    if opt == '-w':
        w = int(arg)
count = 0
with open('annotations.txt', 'r') as f:
    for line in f:
        count += int(line.split()[1])

with open('bg.txt', 'w') as f:
    neg_dir = 'negatives/processed/'
    for cur_file in os.listdir(neg_dir):
        f.write(neg_dir+cur_file+'\n')
        
print('Positives count: {}'.format(count))
os.system("opencv_createsamples -vec annotations.vec -info annotations.txt -num {} -w {} -h {}".format(count, w, h))
