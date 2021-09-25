import subprocess
import sys
import getopt

w = 23
h = 23

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
with open('annotations_to_vec.sh', 'r') as f:
    for line in f:
        count += int(line.split()[1])


subprocess.Popen("opencv_createsamples -vec trees.vec -info annotations.txt -num {} -w {} -h {}".format(count, w, h))
