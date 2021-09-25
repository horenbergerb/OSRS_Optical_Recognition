#!/bin/bash

opencv_traincascade -data classifier/ -vec trees.vec -bg bg.txt -numPos 200 -numNeg 500 -numStages 20 -minHitRate 0.99 -maxFalseAlarmRate 0.4 -mode ALL -w 20 -h 23 -precalcValBufSize 1536 -precalcIdxBufSize 1536 -weightTrimRate 0.95 -maxDepth 1 -maxWeakCount 100
