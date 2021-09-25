#!/bin/bash

opencv_traincascade -data classifier/ -vec annotations.vec -bg bg.txt -numPos 250 -numNeg 700 -numStages 17 -minHitRate 0.99 -maxFalseAlarmRate 0.4 -mode ALL -w 20 -h 20 -precalcValBufSize 1536 -precalcIdxBufSize 1536 -weightTrimRate 0.95 -maxDepth 1 -maxWeakCount 100
