#!/bin/bash

mkdir -p data

for t in 0.1 0.2 0.5 1 2 5 10
do
    for s in 0.05 0.10 0.20
    do
        for i in $(seq -w 0 10)
        do
            echo "$t $s $i"
            python main.py \
                --follow-threshold 1 \
                --follow-step 0.1 \
                --speed-threshold $t \
                --speed-step $s \
                > data/follow-1-0.1-speed-$t-$s.$i.log
        done
    done
done
