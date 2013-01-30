#!/bin/bash

N=${1:-10}

mkdir -p data

for fs in 0.01 0.02 0.05 0.1 0.2 0.5
do
    for st in 1 2 5
    do
        for ss in 0.01 0.02 0.05 0.1 0.2 0.5
        do
            for i in $(seq -w $N)
            do
                echo "follow 1,$fs speed $st,$ss: $i"
                python main.py \
                    --follow-threshold 1 \
                    --follow-step $fs \
                    --speed-threshold $st \
                    --speed-step $ss \
                    > data/follow-1-$fs-speed-$st-$ss.$i.log
            done
        done
    done
done
