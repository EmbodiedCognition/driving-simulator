#!/bin/bash

N=${1:-10}

mkdir -p data

for fs in 0.1 0.2 0.5
do
    for st in 1 2 3
    do
        for ss in 0.5
        do
            for i in $(seq -w $N)
            do
                echo "follow 1,$fs speed $st,$ss: $i"
                ( nice python main.py \
                    --follow-threshold 1 \
                    --follow-step $fs \
                    --speed-threshold $st \
                    --speed-step $ss \
                    --lane-threshold 10 \
                    --lane-step 0.001 \
                    > data/follow-1-$fs-speed-$st-$ss.$i.log & )
            done
        done
    done
done
