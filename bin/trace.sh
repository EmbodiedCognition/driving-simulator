#!/bin/bash

mkdir -p trace

python main.py \
    --follow-threshold 1 \
    --follow-step 0.1 \
    --speed-threshold 5 \
    --speed-step 0.1 \
    --trace TrafficPositionsSmoothed.txt.gz \
    > trace/metrics.txt

mv *.path trace
