#!/bin/bash

i=$1

st=$2
ss=$3

ft=$4
fs=$5

nice python main.py \
  --speed-threshold $st \
  --speed-step $ss \
  --follow-threshold $ft \
  --follow-step $fs \
    > data/speed-$st-$ss-follow-$ft-$fs.$i.log

echo $i $st $ss $ft $fs
