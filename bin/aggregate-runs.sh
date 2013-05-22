#!/bin/bash

for p in $(find data -type f | grep log$ | sort | sed "s|\.[0-9]*\.log$||" | uniq)
do python analysis/aggregate-runs.py $p
done
