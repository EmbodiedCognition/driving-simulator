#!/bin/bash

N=${1:-10}

SCRIPT=$(mktemp)

cat <<EOF > $SCRIPT
i=\$(echo \$1 | cut -d' ' -f1)
fs=\$(echo \$1 | cut -d' ' -f2)
st=\$(echo \$1 | cut -d' ' -f3)
ss=\$(echo \$1 | cut -d' ' -f4)
lt=\$(echo \$1 | cut -d' ' -f5)
ls=\$(echo \$1 | cut -d' ' -f6)
python main.py \
  --follow-threshold 1 \
  --follow-step \$fs \
  --speed-threshold \$st \
  --speed-step \$ss \
  --lane-threshold \$lt \
  --lane-step \$ls \
  > data/follow-1-\$fs-speed-\$st-\$ss-lane-\$lt-\$ls.\$i.log
echo "follow 1,\$fs speed \$st,\$ss lane \$lt,\$ls: \$i"
EOF

chmod +x $SCRIPT

mkdir -p data

# use xargs to run as many jobs in parallel as we have cores on this machine.
for i in $(seq -w $N); do
for fs in 0.05 0.1 0.2 0.5; do
for st in 1 2 5; do
for ss in 0.05 0.1 0.2 0.5; do
for lt in 2 5 10; do
for ls in 0.05 0.1 0.2 0.5; do
echo -ne "$i $fs $st $ss $lt $ls \0 "
done
done
done
done
done
done | xargs -P$(( $(nproc) - 1 )) -L1 -0 $SCRIPT

rm -f $SCRIPT
