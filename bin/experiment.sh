#!/bin/bash

N=${1:-10}

SCRIPT=$(mktemp)

cat <<EOF > $SCRIPT
#!/bin/bash

i=\$(echo \$1 | cut -d' ' -f1)
ft=\$(echo \$1 | cut -d' ' -f2)
fs=\$(echo \$1 | cut -d' ' -f3)
fa=\$(echo \$1 | cut -d' ' -f4)
st=\$(echo \$1 | cut -d' ' -f5)
ss=\$(echo \$1 | cut -d' ' -f6)
sa=\$(echo \$1 | cut -d' ' -f7)
lt=\$(echo \$1 | cut -d' ' -f8)
ls=\$(echo \$1 | cut -d' ' -f9)
la=\$(echo \$1 | cut -d' ' -f10)

if [ -z "\$ft" ]; then exit; fi

nice python main.py \
  --follow-threshold \$ft \
  --follow-step \$fs \
  --follow-accrual \$fa \
  --speed-threshold \$st \
  --speed-step \$ss \
  --speed-accrual \$sa \
  --lane-threshold \$lt \
  --lane-step \$ls \
  --lane-accrual \$la \
  > data/follow-\$ft-\$fs-\$fa-speed-\$st-\$ss-\$sa-lane-\$lt-\$ls-\$la.\$i.log

echo "follow \$ft,\$fs,\$fa speed \$st,\$ss,\$sa lane \$lt,\$ls,\$la: \$i"
EOF

chmod +x $SCRIPT

mkdir -p data

# use xargs to run as many jobs in parallel as we have cores on this machine.
for i in $(seq -w $N); do
for ft in 0.97220; do
for fs in 0.00287; do
for fa in 0.01554; do
for st in 3.31184 2.31014; do
for ss in 0.05098 0.10605; do
for sa in 0.04207; do
for lt in 10; do
for ls in 0; do
for la in 0.1; do
echo -ne "$i $ft $fs $fa $st $ss $sa $lt $ls $la\0 "
done; done; done; done; done; done; done; done; done; done | xargs -P$(nproc) -L1 -0 $SCRIPT

rm -f $SCRIPT
