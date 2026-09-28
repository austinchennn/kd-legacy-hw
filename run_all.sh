#!/bin/sh
# Full pipeline used for the README numbers (Apple M4 Max, MPS for training).
set -e
PY=${PY:-python}
$PY train.py --mode teacher --epochs 100
for s in 0 1 2; do
  $PY train.py --mode scratch --epochs 40 --seed $s
  $PY train.py --mode kd --epochs 40 --seed $s
done
$PY bench.py
$PY plot.py
