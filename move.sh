#!/bin/bash
rsync -a --no-owner --no-group --progress \
  --exclude 'build/' \
  --exclude 'install/' \
  --exclude 'log/' \
  --exclude 'data/' \
  --exclude 'src/data_analysis/' \
  --exclude 'src/ergodic/' \
  /home/nova/research/aiet/ArmBilateral/ \
  msr@192.168.18.1:/home/msr/Bilateral/
