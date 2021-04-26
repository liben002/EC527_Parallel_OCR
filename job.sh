#!/bin/bash -l

#$ -P ec527

#$ -l avx

module load gcc
cd /usr4/ec526/ptaranat/ec527/EC527_Parallel_OCR || exit
make
#OMP_NUM_THREADS=$NSLOTS /usr4/ec526/ptaranat/ec527/EC527_Parallel_OCR/parallel
./serial
