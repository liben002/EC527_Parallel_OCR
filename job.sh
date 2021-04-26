#!/bin/bash -l

#$ -P ec527

#$ -pe omp 4


module load gcc
cd /usr4/ec526/ptaranat/ec527/EC527_Parallel_OCR || exit
make parallel
OMP_NUM_THREADS=$NSLOTS /usr4/ec526/ptaranat/ec527/EC527_Parallel_OCR/parallel
