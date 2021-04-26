#!/bin/bash -l

#$ -P ec527

#$ -pe omp 4
#$ -l avx


module load intel/2016
cd /usr4/ec526/ptaranat/ec527/EC527_Parallel_OCR || exit
make clean
make avx
#OMP_NUM_THREADS=$NSLOTS /usr4/ec526/ptaranat/ec527/EC527_Parallel_OCR/parallel
OMP_NUM_THREADS=$NSLOTS ./avx
