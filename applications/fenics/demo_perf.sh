#!/bin/bash

if [[ $1 == "lm_spaces" ]]
then
for N in {2..6}
do
    echo "lm spaces - N =  $N "
    # Clearing cache
    python3 ./demo_perf.py -N=$N -clear_cache=1
done
mkdir -p ./plots_perf_cache0
cp ./plots_perf/timings_lm_spaces.txt ./plots_perf_cache0/timings_lm_spaces.txt
for N in {2..6}
do
    echo "lm spaces - N =  $N "
    # Clearing cache
    python3 ./demo_perf.py -N=$N -clear_cache=1
    # Run again without clearing cache
    python3 ./demo_perf.py -N=$N -clear_cache=0
done
mkdir -p ./plots_perf_cache1
cp ./plots_perf/timings_lm_spaces.txt ./plots_perf_cache1/timings_lm_spaces.txt
fi

if [[ $1 == "jump_vectors" ]]
then
# Clearing cache
for N in {2..6}
do
    echo "jump vectors - N =  $N "
    python3 ./demo_perf.py -N=$N -lm_spaces=0 -lm_jump_vectors=1 -clear_cache=1
    mkdir -p ./plots_perf_cache0
    cp ./plots_perf/timings_jump_vectors.txt ./plots_perf_cache0/timings_jump_vectors.txt
done
