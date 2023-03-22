#!/bin/bash

if [ $1 == "lm_spaces" ]
then
for N in {2..6}
do
    echo "lm spaces - N =  $N "
    python3 ./demo_perf.py -N=$N
done
fi

if [ $1 == "jump_vectors" ]
then
for N in {2..6}
do
    echo "jump vectors - N =  $N "
    python3 ./demo_perf.py -N=$N -lm_spaces=0 -lm_jump_vectors=1
done
fi
