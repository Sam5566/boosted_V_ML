#!/bin/bash

for filename in */*.log; do
    #echo "organize model log of "$filename
    source organize_model_log.sh $filename &
done
wait
