#!/bin/bash

files=`cat path.txt`
mkdir path_res_files

count=0
for file in $files; do  
    cat per_structures/${file}.res >> path.res
    printf "\nEND\n\n" >> path.res

    cppp=`printf "%03i" "$count"`
    cp per_structures/${file}.res path_res_files/${cppp}.res 
    count=`echo "$count + 1" | bc`
done
