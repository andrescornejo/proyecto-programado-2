#!/bin/bash

#Desctiption: I'm using python trace to give me a file with the amount of times each block of code is executed. This script deletes everything but the numbers, using that file as an input.
#Example of python trace usage: python -m trace -c kmeans_test.py
#To sum script output, pipe like this "filter_cover.sh $file | paste -sd+ | bc"

if [[ -f "$1" ]]; then
  file="$1"
  cat "$file" | sed "s/[[:space:]]//g;s/:.*//g;/^[[:space:]]*$/d" | sed -n '/^[0-9]/p'
else
  echo "No file input."
fi

# Example usage: 
# python -m trace -c kmeans_tester.py && ./filter_cover.sh kmeans.cover | paste -sd+ | bc
