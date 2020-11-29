#!/bin/zsh
start_idx=100
end_idx=110

for ((i=start_idx; i<=end_idx; i++)); do
    echo "$i"
    scancel $i
done
