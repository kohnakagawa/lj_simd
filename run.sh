#!/bin/sh

execs=("./comp_1x4.out" "./comp_4x1.out" "./comp_ref.out")
files=("1x4.txt" "4x1.txt" "ref.txt")
Ni="100003"
seed=$RANDOM

make ${execs[@]}

for i in `seq 0 2`
do
    rm -f ${files[$i]}
    touch ${files[$i]}
    for Nj in `seq 4 1 200`
    do
        ${execs[$i]} $Ni $Nj $seed >> ${files[$i]}
    done
done

echo "# num_iloop num_jloop 1x4 4x1 ref" > result_$seed.txt
paste 1x4.txt 4x1.txt ref.txt | awk '{print $1, $2, $3, $7, $11, $4}' >> result_$seed.txt
rm ${files[@]}
