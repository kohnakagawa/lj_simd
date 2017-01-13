#!/bin/sh

make clean
make

execs=("./comp_1x4.out" "./comp_4x1.out" "./comp_ref_4.out" "comp_1x8_v2.out" "comp_8x1_v2.out" "./comp_ref_8.out")
files=("1x4.txt" "4x1.txt" "ref_4.txt" "1x8.txt" "8x1.txt" "ref_8.txt")
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

echo "# num_iloop num_jloop 1x4 4x1 ref" > result4_$seed.txt
echo "# num_iloop num_jloop 1x8 8x1 ref" > result8_$seed.txt
paste 1x4.txt 4x1.txt ref_4.txt | awk '{print $1, $2, $3, $7, $11, $4}' >> result4_$seed.txt
paste 1x8.txt 8x1.txt ref_8.txt | awk '{print $1, $2, $3, $7, $11, $4}' >> result8_$seed.txt
rm ${files[@]}
