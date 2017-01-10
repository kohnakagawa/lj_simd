#!/bin/sh

exec="./aos_brute_force1x4.out"; out_file="force1x4.txt"
# exec="./aos_brute_force4x1.out"; out_file="force4x1.txt"
# exec="./aos_brute_force_ref.out"; out_file="force_ref.txt"
# exec="./aos_brute_force1x4_recless.out"; out_file="force1x4_recless.txt"
# exec="./aos_brute_force4x1_recless.out"; out_file="force4x1_recless.txt"
# exec="./aos_brute_force_ref_rectless.out"; out_file="force_ref_recless.txt"

rm $out_file
touch $out_file

for i in `seq 20 20 200`
do
    $exec $i >> $out_file
done

for i in `seq 500 500 10000`
do
    $exec $i >> $out_file
done
