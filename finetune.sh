#!/usr/bin/env bash

source activate gentranslate

### Note: "Llama-2-7b-hf" for x-en, and "Llama-2-13b-hf" for en-x;
dataset=meld-st
srclang=en
tgtlang=de
task=st
seamless_size=large
data_dir=/mnt/clover/zd-yang/resources/datasets/MELD_ST/ENG_DEU/hyps
llm_dir=checkpoints/meta-llama/Llama-2-7b-hf

python finetune/gentrans.py \
       --dataset ${dataset} --srclang ${srclang} --tgtlang ${tgtlang} --task ${task} --d 1 \
       --seamless_size ${seamless_size} --data_dir ${data_dir} --llm_dir ${llm_dir}  \
       --lr 0.01 --num_epochs 2

