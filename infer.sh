#!/usr/bin/env bash

#source activate <your-conda-env>

### Note: "Llama-2-7b-hf" for x-en, and "Llama-2-13b-hf" for en-x;
dataset=meld-st-emof
srclang=en
tgtlang=ja
task=st
seamless_size=large
data_dir=/mnt/clover/zd-yang/resources/datasets/MELD_ST/ENG_JPN/hyps_emof
llm_dir=./checkpoints/meta-llama/Llama-2-7b-hf
adapter_path=./runs/gentrans_meld-st-emof_en_ja_st_large/best_adapter.pth
#precision=bf16-mixed

python inference/gentrans.py \
        --dataset ${dataset} --srclang ${srclang} --tgtlang ${tgtlang} --task ${task} \
        --seamless_size ${seamless_size} --data_dir ${data_dir} --llm_dir ${llm_dir} \
        --adapter_path ${adapter_path}
