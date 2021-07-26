#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi

model_name="$1"

mkdir checkpoints/logs/$model_name
mkdir checkpoints/$model_name
mkdir model/$model_name


python -u DeepSpeech.py \
        --train_files cv1/clips/train.csv \
        --dev_files cv1/clips/dev.csv \
        --test_files cv1/clips/test.csv \
        --summary_dir checkpoints/logs/$model_name \
        --train_batch_size 64 \
        --dev_batch_size 64 \
        --test_batch_size 64 \
        --n_hidden 512 \
        --epochs 100 \
        --checkpoint_dir checkpoints/$model_name\
        --export_dir model/$model_name\
        --dropout_rate 0.3 \
        --learning_rate 0.0001 \
        --train_cudnn \
        --reduce_lr_on_plateau \
        "$@"
