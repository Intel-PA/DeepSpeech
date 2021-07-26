#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi

python -u DeepSpeech.py \
        --train_files cv1/clips/train.csv \
        --dev_files cv1/clips/dev.csv \
        --test_files cv1/clips/test.csv \
        --train_batch_size 32 \
        --dev_batch_size 16 \
        --test_batch_size 16 \
        --epochs 100 \
        --checkpoint_dir checkpoints/test_run4 \
        --export_dir model/test_run4 \
        --n_hidden 512 \
        --dropout_rate 0.30 \
        --learning_rate 0.0001 \
        --train_cudnn \
        "$@"
