#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate total_segmentator

export TOTALSEG_LICENSE_NUMBER="xxx"
export TOTALSEG_WEIGHTS_PATH="model/"

rm -rf output/*
rm -rf log/*

python segmentation.py > log/segmentation.log
python class_assignement.py