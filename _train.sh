#!/bin/bash

# Load python virtualenv
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Comment if using GPU
export CUDA_VISIBLE_DEVICES=0

# Begin experiment
python3 main.py \
--config continuous_soc.yaml
