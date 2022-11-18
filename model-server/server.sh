#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate training

python main.py
