#!/bin/bash

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./data"

python textual_inversion.py \
  --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
  --train_data_dir=./data \
  --learnable_property="object" \
  --placeholder_token="<bored-ape>" --initializer_token="ape" \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 --scale_lr \
  # https://cloud.google.com/vertex-ai/docs/training/code-requirements#fuse
  --output_dir="gs://md-ml"


