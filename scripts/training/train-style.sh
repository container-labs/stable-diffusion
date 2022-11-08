#!/bin/bash

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./data"

# import torch
# import math
# # this ensures that the current MacOS version is at least 12.3+
# print(torch.backends.mps.is_available())
# # this ensures that the current current PyTorch installation was built with MPS activated.
# print(torch.backends.mps.is_built())

python textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<bored-ape>" --initializer_token="ape" \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 --scale_lr \
  --output_dir="textual_inversion_bored_ape"
