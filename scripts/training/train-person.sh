#!/bin/bash

export MODEL_NAME="runwayml/stable-diffusion-v1-5"

python textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=data-will \
  --learnable_property="object" \
  --placeholder_token="<will-beebe>" --initializer_token="beebe" \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 --scale_lr \
  --output_dir="textual_inversion_bored_beebe"


# --learning_rate=1.0e-06 --scale_lr \
