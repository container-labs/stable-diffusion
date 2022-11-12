#!/bin/bash

# "gs://md-ml"
# /gcs/md-ml/model_out

for i in "$@"; do
  case $i in
    -m=*|--model=*)
      MODEL_NAME="${i#*=}"
      shift # past argument=value
      ;;
    -o=*|--output=*)
      OUTPUT_DIR="${i#*=}"
      shift # past argument=value
      ;;
    --default)
      DEFAULT=YES
      shift # past argument with no value
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

echo "Model Name  = ${MODEL_NAME}"
echo "Output Dir  = ${OUTPUT_DIR}"
echo "DEFAULT     = ${DEFAULT}"

conda init bash
. /root/.bashrc
conda activate training

mkdir -p ${OUTPUT_DIR}

python textual_inversion.py \
  --pretrained_model_name_or_path=${MODEL_NAME} \
  --train_data_dir=./data \
  --learnable_property="object" \
  --placeholder_token="<bored-ape>" --initializer_token="ape" \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 --scale_lr \
  # https://cloud.google.com/vertex-ai/docs/training/code-requirements#fuse
  --output_dir=${OUTPUT_DIR}


