#!/bin/bash

# https://cloud.google.com/vertex-ai/docs/training/code-requirements#fuse
# "gs://md-ml"
# /gcs/md-ml/model_out

for i in "$@"; do
  case $i in
    -m=*|--model=*)
      MODEL_NAME="${i#*=}"
      shift # past argument=value
      ;;
    -d=*|--data=*)
      DATA_DIR="${i#*=}"
      shift # past argument=value
      ;;
    -o=*|--output=*)
      OUTPUT_DIR="${i#*=}"
      shift # past argument=value
      ;;
    -s=*|--steps=*)
      MAX_STEPS="${i#*=}"
      shift # past argument=value
      ;;
    -p=*|--phrase=*)
      PHRASE="${i#*=}"
      shift # past argument=value
      ;;
    -r=*|--repeat=*)
      REPEAT_TRAINING_COUNT="${i#*=}"
      shift # past argument=value
      ;;
    -b=*|--batch=*)
      BATCH_SIZE="${i#*=}"
      shift # past argument=value
      ;;
    -t=*|--token=*)
      PHRASE_TOKEN="${i#*=}"
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
echo "Data Dir    = ${DATA_DIR}"
echo "Output Dir  = ${OUTPUT_DIR}"
echo "DEFAULT     = ${DEFAULT}"
echo "Max Steps   = ${MAX_STEPS}"
echo "Phrase      = ${PHRASE}"
echo "Phrase token     = ${PHRASE_TOKEN}"
echo "Repeat trainng      = ${REPEAT_TRAINING_COUNT}"
echo "Batch size     = ${BATCH_SIZE}"

# conda init bash
eval "$(conda shell.bash hook)"
conda activate training

mkdir -p ${OUTPUT_DIR}

python main.py \
  --pretrained_model_name_or_path=${MODEL_NAME} \
  --output_dir=${OUTPUT_DIR} \
  --max_steps=${MAX_STEPS}
