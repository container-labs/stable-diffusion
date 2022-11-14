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
    -x=*|--style=*)
      STYLE="${i#*=}"
      shift # past argument=value
      ;;
    -p=*|--phrase=*)
      PHRASE="${i#*=}"
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
    -n=*|--number=*)
      NUM_IMAGES="${i#*=}"
      shift # past argument=value
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
echo "Style      = ${STYLE}"
echo "Phrase      = ${PHRASE}"
echo "Output Dir  = ${OUTPUT_DIR}"
echo "Max Steps   = ${MAX_STEPS}"

eval "$(conda shell.bash hook)"
conda activate training
mkdir -p ${OUTPUT_DIR}

python main.py \
  --pretrained_model_name_or_path=${MODEL_NAME} \
  --style=${STYLE} \
  --phrase="${PHRASE}" \
  --num_images=${NUM_IMAGES} \
  --output_dir=${OUTPUT_DIR} \
  --max_steps=${MAX_STEPS}
