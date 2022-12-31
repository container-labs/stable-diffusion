#!/bin/bash

# https://cloud.google.com/vertex-ai/docs/training/code-requirements#fuse
# "gs://md-ml"
# /gcs/md-ml/model_out

set +ex

# conda init bash
eval "$(conda shell.bash hook)"
conda activate training

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
    -l=*|--learning=*)
      LEARNING_RATE="${i#*=}"
      shift # past argument=value
      ;;
    -m=*|--mixed=*)
      MIXED_PRECISION="${i#*=}"
      shift # past argument=value
      ;;
    -k=*|--kind=*)
      LEARNABLE_PROP="${i#*=}"
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

mkdir -p ${OUTPUT_DIR}
echo "Model Name      = ${MODEL_NAME}" >> ${OUTPUT_DIR}/training.metadata
echo "Data Dir        = ${DATA_DIR}" >> ${OUTPUT_DIR}/training.metadata
echo "Learnable prop  = ${LEARNABLE_PROP}" >> ${OUTPUT_DIR}/training.metadata
echo "Output Dir      = ${OUTPUT_DIR}" >> ${OUTPUT_DIR}/training.metadata
echo "Max Steps       = ${MAX_STEPS}" >> ${OUTPUT_DIR}/training.metadata
echo "Phrase          = ${PHRASE}" >> ${OUTPUT_DIR}/training.metadata
echo "Phrase token    = ${PHRASE_TOKEN}" >> ${OUTPUT_DIR}/training.metadata
echo "Learning rate   = ${LEARNING_RATE}" >> ${OUTPUT_DIR}/training.metadata
echo "Mixed precision = ${MIXED_PRECISION}" >> ${OUTPUT_DIR}/training.metadata
echo "Repeat trainng  = ${REPEAT_TRAINING_COUNT}" >> ${OUTPUT_DIR}/training.metadata
echo "Batch size      = ${BATCH_SIZE}" >> ${OUTPUT_DIR}/training.metadata
cat ${OUTPUT_DIR}/training.metadata

python textual_inversion.py \
  --pretrained_model_name_or_path=${MODEL_NAME} \
  --train_data_dir=${DATA_DIR} \
  --learnable_property=${LEARNABLE_PROP} \
  --repeats=${REPEAT_TRAINING_COUNT} \
  --placeholder_token=${PHRASE} \
  --initializer_token=${PHRASE_TOKEN} \
  --resolution=512 \
  --train_batch_size=${BATCH_SIZE} \
  --max_train_steps=${MAX_STEPS} \
  --learning_rate=${LEARNING_RATE} \
  --scale_lr \
  --mixed_precision=${MIXED_PRECISION} \
  --output_dir=${OUTPUT_DIR}


