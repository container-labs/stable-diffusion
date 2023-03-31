#!/bin/bash

for i in {1..100}; do
  curl --location --request POST 'localhost:6000' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "phrase": "highly detailed beebz in the style of impressionism",
        "steps": 250,
        "guidance_scale": 15.0,
        "height": 768,
        "width": 768,
        "model": "/mnt/md-ml-public/training-job-1672713198/model"
    }'
done
