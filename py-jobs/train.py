import argparse
import os
import time

from google.cloud import aiplatform

parser = argparse.ArgumentParser()
JOB_NAME = "job-{}".format(int(time.time()))
TRAIN_IMAGE = "us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/training:latest"
MACHINE_TYPE_TRAINING = "n1-highmem-16"
ACCELERATOR_TYPE_TRAINING = "NVIDIA_TESLA_P100"
# Training doesn't use more than 1 GPU with our config
ACCELERATOR_COUNT = 1

# learning rates
# "low" 2e-6
# "high" 5e-6
# initial - 5e-4

CMDARGS = [
    # TODO: upgrade this to use the new stable diffusion model
    "--model=CompVis/stable-diffusion-v1-4",
    "--data=/gcs/md-ml/training-data",
    f"--output=/gcs/md-ml/{JOB_NAME}",
    # Higher training step values will lead to a more accurate representation of the concept
    "--steps=2000",
    "--phrase=epaderod",
    "--token=artwork",
    "--repeat=200",
    f"--batch={ACCELERATOR_COUNT}",
    "--learning=5.0e-06",
    "--kind=style",
    #  seed will change the 'randomness' the diffusion model is using to construct the sample images to calculate the loss
    # TODO: expose seed as a hyperparameter to train
    #  change the train_batch_size if we are on a GPU with more than ~16GB of VRAM
    # TODO: explore batch size
]

aiplatform.init(project=os.getenv('GOOGLE_CLOUD_PROJECT'), location=os.getenv('REGION'), staging_bucket=os.getenv('GCS_BUCKET'))

job = aiplatform.CustomContainerTrainingJob(
        display_name=JOB_NAME,
        container_uri=TRAIN_IMAGE,
        command=["./train.sh"],
        model_serving_container_image_uri=TRAIN_IMAGE
    )

# https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomContainerTrainingJob#google_cloud_aiplatform_CustomContainerTrainingJob_run
job.run(
    model_display_name="hello_world",
    args=CMDARGS,
    replica_count=1,
    machine_type=MACHINE_TYPE_TRAINING,
    accelerator_type=ACCELERATOR_TYPE_TRAINING,
    accelerator_count=ACCELERATOR_COUNT,
    environment_variables={
      'HUGGINGFACE_TOKEN': os.getenv('HUGGINGFACE_TOKEN')
    },
)
