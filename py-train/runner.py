import argparse
import os
import time

from google.cloud import aiplatform

parser = argparse.ArgumentParser()
JOB_NAME = "process-{}".format(int(time.time()))
TRAIN_IMAGE = "us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/process:latest"
MACHINE_TYPE_TRAINING = "n1-highmem-16"
ACCELERATOR_TYPE_TRAINING = "NVIDIA_TESLA_P100"
ACCELERATOR_COUNT = 1

CMDARGS = [
    "--model=/gcs/md-ml/job-1668365875",
    "--data=/gcs/md-ml/training-data",
    f"--output=/gcs/md-ml/{JOB_NAME}",
    "--steps=100",
    "--batch=1",
    "--num_images=50",
    "--style=boredape",
    "--phrase=\"Elon Musk\"",
    # TODO: explore batch size
]

aiplatform.init(project=os.getenv('GOOGLE_CLOUD_PROJECT'), location=os.getenv('REGION'), staging_bucket=os.getenv('GCS_BUCKET'))

job = aiplatform.CustomContainerTrainingJob(
        display_name=JOB_NAME,
        container_uri=TRAIN_IMAGE,
        command=["./process.sh"],
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
