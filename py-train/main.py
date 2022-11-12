import argparse
import os
import time

from google.cloud import aiplatform

parser = argparse.ArgumentParser()
# parser.add_argument('--lr', dest='lr',
#                     default=0.01, type=float,
#                     help='Learning rate.')
# parser.add_argument('--epochs', dest='epochs',
#                     default=10, type=int,
#                     help='Number of epochs.')
# parser.add_argument('--steps', dest='steps',
#                     default=200, type=int,
#                     help='Number of steps per epoch.')

JOB_NAME = "my_job_{}".format(int(time.time()))
TRAIN_IMAGE = "us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/training:latest"
MACHINE_TYPE_TRAINING = "n1-standard-8"
# MACHINE_TYPE_TRAINING = "a2-highgpu-1g"

CMDARGS = [
    "./train-style.sh",
    # "--steps=" + str(STEPS),
    # "--distribute=" + TRAIN_STRATEGY,
]

aiplatform.init(project=os.getenv('PROJECT_ID'), location=os.getenv('REGION'), staging_bucket=os.getenv('GCS_BUCKET'))

job = aiplatform.CustomContainerTrainingJob(
    display_name=JOB_NAME,
    container_uri=TRAIN_IMAGE,
    # command=CMDARGS,
    model_serving_container_image_uri=TRAIN_IMAGE
)

job.run(
    model_display_name="hello_world",
    args=CMDARGS,
    replica_count=1,
    machine_type=MACHINE_TYPE_TRAINING,
    accelerator_type="NVIDIA_TESLA_K80",
    # accelerator_type="NVIDIA_TESLA_P100",
    accelerator_count=1,
)
