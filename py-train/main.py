import argparse
import os
import time

from google.cloud import aiplatform

topic_name = 'projects/{project_id}/topics/{topic}'.format(
    project_id=os.getenv('GOOGLE_CLOUD_PROJECT'),
    topic=os.getenv('GCP_TOPIC'),  # Set this to something appropriate.
)

subscription_name = 'projects/{project_id}/subscriptions/{sub}'.format(
    project_id=os.getenv('GOOGLE_CLOUD_PROJECT'),
    sub=os.getenv('GCP_SUBSCRIPTION'),  # Set this to something appropriate.
)

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

JOB_NAME = "job-{}".format(int(time.time()))
TRAIN_IMAGE = "us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/training:latest"
# MACHINE_TYPE_TRAINING = "n1-highmem-8"
# ACCELERATOR_TYPE_TRAINING = "NVIDIA_TESLA_K80"
MACHINE_TYPE_TRAINING = "n1-highmem-16"
ACCELERATOR_TYPE_TRAINING = "NVIDIA_TESLA_P100"
# Training doesn't use more than 1 GPU with our config
ACCELERATOR_COUNT = 1
# MACHINE_TYPE_TRAINING = "a2-highgpu-1g"
# ACCELERATOR_TYPE_TRAINING = "NVIDIA_TESLA_A100"
# ACCELERATOR_COUNT = 1

CMDARGS = [
    "--model=CompVis/stable-diffusion-v1-4",
    "--data=/gcs/md-ml/training-data",
    f"--output=/gcs/md-ml/{JOB_NAME}",
    # Higher training step values will lead to a more accurate representation fo the concept
    "--steps=2000",
    "--phrase=boredapestyle",
    "--token=boredape",
    "--repeat=100",
    "--batch=1",
    #  seed will change the 'randomness' the diffusion model is using to construct the sample images to calculate the loss
    # TODO: expose seed as a hyperparameter to train
    #  change the train_batch_size if we are on a GPU with more than ~16GB of VRAM
    # TODO: explore batch size
]

aiplatform.init(project=os.getenv('GOOGLE_CLOUD_PROJECT'), location=os.getenv('REGION'), staging_bucket=os.getenv('GCS_BUCKET'))

# def callback(message):
#     print(message.data)
#     message.ack()


# with pubsub_v1.SubscriberClient() as subscriber:
#     subscriber.create_subscription(
#         name=subscription_name, topic=topic_name)
#     future = subscriber.subscribe(subscription_name, callback)


job = aiplatform.CustomContainerTrainingJob(
        display_name=JOB_NAME,
        container_uri=TRAIN_IMAGE,
        command=["./train-style.sh"],
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
