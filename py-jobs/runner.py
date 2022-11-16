import argparse
import os

from google.cloud import aiplatform

parser = argparse.ArgumentParser()
TRAINING_JOB = "job-1668492845"
JOB_PATH = f"/gcs/md-ml/{TRAINING_JOB}/img_out"
TRAIN_IMAGE = "us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/process:latest"
# use smaller machines for inference
MACHINE_TYPE_TRAINING = "n1-highmem-8"
ACCELERATOR_TYPE_TRAINING = "NVIDIA_TESLA_K80"
ACCELERATOR_COUNT = 1

# job 1
# job-1668383425
# epaderod
# Elon Musk

# https://huggingface.co/blog/dreambooth

CMDARGS = [
    f"--model=/gcs/md-ml/{TRAINING_JOB}",
    "--style=epaderod",
    "--phrase=\"artistic portrait of Elon Musk, solid pastel background\"",
    f"--output={JOB_PATH}",
    "--steps=100",
    "--number=20",
]

aiplatform.init(project=os.getenv('GOOGLE_CLOUD_PROJECT'), location=os.getenv('REGION'), staging_bucket=os.getenv('GCS_BUCKET'))

job = aiplatform.CustomContainerTrainingJob(
        display_name=f"inference-{TRAINING_JOB}",
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
