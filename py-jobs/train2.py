import argparse
import os
import time

from google.cloud import aiplatform

parser = argparse.ArgumentParser()
JOB_NAME = "training-job-{}".format(int(time.time()))
TRAIN_IMAGE = "us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/training:latest"
MODEL_SERVER_IMAGE = "us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/simple-server:latest"
MACHINE_TYPE_TRAINING = "a2-highgpu-1g"
ACCELERATOR_TYPE_TRAINING = "NVIDIA_TESLA_A100"
# Training doesn't use more than 1 GPU with our config
ACCELERATOR_COUNT = 1
MODEL_NAME="jpl_512_dopeaf_1-5"

# learning rates
# "low" 2e-6
# "high" 5e-6
# initial - 5e-4

CMDARGS = [
    # TODO: upgrade this to use the new stable diffusion model
    # "--model=CompVis/stable-diffusion-v1-4",
    "--model=runwayml/stable-diffusion-v1-5",
    # "--model=stabilityai/stable-diffusion-2-1",
    "--data=/gcs/md-ml/training-data-styles/jpl-512",
    f"--output=/gcs/md-ml/{JOB_NAME}/model",
    # https://huggingface.co/blog/dreambooth#:~:text=In%20our%20experiments%2C%20a%20learning,learning%20rate%20is%20too%20high.
    "--steps=1500",
    "--phrase=dopeaf",
    "--token=poster",
    "--repeat=100",
    f"--batch={ACCELERATOR_COUNT}",
    "--learning=5.0e-06",
    "--mixed_precision=bf16",
    "--resolution=512",
    "--kind=style",
]

aiplatform.init(project=os.getenv('GOOGLE_CLOUD_PROJECT'), location=os.getenv('REGION'), staging_bucket=os.getenv('GCS_BUCKET'))

job = aiplatform.CustomContainerTrainingJob(
        display_name=JOB_NAME,
        container_uri=TRAIN_IMAGE,
        command=["./train.sh"],
        model_serving_container_image_uri=MODEL_SERVER_IMAGE,
        model_serving_container_ports=[5000],
        model_serving_container_predict_route="/predict",
        model_serving_container_health_route="/health",
        # model_serving_container_command=["./serve.sh"],
    )

# https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomContainerTrainingJob#google_cloud_aiplatform_CustomContainerTrainingJob_run
job.run(
    model_display_name=MODEL_NAME,
    args=CMDARGS,
    replica_count=1,
    machine_type=MACHINE_TYPE_TRAINING,
    accelerator_type=ACCELERATOR_TYPE_TRAINING,
    accelerator_count=ACCELERATOR_COUNT,
    environment_variables={
      'HUGGINGFACE_TOKEN': os.getenv('HUGGINGFACE_TOKEN')
    },
    base_output_dir=f"gs://md-ml/{JOB_NAME}",
)
