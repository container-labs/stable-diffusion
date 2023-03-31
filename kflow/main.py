import os
import time

from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.v2 import compiler, dsl

# https://github.com/kubeflow/examples/blob/master/pipelines/simple-notebook-pipeline/Simple%20Notebook%20Pipeline.ipynb

JOB_NAME = "kflow-job-{}".format(int(time.time()))
TRAIN_IMAGE = "us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/training:latest"
MODEL_SERVER_IMAGE = "us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/simple-server:latest"
MACHINE_TYPE_TRAINING = "a2-highgpu-1g"
ACCELERATOR_TYPE_TRAINING = "NVIDIA_TESLA_A100"
# Training doesn't use more than 1 GPU with our config
ACCELERATOR_COUNT = 1
MODEL_NAME="jpl_512_dopeaf_5"

# train_component = components.load_component_from_file('./components/train.yaml')
# auto_infer_component = components.load_component_from_file('./components/auto_infer.yaml')

CMDARGS = [
    # TODO: upgrade this to use the new stable diffusion model
    # "--model=CompVis/stable-diffusion-v1-4",
    "--model=runwayml/stable-diffusion-v1-5",
    "--data=/gcs/md-ml/training-data-styles/jpl-512",
    f"--output=/gcs/md-ml/{JOB_NAME}/model",
    # https://huggingface.co/blog/dreambooth#:~:text=In%20our%20experiments%2C%20a%20learning,learning%20rate%20is%20too%20high.
    "--steps=1500",
    "--phrase=dopeaf",
    "--token=poster",
    "--repeat=100",
    f"--batch={ACCELERATOR_COUNT}",
    "--learning=5.0e-06",
    "--kind=style",
]

@dsl.pipeline(
   name='stable-diffusion',
   description='A pipeline',
   pipeline_root=f"gs://md-ml/{JOB_NAME}"
)
def add_pipeline(
):
    train_task = gcc_aip.CustomContainerTrainingJobRunOp(
        display_name=JOB_NAME,
        container_uri=TRAIN_IMAGE,
        command=[
            "./train.sh",
        ],
        args=CMDARGS,
        staging_bucket="gs://md-ml",
        model_serving_container_image_uri=MODEL_SERVER_IMAGE,
        machine_type=MACHINE_TYPE_TRAINING,
        accelerator_type=ACCELERATOR_TYPE_TRAINING,
        environment_variables={
            'HUGGINGFACE_TOKEN': os.getenv('HUGGINGFACE_TOKEN')
        },
        base_output_dir=f"gs://md-ml/{JOB_NAME}",
    )

# TODO: save this somewhere
# and put this file in a pubsub topic to run piplines dynamically
compiler.Compiler().compile(add_pipeline, '/os-shared/workflow.json')

job = aiplatform.PipelineJob(display_name = 'stable-pipe',
                             template_path = '/os-shared/workflow.json',
                             job_id = JOB_NAME,
                            #  pipeline_root = PIPELINE_ROOT_PATH,
                            #  parameter_values = PIPELINE_PARAMETERS,
                            #  enable_caching = ENABLE_CACHING,
                            #  labels = LABELS,
                            #  credentials = GOOGLE_APPLICATION_CREDENTIALS,
                             project = os.getenv('PROJECT_ID'),
                            #  location = LOCATION,
                            #  failure_policy = FAILURE_POLICY,
                        )

# job.submit(service_account=SERVICE_ACCOUNT,
#            network=NETWORK)
job.submit()
