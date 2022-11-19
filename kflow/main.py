import os
import time

from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.v2 import compiler, dsl

# https://github.com/kubeflow/examples/blob/master/pipelines/simple-notebook-pipeline/Simple%20Notebook%20Pipeline.ipynb

JOB_ID = "{}".format(int(time.time()))
TRAIN_IMAGE = "us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/training:latest"
INFER_IMAGE = "us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/process:latest"

# train_component = components.load_component_from_file('./components/train.yaml')
# auto_infer_component = components.load_component_from_file('./components/auto_infer.yaml')

@dsl.pipeline(
   name='stable-diffusion',
   description='A pipeline'
)
def calc_pipeline(
):
    train_task = gcc_aip.CustomContainerTrainingJobRunOp(
        display_name=f"training-{JOB_ID}",
        container_uri=TRAIN_IMAGE,
        command=[
            "./train.sh",
            "--model=CompVis/stable-diffusion-v1-4",
            "--data=/gcs/md-ml/training-data",
            f"--output=/gcs/md-ml/job-{JOB_ID}/model",
            "--steps=100",
            "--phrase=epaderod",
            "--token=artwork",
            "--repeat=100",
            f"--batch=1",
            "--learning=2.0e-06",
            "--kind=style",
        ],
        staging_bucket="gs://md-ml",
        model_serving_container_image_uri=TRAIN_IMAGE,
        machine_type="n1-highmem-16",
        accelerator_type="NVIDIA_TESLA_P100",
        # base_output_dir=f"/gcs/md-ml/job-{JOB_ID}/output",
        # base_output_dir=f"job-{JOB_ID}",
        environment_variables={
            'HUGGINGFACE_TOKEN': os.getenv('HUGGINGFACE_TOKEN')
        },
    )

    # gcc_aip.CustomContainerModelBatchPredictOp(
    #     model= f"/gcs/md-ml/job-{JOB_ID}",
    #     container_uri=INFER_IMAGE,
    # ).after(train_task)
    # gcc_aip.BatchPredictOp(
    #     model=train_task.outputs['model'],
    # )

    auto_infer_task = gcc_aip.CustomContainerTrainingJobRunOp(
        display_name=f"inference-{JOB_ID}",
        container_uri=INFER_IMAGE,
        command=[
            "./process.sh",
            f"--model=/gcs/md-ml/job-{JOB_ID}",
            "--style=beebe",
            "--phrase=\" solid pastel background\"",
            f"--output=/gcs/md-ml/job-{JOB_ID}/images",
            "--steps=100",
            "--number=20",
        ],
        staging_bucket="gs://md-ml",
        model_serving_container_image_uri=TRAIN_IMAGE,
        machine_type="n1-highmem-8",
        accelerator_type="NVIDIA_TESLA_K80",
        accelerator_count=1,
    ).after(train_task)

    auto_infer_task2 = gcc_aip.CustomContainerTrainingJobRunOp(
        display_name=f"inference-{JOB_ID}",
        container_uri=INFER_IMAGE,
        command=[
            "./process.sh",
            f"--model=/gcs/md-ml/job-{JOB_ID}",
            "--style=beebe",
            "--phrase=\" another phrase\"",
            f"--output=/gcs/md-ml/job-{JOB_ID}/images",
            "--steps=100",
            "--number=20",
        ],
        staging_bucket="gs://md-ml",
        model_serving_container_image_uri=TRAIN_IMAGE,
        machine_type="n1-highmem-8",
        accelerator_type="NVIDIA_TESLA_K80",
        accelerator_count=1,
    ).after(train_task)

# TODO: save this somewhere
# and put this file in a pubsub topic to run piplines dynamically
compiler.Compiler().compile(calc_pipeline, '/os-shared/workflow.json')

job = aiplatform.PipelineJob(display_name = 'stable-pipe',
                             template_path = '/os-shared/workflow.json',
                             job_id = f"job-{JOB_ID}",
                            #  pipeline_root = PIPELINE_ROOT_PATH,
                            #  parameter_values = PIPELINE_PARAMETERS,
                            #  enable_caching = ENABLE_CACHING,
                            #  encryption_spec_key_name = CMEK,
                            #  labels = LABELS,
                            #  credentials = GOOGLE_APPLICATION_CREDENTIALS,
                             project = os.getenv('PROJECT_ID'),
                            #  location = LOCATION,
                            #  failure_policy = FAILURE_POLICY,
                             )

# job.submit(service_account=SERVICE_ACCOUNT,
#            network=NETWORK)
job.submit()
