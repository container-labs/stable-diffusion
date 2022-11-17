from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.v2 import compiler
from kfp.v2.dsl import pipeline

# https://github.com/kubeflow/examples/blob/master/pipelines/simple-notebook-pipeline/Simple%20Notebook%20Pipeline.ipynb

BASE_IMAGE = "us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/process:latest"

# train_component = components.load_component_from_file('./components/train.yaml')
# auto_infer_component = components.load_component_from_file('./components/auto_infer.yaml')

@pipeline(
   name='stable-diffusion',
   description='A toy pipeline that performs arithmetic calculations.'
)
def calc_pipeline(
):
    train_task = gcc_aip.CustomContainerTrainingJobRunOp(
        display_name=f"inference-TRAINING_JOB",
        container_uri="TRAIN_IMAGE",
        command=["./train.sh"],
        model_serving_container_image_uri="TRAIN_IMAGE"
    )

    auto_infer_task = gcc_aip.CustomContainerTrainingJobRunOp(
        display_name=f"inference-TRAINING_JOB",
        container_uri="TRAIN_IMAGE",
        command=["./process.sh"],
        model_serving_container_image_uri="TRAIN_IMAGE"
    )

    # train_task = train_component("train")
    # auto_infer_task = auto_infer_component(train_task.output)


compiler.Compiler().compile(calc_pipeline, '/os-shared/workflow.json')


# pipeline_job = aiplatform.PipelineJob(
#     display_name="custom-train-pipeline",
#     template_path="custom_train_pipeline.json",
#     job_id="custom-train-pipeline-{0}".format(TIMESTAMP),
#     parameter_values={
#         "project": PROJECT_ID,
#         "bucket": BUCKET_NAME,
#         "bq_dest": "bq://{0}".format(PROJECT_ID),
#         "container_uri": "gcr.io/{0}/scikit:v1".format(PROJECT_ID),
#         "batch_destination": "{0}/batchpredresults".format(BUCKET_NAME)
#     },
#     enable_caching=True,
# )
