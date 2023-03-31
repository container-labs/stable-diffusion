import argparse
import os
import time

from google.cloud import aiplatform

parser = argparse.ArgumentParser()
TRAIN_IMAGE = "us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/training:latest"
MODEL_SERVER_IMAGE = "us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/simple-server:latest"

model_config = {
    "will": {
        "name": "will",
        "phrase": "beebz",
        "token": "Guy",
        "kind": "object",
        "id": "8158026633400287232"
    },
    "jpl": {
        "name": "jpl",
        "phrase": "dopeaf",
        "token": "poster",
        "kind": "style",
        "id": "4357832972829720576"
    },
    "alan": {
        "name": "alan",
        "phrase": "massjimmy",
        "token": "artwork",
        "kind": "style"
    }
}

# https://cloud.google.com/vertex-ai/docs/training/configure-compute#gpu-compatibility-table
machine_config = {
    "a2-highgpu-1g": {
        "batch": 1, # 1 for 768
        "mixed": "bf16",
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 1
    },
    "a2-highgpu-2g": {
        "batch": 4,
        "mixed": "bf16",
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 2
    },
     "a2-ultragpu-1g": {
        "batch": 12,
        "mixed": "bf16",
        "accelerator_type": "NVIDIA_A100_80GB",
        "accelerator_count": 1
    }
}

job_config = {
    "name": "training-job-{}".format(int(time.time())),
    "description": "beebz with Guy token",
    "base_model": "stabilityai/stable-diffusion-2-1",
    # "base_model": "dallinmackay/Van-Gogh-diffusion",
    # "base_model": "/gcs/md-ml/training-job-1672638810/model",
    "model": "will",
    "machine": "a2-highgpu-1g",
    "resolution": 768,
    "steps": 800,
    "learning_rate": "1.0e-06"
}
job_config['model_config'] = model_config[job_config['model']]
job_config['machine_config'] = machine_config[job_config['machine']]

print(job_config)

CMDARGS = [
    f"--model={job_config['base_model']}",
    f"--resolution={job_config['resolution']}",

    # 'model' args
    f"--data=/gcs/md-ml/training-data-styles/{job_config['model_config']['name']}",
    f"--phrase={job_config['model_config']['phrase']}",
    f"--token={job_config['model_config']['token']}",
    f"--kind={job_config['model_config']['kind']}",

    # tuneable params
    "--repeat=100",
    f"--learning={job_config['learning_rate']}",
    f"--steps={job_config['steps']}",

    # 'job' args
    f"--batch={job_config['machine_config']['batch']}",
    f"--mixed={job_config['machine_config']['mixed']}",
    f"--output=/gcs/md-ml/{job_config['name']}/model",
]

aiplatform.init(project=os.getenv('GOOGLE_CLOUD_PROJECT'), location=os.getenv('REGION'), staging_bucket=os.getenv('GCS_BUCKET'))

existing_model = None
try:
    aiplatform.Model(
        model_name=job_config['model_config']['id']
    )
    existing_model = job_config['model_config']['id']
    print('it exists')
except:
    print('not found')

job = aiplatform.CustomContainerTrainingJob(
        model_description=job_config['description'],
        display_name=job_config["name"],
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
    model_display_name=f"{job_config['model_config']['name']}_{job_config['model_config']['phrase']}",
    parent_model=existing_model,
    args=CMDARGS,
    replica_count=1,
    machine_type=job_config['machine'],
    accelerator_type=job_config['machine_config']['accelerator_type'],
    accelerator_count=job_config['machine_config']['accelerator_count'],
    environment_variables={
      'HUGGINGFACE_TOKEN': os.getenv('HUGGINGFACE_TOKEN'),
      'SD_MODEL': job_config['base_model']
    },
    base_output_dir=f"gs://md-ml/{job_config['name']}",
)
