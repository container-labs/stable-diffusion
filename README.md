# stable-diffusion

Misc python scripts and Docker images for training stable diffusion embeddings on Google AI Platform (Vertex AI).

- `docker` - (Old) Docker files for Jupyter notebooks, the beginnings of `ml-base`
- `kflow` - Attempt at moving from CustomTrainingJob to Kubeflow, work in progress.
- `ml-base` - Docker images for training and inference
- `model-server` - Kubernetes yaml and Massdriver Terraform to run a simple k8s Deployment
- `model-server-osx` - Python to run the model server locally on M1 Macs
- `py-jobs` - Train custom embeddings on Google Cloud via the AI Platform (Vertex AI).
- `scripts` - Misc python scripts to run Stable Diffusion locally
