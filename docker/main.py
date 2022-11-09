import os

import torch
# TPUS
from diffusers import StableDiffusionPipeline
from GPUtil import showUtilization as gpu_usage


def get_device():
    # TODO:
    # print(xm.xla_device())
    # # https://pytorch.org/xla/release/1.13/index.html#xla-tensors-are-pytorch-tensors
    # if xm.xla_device().type == "xla":
    #     return "xla"

    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

print("Initial GPU Usage")
gpu_usage()
print(f"Cuda Available: {torch.cuda.is_available()}")
print(f"MPS Available: {torch.backends.mps.is_available()}")
print(f"Using device: {get_device()}")

pipe = StableDiffusionPipeline.from_pretrained(
  "dallinmackay/Van-Gogh-diffusion",
  use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
  cache_dir="/root/.cache/huggingface"
).to(get_device())
print("done loading model")

result = pipe(
        "a dog",
        num_inference_steps=10)
image = result.images[0]
image.save("dog.png")

print("Final GPU Usage")
gpu_usage()
