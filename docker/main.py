import os

from diffusers import StableDiffusionPipeline
from GPUtil import showUtilization as gpu_usage

print("Initial GPU Usage")
gpu_usage()


pipe = StableDiffusionPipeline.from_pretrained(
  "dallinmackay/Van-Gogh-diffusion",
  use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
  cache_dir="/root/.cache/huggingface"
).to("mps")
print("done loading model")

result = pipe(
        "a dog",
        num_inference_steps=10)
image = result.images[0]
image.save("/image_dir/dog.png")

print("all done")
