import os

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-2-1",
  use_auth_token=os.getenv('HUGGINGFACE_TOKEN'),
).to("mps")
result = pipe("donkey",
  num_inference_steps=50,
)
