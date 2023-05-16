import time

from diffusers import StableDiffusionPipeline
from GPUtil import showUtilization as gpu_usage

gpu_usage()

pipe = StableDiffusionPipeline.from_pretrained(
  #"runwayml/stable-diffusion-v1-5",
  "./gcloud-vol/job-1668365875",
  use_auth_token=""
).to("mps")

for i in range(10):
  # run it
  result = pipe(
        "\"bored ape\" style Elon Musk",
        num_inference_steps=100)
  image = result.images[0]

  gpu_usage()
  # notebook only
  # display(image)
  image.save(f"image_dir/out/ape-{i}-{int(time.time())}.png")

# this paired down example will be the first prod pipeline
# after a fucking week of exploration and building and docker hell
# I can run another job as save it to gcs
