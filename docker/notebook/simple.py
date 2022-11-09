from diffusers import StableDiffusionPipeline
from GPUtil import showUtilization as gpu_usage

gpu_usage()

pipe = StableDiffusionPipeline.from_pretrained(
  #"runwayml/stable-diffusion-v1-5",
  "dallinmackay/Van-Gogh-diffusion",
  use_auth_token=""
).to("cuda")

# run it
result = pipe(
        "lvngvncnt, Elon Musk",
        num_inference_steps=100)
image = result.images[0]

gpu_usage()
# notebook only
display(image)


