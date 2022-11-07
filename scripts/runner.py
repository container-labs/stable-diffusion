import base64

from diffusers import StableDiffusionPipeline
from torch import Generator


def dummy(images, **kwargs):
    return images, False

class Runner:
  def __init__(self, opts = {}):
    model_name = opts["model_name"]
    prompt = opts["prompt"]
    self.model_name = model_name
    self.prompt = prompt
    self.number_of_images = opts["number_of_images"] if "number_of_images" in opts else 1
    self.steps = opts["steps"] if "steps" in opts else 10
    self.model_style = model_name.split("/")[1]
    self.seed_start = opts["seed_start"] if "seed_start" in opts else None
    self.seed_end = opts["seed_end"] if "seed_end" in opts else None
    self.guidance_scale = opts["guidance_scale"] if "guidance_scale" in opts else 7.5
    self.facetool_strength = opts["facetool_strength"] if "facetool_strength" in opts else 0.0

  def setup_pipeline(self):
    pipe = StableDiffusionPipeline.from_pretrained(
      self.model_name,
      use_auth_token=True
    )
    pipe = pipe.to("mps")
    pipe.safety_checker = dummy
    self.pipe = pipe

  def run(self):
    for n in range(self.number_of_images):
      seed = n
      if self.seed_start is not None:
        seed = self.seed_start + n

      base64_bytes = base64.b64encode(self.prompt.encode("ascii"))
      uid = base64_bytes.decode('ascii')
      run_name = f"{self.model_style}-{seed}-{uid}"
      print(f"Running {run_name}")

      generator = Generator().manual_seed(seed)
      result = self.pipe(
        self.prompt,
        num_inference_steps=self.steps,
        guidance_scale=self.guidance_scale,
        facetool_strength=self.facetool_strength,
        generator=generator)
      image = result.images[0]
      image.save(f"styles-{self.model_style}-{seed}-{uid}.png")

