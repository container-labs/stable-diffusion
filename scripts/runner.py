import base64

from diffusers import StableDiffusionPipeline
from PIL import Image
# ln -s ../../ldm/invoke/restoration/gfpgan ./
# from gfpgan import gfpgan
from restoration import Restoration
from torch import Generator

# ln -s ../Common ./

# todo: iterate over samplers

def dummy(images, **kwargs):
    return images, False

class Runner:
  def __init__(self, opts = {}):
    model_name = opts["model_name"]
    prompt = opts["prompt"]
    self.model_name = model_name
    self.prompt = prompt
    self.strength = opts["strength"] if "strength" in opts else 0.3
    self.number_of_images = opts["number_of_images"] if "number_of_images" in opts else 1
    self.steps = opts["steps"] if "steps" in opts else 10
    self.model_style = model_name.split("/")[1]
    self.seed_start = opts["seed_start"] if "seed_start" in opts else None
    self.seed_end = opts["seed_end"] if "seed_end" in opts else None
    self.guidance_scale = opts["guidance_scale"] if "guidance_scale" in opts else 7.5
    self.source_guidance_scale = opts["source_guidance_scale"] if "source_guidance_scale" in opts else 1.0
    self.source_image = opts["source_image"] if "source_image" in opts else None
    self.facetool_strength = opts["facetool_strength"] if "facetool_strength" in opts else 0.0
    self.codeformer_fidelity = opts["codeformer_fidelity"] if "codeformer_fidelity" in opts else 0.8
    self.facetool = opts["facetool"] if "facetool" in opts else "gfpgan"


  def setup_pipeline(self):
    pipe = StableDiffusionPipeline.from_pretrained(
      self.model_name,
      use_auth_token=True
    )
    pipe = pipe.to("mps")
    pipe.safety_checker = dummy
    self.pipe = pipe

  def run(self):
    faces = Restoration()
    face_restore_models = faces.load_face_restore_models()
    gfpgan_instance = face_restore_models[0]
    codeformer_instance = face_restore_models[1]

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
        # width=1024,
        # height=1024,
        num_inference_steps=self.steps,
        guidance_scale=self.guidance_scale,
        source_guidance_scale=self.source_guidance_scale,
        init_image=self.source_image,
        strength=self.strength,
        generator=generator)
      image = result.images[0]
      image_path = f"styles-{self.model_style}-{uid}-{seed}.png"
      image.save(image_path)
      if self.facetool_strength > 0.0:
        image_file = Image.open(image_path)
        if self.facetool == "gfpgan":
          image = gfpgan_instance.process(image_file, self.facetool_strength, seed)
          image.save(image_path)
        if self.facetool == "codeformer":
          image = codeformer_instance.process(image_file, self.facetool_strength, "mps", seed, self.codeformer_fidelity)
          image.save(image_path)



