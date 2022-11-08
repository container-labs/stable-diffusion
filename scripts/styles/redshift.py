
from runner import Runner

opts = {
    "model_name" : "nitrosocke/redshift-diffusion",
  #  "prompt" : "redshift style, young woman wearing clear goggles, holding two fingers with her tongue out",
  #   "steps": 20,
  #   "number_of_images": 50,
  #   "seed_start": 100,
  #   "seed_end": 2048,
  #   "guidance_scale": 9.5,
  #   # "facetool_strength": 0.92,
  #   "source_image": "./images/savanna.png"



    #     "prompt" : "redshift style, young woman wearing clear goggles, with her tongue out",
    # "steps": 50,
    # "number_of_images": 50,
    # "seed_start": 150,
    # "strength": 0.9,
    # "seed_end": 2048,
    # "guidance_scale": 9.5,
    # "facetool_strength": 0.92,
    # "source_image": "./images/savanna.png"


    #     "prompt" : "redshift style, attractive young woman with straight blonde hair sitting on a bed with a black cat by her side",
    # "steps": 50,
    # "number_of_images": 100,
    # "seed_start": 357,
    # "strength": 0.5,
    # "seed_end": 2048,
    # "guidance_scale": 15.5,
    # # "facetool_strength": 0.92,
    # "source_image": "./images/savanna-room.png"


    # "prompt" : "redshift style, white fluffy cat with blue eyes",
    # "steps": 50,
    # "number_of_images": 100,
    # "seed_start": 352,
    # "strength": 0.2,
    # "seed_end": 2048,
    # "guidance_scale": 7.5,
    # "facetool_strength": 0.92,
    # "source_image": "./images/cat.png"

    "prompt" : "redshift style, highly detailed realistic photo of Charlie sheen as a deadbeat landlord",
    "steps": 40,
    "number_of_images": 100,
    "seed_start": 252,
    "strength": 0.1,
    "seed_end": 2048,
    "guidance_scale": 8.5,
    "facetool_strength": 0.92,
    # "source_image": "./images/berger.png"
}

runner = Runner(opts)
runner.setup_pipeline()
runner.run()
