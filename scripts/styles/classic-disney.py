from runner import Runner

opts = {
    "model_name" : "nitrosocke/classic-anim-diffusion",
    # "prompt": "disney classic style, Elon Musk",
    # "steps": 10,

    # "prompt" : "disney classic style, young woman with blonde hair sitting on a bed with a black cat by her side",
    # "steps": 50,
    # "number_of_images": 100,
    # "seed_start": 352,
    # "strength": 1,
    # "seed_end": 2048,
    # "guidance_scale": 7.5,
    # "facetool_strength": 0.92,
    # "source_image": "./images/savanna-room.png"

      "prompt" : "disney classic style, white fluffy cat with blue eyes",
    "steps": 150,
    "number_of_images": 100,
    "seed_start": 352,
    "strength": 0.2,
    "seed_end": 2048,
    "guidance_scale": 7.5,
    "facetool_strength": 0.92,
    "source_image": "./images/cat.png"
}

runner = Runner(opts)
runner.setup_pipeline()
runner.run()

