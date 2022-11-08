

from runner import Runner

opts = {
    "model_name" : "dallinmackay/Van-Gogh-diffusion",
    # "prompt": "lvngvncnt, Elon Musk",
    # "steps": 100,
    # "number_of_images": 1,
    # "seed_start": 0,
    # "seed_end": 2048,
    # "guidance_scale": 8.5,
    # "facetool_strength": 0.9


    "prompt" : "lvngvncnt, a bearded man wearing sunglasses",
    "steps": 50,
    "number_of_images": 100,
    "seed_start": 252,
    "strength": 0.1,
    "seed_end": 2048,
    "guidance_scale": 15.5,
    "facetool_strength": 0.92,
    "source_image": "./images/berger.png"
}

runner = Runner(opts)
runner.setup_pipeline()
runner.run()
