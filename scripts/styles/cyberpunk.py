from runner import Runner

opts = {
    "model_name" : "DGSpitzer/Cyberpunk-Anime-Diffusion",
    "prompt": "cyberpunk style, Barack Obama",
    "steps": 20,
    "number_of_images": 20,
    "seed_start": 0,
    "seed_end": 2048,
    "guidance_scale": 7.5,
    # "facetool_strength": 0.9,
}

runner = Runner(opts)
runner.setup_pipeline()
runner.run()

