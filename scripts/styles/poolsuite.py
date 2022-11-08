from runner import Runner

opts = {
    "model_name" : "prompthero/poolsuite-diffusion",
    # "prompt": "poolsuite style, Elon Musk hanging out with Barack Obama",
    # "steps": 50,
    # "seed_start": 2,
    # "seed_end": 2048,
    # "number_of_images": 10,
    # "guidance_scale": 12.5,
    # "strength": 0.9,


    #  "prompt" : "redshift style, young woman wearing clear goggles, holding two fingers with her tongue out",
    # "steps": 50,
    # "number_of_images": 50,
    # "seed_start": 150,
    # "strength": 0.1,
    # "seed_end": 2048,
    # "guidance_scale": 9.5,
    # "facetool_strength": 0.92,
    # "source_image": "./images/savanna.png"


    #   "prompt" : "redshift style, naked woman laying on her back inside of a giant tea cup",
    # "steps": 50,
    # "number_of_images": 50,
    # "seed_start": 100,
    # "seed_end": 2048,
    # "guidance_scale": 9.5,
    # "facetool_strength": 0.92,
    # "source_image": "./images/woman.png"

    # "prompt" : "poolsuite style, Charlie sheen as a deadbeat landlord",
    # "steps": 60,
    # "number_of_images": 100,
    # "seed_start": 361,
    # "strength": 0.1,
    # "seed_end": 2048,
    # "guidance_scale": 8.5,
    # "facetool_strength": 0.85,

    "prompt" : "poolsuite style, naked young woman, atheletic, highly detailed",
    "steps": 80,
    "number_of_images": 100,
    "seed_start": 561,
    "strength": 0.1,
    "seed_end": 2048,
    "guidance_scale": 8.5,
    "facetool_strength": 0.90,
}

runner = Runner(opts)
runner.setup_pipeline()
runner.run()
