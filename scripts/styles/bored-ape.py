

from runner import Runner

opts = {
    "model_name" : "scripts/training/textual_inversion_bored_ape/",
    # "prompt": "lvngvncnt, Elon Musk",
    # "steps": 100,
    # "number_of_images": 1,
    # "seed_start": 0,
    # "seed_end": 2048,
    # "guidance_scale": 8.5,
    # "facetool_strength": 0.9


    "prompt" : "<bored-ape>, Queen Elizabeth",
    "steps": 20,
    "number_of_images": 100,
    "seed_start": 356,
    "strength": 0.1,
    "seed_end": 2048,
    "guidance_scale": 7.5,
    # "facetool_strength": 0.92,
    # "source_image": "./images/savanna-room.png"
}

runner = Runner(opts)
runner.setup_pipeline()
runner.run()
