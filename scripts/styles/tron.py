from runner import Runner

opts = {
    "model_name" : "dallinmackay/Tron-Legacy-diffusion",
    "prompt": "trnlgcy, Elon Musk",
    "steps": 10,
    "seed_start": 0,
    "seed_end": 2048,
    "number_of_images": 20,
}

runner = Runner(opts)
runner.setup_pipeline()
runner.run()
