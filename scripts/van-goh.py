

from runner import Runner

opts = {
    "model_name" : "dallinmackay/Van-Gogh-diffusion",
    "prompt": "lvngvncnt, Elon Musk",
    "steps": 10,
}

runner = Runner(opts)
runner.setup_pipeline()
runner.run()
