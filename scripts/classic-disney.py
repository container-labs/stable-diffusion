from runner import Runner

opts = {
    "model_name" : "nitrosocke/classic-anim-diffusion",
    "prompt": "disney classic style, Elon Musk",
    "steps": 10,
}

runner = Runner(opts)
runner.setup_pipeline()
runner.run()

