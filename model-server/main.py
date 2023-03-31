import os
import random
import threading
import uuid

from diffusers import StableDiffusionPipeline
from flask import Flask, jsonify, request, send_file

# from flask_cors import CORS, cross_origin
# from torch import Generator

model_trigger_map = {
    "DGSpitzer/Cyberpunk-Anime-Diffusion": "cyberpunk style",
    "nitrosocke/redshift-diffusion": "redshift style",
    "dallinmackay/Van-Gogh-diffusion": "lvngvncnt",
    "nitrosocke/classic-anim-diffusion": "disney classic style",
    "prompthero/poolsuite-diffusion": "poolsuite style",
    "Fictiverse/Stable_Diffusion_Microscopic_model": "Microscopic",
    "Fictiverse/Stable_Diffusion_BalloonArt_Model": "BalloonArt",
    "Fictiverse/Stable_Diffusion_FluidArt_Model": "FluidArt",
    "tuwonga/rotoscopee": "rotoscopee",
    "vinesmsuic/bg-visualnovel-v03": "",
    "MirageML/lowpoly-cyberpunk": "lowpoly_cyberpunk",
    "MirageML/lowpoly-environment": "lowpoly_environment",
    "Allenbv/aimer1024sd2-v8": "",
    "Fireman4740/kurzgesagt-style-v2-768": "Kurzgesagt style",
    "sd-concepts-library/doose-s-realistic-art-style": "<doose-realistic>"
}

# https://flask-cors.readthedocs.io/en/v1.1/#options

app = Flask(__name__)
# CORS(app, origins=["http://localhost:5000", "https://stable.gcp-gcp-gcp.com"])
# app.config['CORS_HEADERS'] = 'Content-Type'
sem = threading.Semaphore()

@app.route("/health")
def health_check():
    return jsonify({ "healthy":True })

@app.route("/", methods=['OPTIONS'])
def preflight_options():
    return jsonify({ "healthy":True })

@app.route("/", methods=['POST', 'PUT'])
def hello_world():
    # faces = Restoration()
    # face_restore_models = faces.load_face_restore_models('./gfpgan')
    # gfpgan_instance = face_restore_models[0]
    # codeformer_instance = face_restore_models[1]

    body = request.json
    phrase = body.get("phrase", "a unicorn playing a rainbow guitar")
    model  = body.get("model", "stabilityai/stable-diffusion-2-1")
    steps = int(body.get("steps", 50))
    if steps > 800:
        return jsonify({"error": "Steps must be less than 800"})
    guidance_scale  = float(body.get("guidance_scale", 8.5))
    height = int(body.get("height", 768))
    width = int(body.get("width", 768))
    # what's the effective range of SD seeds
    random_seed = random.randint(0, 5000)
    seed = int(body.get("seed", random_seed))
    style_trigger = model_trigger_map.get(model, "")
    phrase = f"{style_trigger} {phrase}"
    sem.acquire()
    print(f"phrase: {phrase}\nmodel: {model}\nsteps: {steps}\nguidance_scale: {guidance_scale}\nseed: {seed}")
    # generator = Generator().manual_seed(seed)
    pipe = StableDiffusionPipeline.from_pretrained(
      model,
      use_auth_token=os.getenv('HUGGINGFACE_TOKEN'),
    ).to("cuda")
    result = pipe(phrase,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        # omg this forces it to CPU
        # without a newer version of pytorch
        # generator=generator,
    )
    sem.release()
    image = result.images[0]
    unique_id = str(uuid.uuid4())
    file_name = f"{hash(model)}-{unique_id}-{seed}.png"

    file_path = f"/mnt/md-ml-public/{file_name}"
    url_path = f"https://storage.googleapis.com/md-ml-public/{file_name}"
    image.save(file_path)

    # if self.facetool_strength > 0.0:
    #     image_file = Image.open(image_path)
    #     # if self.facetool == "gfpgan":
    #     #   image = gfpgan_instance.process(image_file, self.facetool_strength, seed)
    #     #   image.save(image_path)
    #     if self.facetool == "codeformer":
    #       image = codeformer_instance.process(
    #           image=image_file,
    #           strength=self.facetool_strength,
    #           # device="mps",
    #           device="cpu",
    #           seed=seed,
    #           fidelity=self.codeformer_fidelity)
    #       image.save(image_path)
    data = {
        "img": url_path
    }
    return jsonify(data)

# @app.route("/predict", methods=['POST'])
# def predict():
#     body = request.json
#     instances = body.get("instances", [])
#     for instance in instances:
#         print(instance)

#     predictions = []
#     return jsonify({
#         "predictions": predictions
#     })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
