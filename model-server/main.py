import os
import threading
import uuid

from diffusers import StableDiffusionPipeline
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS, cross_origin

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
CORS(app, origins=["http://localhost:5000", "https://stable.gcp-gcp-gcp.com"])
app.config['CORS_HEADERS'] = 'Content-Type'
sem = threading.Semaphore()

@app.route("/health")
@cross_origin()
def health_check():
    return jsonify({ "healthy":true })

@app.route("/", methods=['OPTIONS'])
@cross_origin(methods=['OPTIONS'])
def preflight_options():
    return jsonify({ "healthy":true })

@app.route("/", methods=['POST', 'PUT'])
@cross_origin(methods=['POST', 'PUT'])
def hello_world():
    body = request.json
    phrase = body.get("phrase", "a unicorn playing a rainbow guitar")
    model  = body.get("model", "runwayml/stable-diffusion-v1-5")
    steps = int(body.get("steps", 50))
    if steps > 500:
        return jsonify({"error": "Steps must be less than 500"})
    guidance_scale  = float(body.get("guidance_scale", 8.5))
    style_trigger = model_trigger_map.get(model, "")
    phrase = f"{style_trigger} {phrase}"
    sem.acquire()
    print(f"phrase: {phrase}\nmodel: {model}\nsteps: {steps}\nguidance_scale: {guidance_scale}")
    pipe = StableDiffusionPipeline.from_pretrained(
      model,
      use_auth_token=os.getenv('HUGGINGFACE_TOKEN')
    ).to("cuda")
    result = pipe(phrase, num_inference_steps=steps)
    sem.release()
    image = result.images[0]
    unique_id = str(uuid.uuid4())
    img_path = f"/root/.cache/test-{unique_id}.png"
    url_path = f"https://storage.googleapis.com/md-ml-public/test-{unique_id}.png"
    image.save(img_path)
    data = {
        "img": url_path
    }
    return jsonify(data)

@app.route("/predict", methods=['POST'])
@cross_origin(methods=['POST'])
def predict():
    body = request.json
    instances = body.get("instances", [])
    for instance in instances:
        print(instance)

    predictions = []
    return jsonify({
        "predictions": predictions
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
