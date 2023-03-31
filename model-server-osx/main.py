import os
import random
import threading
import uuid

from diffusers import StableDiffusionPipeline
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS, cross_origin

# https://flask-cors.readthedocs.io/en/v1.1/#options

app = Flask(__name__)
# CORS(app, origins=["http://localhost:5000", "https://stable.gcp-gcp-gcp.com"])
# app.config['CORS_HEADERS'] = 'Content-Type'
sem = threading.Semaphore()

@app.route("/", methods=['POST', 'PUT'])
def hello_world():
    body = request.json
    phrase = body.get("phrase", "a unicorn playing a rainbow guitar")
    model  = body.get("model", "runwayml/stable-diffusion-v1-5")
    steps = int(body.get("steps", 50))
    if steps > 500:
        return jsonify({"error": "Steps must be less than 500"})
    guidance_scale  = float(body.get("guidance_scale", 8.5))
    height = int(body.get("height", 512))
    width = int(body.get("width", 512))
    # what's the effective range of SD seeds
    random_seed = random.randint(0, 5000)
    seed = int(body.get("seed", random_seed))
    style_trigger = model_trigger_map.get(model, "")
    phrase = f"{style_trigger} {phrase}"
    sem.acquire()
    print(f"phrase: {phrase}\nmodel: {model}\nsteps: {steps}\nguidance_scale: {guidance_scale}\nseed: {seed}")
    pipe = StableDiffusionPipeline.from_pretrained(
      model,
      use_auth_token=os.getenv('HUGGINGFACE_TOKEN'),
    ).to("mps")
    result = pipe(phrase,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    )
    sem.release()
    image = result.images[0]
    unique_id = str(uuid.uuid4())
    img_path = f"/mnt/md-ml-public/test-out/{hash(model)}-{unique_id}-{seed}.png"
    url_path = f"https://storage.googleapis.com/md-ml-public/test-{unique_id}.png"
    image.save(img_path)
    data = {
        "img": url_path
    }
    return jsonify(data)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
