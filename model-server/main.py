import os
import uuid
import threading

from diffusers import StableDiffusionPipeline
from flask import Flask, request, send_file, jsonify
from flask_cors import cross_origin

app = Flask(__name__)
#CORS(app, resources={r"*": {"origins": "https://stable-app-7x3ry9.flutterflow.app/"}})
sem = threading.Semaphore()

@app.route("/")
@cross_origin()
def hello_world():
    args = request.args
    phrase = args.get("phrase", "a unicorn playing a rainbow guitar")
    steps = int(args.get("steps", 50))
    if steps > 100:
        return "<html><body>Steps must be less than <em>100</em></body></html>"
    sem.acquire()
    pipe = StableDiffusionPipeline.from_pretrained(
      "runwayml/stable-diffusion-v1-5",
      use_auth_token=os.getenv('HUGGINGFACE_TOKEN')
    ).to("cuda")
    result = pipe(phrase, num_inference_steps=steps)
    sem.release()
    image = result.images[0]
    unique_id = str(uuid.uuid4())
    img_path = f"/mnt/md-ml-public/test-{unique_id}.png"
    url_path = f"https://storage.googleapis.com/md-ml-public/test-{unique_id}.png"
    image.save(img_path)
    data = {
        "img": url_path
    }
    return jsonify(data)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
