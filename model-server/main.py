import os
import threading

from diffusers import StableDiffusionPipeline
from flask import Flask, request, send_file

app = Flask(__name__)
sem = threading.Semaphore()

@app.route("/")
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
    image.save("test.png")
    return send_file("test.png", mimetype='image/png')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
