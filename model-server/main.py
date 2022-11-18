import os

from diffusers import StableDiffusionPipeline
from flask import Flask, send_file

app = Flask(__name__)

@app.route("/")
def hello_world():
    pipe = StableDiffusionPipeline.from_pretrained(
      # "nitrosocke/redshift-diffusion",
      "CompVis/stable-diffusion-v1-4",
      use_auth_token=os.getenv('HUGGINGFACE_TOKEN')
    ).to("cuda")
    result = pipe(
        f"Two balloons in the sky",
        10,
        )
    image = result.images[0]
    image.save("test.png")

    return send_file("test.png", mimetype='image/png')

    # return "<p>Hello, World!</p>"

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
