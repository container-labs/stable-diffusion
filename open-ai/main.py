import os

import openai
from flask import Flask, jsonify, request

openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

@app.route("/", methods=['POST'])
def prompt():
    body = request.json
    prompt = body.get("prompt", "Write a tagline for an ice cream shop")
    response = openai.Completion.create(
      model="text-davinci-001",
      prompt=prompt,
      temperature=0.4,
      max_tokens=64,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    result = "I'm not able to answer that, please try again."
    if response['choices'] and len(response['choices']) > 0:
      result = response['choices'][0]['text']

    return jsonify({
        "response": result
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
