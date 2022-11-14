from flask import Flask

app = Flask(__name__)


@app.route('/health')
def health():
    return '<h1>200</h1>'

@app.route('/predict/')
def about():
    return '<h3>no idea what to return yet</h3>'
