import os
import json
from flask import Flask, request, jsonify, render_template
from core.config import METRICS_FILE

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# BASE_DIR = .../NIRWEBPROCESS

TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

# Suas rotas seguem aqui

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Demais rotas...

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
