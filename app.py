# app.py
# Simple Flask app to upload an image and return predicted class (Olivetti face id)

from flask import Flask, request, render_template_string, redirect, url_for
import joblib
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = joblib.load("savedmodel.pth")

INDEX_HTML = """
<!doctype html>
<title>Olivetti Face Classifier</title>
<h1>Upload a face image (grayscale or RGB)</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if pred is not none %}
  <h2>Predicted class: {{ pred }}</h2>
{% endif %}
"""

def preprocess_image(file_stream):
    # Olivetti images are 64x64 grayscale; we'll convert incoming image to grayscale 64x64 and flatten
    image = Image.open(io.BytesIO(file_stream)).convert("L").resize((64, 64))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    flat = arr.reshape(1, -1)
    return flat

@app.route("/", methods=["GET", "POST"])
def index():
    pred = None
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        f = request.files["file"]
        data = f.read()
        x = preprocess_image(data)
        pred = int(model.predict(x)[0])
    return render_template_string(INDEX_HTML, pred=pred)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
