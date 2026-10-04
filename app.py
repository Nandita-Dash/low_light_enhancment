import base64

import cv2
from flask import Flask, jsonify, render_template, request

import enhance

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


def to_data_url(img_rgb) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def read_optional(name):
    f = request.files.get(name)
    return enhance.decode_image(f.read()) if f and f.filename else None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    return jsonify({"mode": "autoencoder" if enhance.get_model() is not None else "demo"})


@app.post("/api/enhance")
def api_enhance():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "No image uploaded"}), 400
    try:
        res = enhance.run_pipeline(
            enhance.decode_image(f.read()),
            ref_img=read_optional("reference"),
            truth_img=read_optional("truth"),
            use_matching=request.form.get("matching", "1") == "1",
            strength=float(request.form.get("strength", 1.0)),
            brightness=float(request.form.get("brightness", enhance.DEFAULT_BRIGHTNESS)),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    res["images"] = {k: to_data_url(v) for k, v in res["images"].items()}
    return jsonify(res)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
