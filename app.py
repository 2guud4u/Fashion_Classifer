import os
from flask import (
    Flask,
    flash,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
)
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


UPLOAD_FOLDER = "images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.add_url_rule("/uploads/<name>", endpoint="download_file", build_only=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/uploads/<name>")
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


@app.route("/results/<name>", methods=["GET"])
def display_results(name):
    return render_template("results.html", name=name)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename is None or file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return redirect(url_for("display_results", name=filename))

    return render_template("index.html")


def classify_image(image_path):
    # fashion, brands, gen
    pass

