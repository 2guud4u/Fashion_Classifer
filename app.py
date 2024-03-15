import os
from pathlib import Path
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
import numpy as np


UPLOAD_FOLDER = "images"
ALLOWED_EXTENSIONS = {"jpg", "jpeg"}

DECIDER_MODEL_PATH = "cp-decider.weights.h5"
DECIDER_CLASS_NAMES = ['Accessories', 'Apparel', 'Footwear']

STRATA_MODEL_PATHS = {
    "Apparel": "cp-Apparel.weights.h5",
    "Accessories": "cp-Accessories.weights.h5",
    "Footwear": "cp-Footwear.weights.h5",
}
STRATA_CLASS_NAMES = {
    "Apparel": ['adidas', 'alayna', 'allen solly', 'alma', 'amante', 'aneri', 'angry birds', 'arrow', 'asics', 'aurelia', 'basics', 'belmonte', 'biara', 'biba', 'black coffee', 'calvin klein', 'chromozome', 'classic polo', 'deni yo', 'denizen', 'disney', 'diva', 'do u speak green', 'doodle', 'ed hardy', 'elle', 'enamor', 'fabindia', 'facit', 'fcuk underwear', 'femella', 'fifa', 'fila', 'flying machine', 'fnf multi coloured printed', 'fnf multi coloured sari', 'folklore', 'forever new', 'french connection', 'gas', 'genesis', 'gini and jony', 'gini jony', 'global desi', 'hanes', 'highlander', 'hm', 'id', 'image', 'indian terrain', 'indigo nation', 'inkfruit', 'jealous 21', 'jockey', 'john miller', 'john players', 'just natural', 'kraus', 'latin quarters', 'lee', "levi's", 'levis', 'lino perros', 'little miss', 'locomotive', 'lotto', 'lovable', 'manchester united', 'mark taylor', 'marvel', 'mineral', 'mother earth', 'mr', 'mumbai slang', 'myntra', 'nike', 'only', 'palm tree', 'park avenue', 'parx', 'pepe', 'peri peri', 'peter england', 'pieces', 'playboy', 'probase', 'proline', 'provogue', 'puma', 'quechua', 'quiksilver', 'red rose', 'reebok', 'reid & taylor', 'remanika', 'roxy', 'scullers', 'sdl by sweet dreams', 'sepia', 'shree', 'span', 'spykar', 'status quo', 'sushilas', 'tantra', 'timberland', 'tokyo talkies', 'tonga', 'turtle', 'ucb', 'undercolors of benetton', 'united colors of benetton', 'unk', 'urban yoga', 'van heusen', 'vero moda', 'vishudh', 'wildcraft', 'wills lifestyle', 'wrangler'],
    "Accessories": ['adidas', 'adrika', 'allen solly', 'american tourister', 'arrow', 'asics', 'baggit', 'basics', 'be for bag', 'belmonte', 'bulchee', 'cabarelli', 'carrera', 'casio edifice', 'casio enticer', 'casio g-shock', 'casio sheen', 'celine dion', 'citizen', 'dkny', 'ed hardy', 'elle', 'envirosax', 'esprit', 'fabindia', 'fastrack', 'femella', 'fila', 'flying machine', 'fossil', 'french connection', 'gas', 'genesis', 'giordano', 'guess', 'hakashi', 'hanes', 'helix', 'id', 'idee', 'image', 'indigo nation', 'ivory tag', 'jag', 'jockey', 'kiara', 'lee', 'levis', 'lino perros', 'lotto', 'louis philippe', 'lucera', 'm tv', 'manchester united', 'maxima', 'mayhem', 'miami blues', "mod'acc", 'mr', 'murcia', 'nautica', 'nike', 'nyk', 'oakley', 'opium', 'otls', 'park avenue', 'parx', 'pepe', 'peter england', 'pieces', 'playboy', 'police', 'probase', 'proline', 'provogue', 'puma', 'q&q', 'quechua', 'quiksilver', 'ray-ban', 'raymond', 'reebok', 'reid & taylor', 'revv', 'rocia', 'rocky s', 'roxy', 'royal diadem set of', 'scullers', 'spice art', 'stoln', 'timberland', 'timex', 'titan', 'toniq', 'turtle', 'united colors of benetton', 'unk', 'van heusen', 'wildcraft', 'wills lifestyle', 'wrangler'],
    "Footwear": ['adidas', 'arrow', 'asics', 'basics', 'buckaroo', 'calvin klein', 'carlton london', 'catwalk', 'clarks', 'cobblerz', 'converse', 'coolers', 'crocs', 'disney', 'elle', 'enroute', 'estd', 'f sports', 'fila', 'flying machine', 'force 10', 'franco leone', 'ganuchi', 'gas', 'globalite', 'grendha', 'hm', 'hush puppies', 'id', 'inc', 'ipanema', 'lee', 'lotto', 'marvel', 'mr', 'nike', 'numero uno', 'only', 'playboy', 'portia', 'provogue', 'puma', 'quechua', 'quiksilver', 'red chief', 'red tape', 'reebok', 'rocia', 'rockport', 'roxy', 'senorita', 'skechers', 'spinn', 'timberland', 'tiptopp', 'united colors of benetton', 'unk', 'vans', 'warner bros kids', 'woodland', 'wrangler']
}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.add_url_rule("/uploads/<name>", endpoint="download_file", build_only=True)


def allowed_file(filename):
    return Path(filename).suffix[1:].lower() in ALLOWED_EXTENSIONS


@app.route("/uploads/<name>")
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


def create_model(num_classes):
    input_shape=(60, 60, 3)
    model = keras.applications.ResNet50(  # Add the rest of the model
        weights=None, input_shape=input_shape, classes=num_classes
    )
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_model(num_classes, path):
    model = create_model(num_classes)
    model.load_weights(path)
    return model

decider_model = load_model(len(DECIDER_CLASS_NAMES), DECIDER_MODEL_PATH)
strata_models = {
    strata: load_model(len(STRATA_CLASS_NAMES[strata]), path) for strata, path in STRATA_MODEL_PATHS.items()
}


@app.route("/results/<name>", methods=["GET"])
def display_results(name):
    path = os.path.join(app.config["UPLOAD_FOLDER"], name)
    class_ = classify_image(path)
    return render_template("results.html", name=name, class_=class_)

img_width = 60
img_height = 60
def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])

def load_path(path):
    img = tf.io.read_file(path)
    return decode_img(img)

def classify_image(image_path):
    # fashion, brands, gen
    img = load_path(image_path)
    img = tf.reshape(img, (1, 60, 60, 3))
    category = decider_model(img)
    category_name = DECIDER_CLASS_NAMES[np.argmax(category)]
    brand = strata_models[category_name](img)
    brand_name = STRATA_CLASS_NAMES[category_name][np.argmax(brand)]
    return f"{category_name} {brand_name}"


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
