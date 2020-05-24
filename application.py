from flask import Flask, render_template, flash, request, redirect, url_for
from Two_model_prediction import predictresult
import json, os
from werkzeug.utils import secure_filename
import urllib.request

application = Flask(__name__)

UPLOAD_FOLDER = "/opt/python/current/app/static/uploads/MRI/"
application.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
application.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@application.route("/")
def index():
    return render_template("get_image.html")


@application.route("/success", methods=["POST"])
def success():
    if request.method == "POST":
        f = request.files["file"]
        f.save(os.path.join(application.config["UPLOAD_FOLDER"], "Patient_MRI.jpg"))
        prediction_result = results()
        return render_template("result.html", prediction_result=prediction_result)


def results():
    test_image = os.path.join(application.config["UPLOAD_FOLDER"],  "Patient_MRI.jpg")
    test_result = predictresult(test_image)
    parsed_json = json.loads(test_result)
    return parsed_json


if __name__ == "__main__":
    application.run(debug=False, threaded=False)
