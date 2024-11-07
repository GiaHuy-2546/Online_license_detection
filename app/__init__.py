from flask import Flask, render_template, request
import os
from Yolo_prediction import detect_license_plate
# webserver gateway interface
app = Flask(__name__)

BASE_PATH = os.path.dirname(__file__)
UPLOAD_PATH = BASE_PATH + 'web/app/static/upload'


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        BASE_PATH = os.path.dirname(__file__)
        UPLOAD_PATH = BASE_PATH + 'web/app/static/upload'
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)
        image, roi = detect_license_plate(path_save, filename)
        UPLOAD_PATH = BASE_PATH + 'web/app/static/predict'
        filename_image = image.filename
        path_save = os.path.join(UPLOAD_PATH, filename_image)
        image.save(path_save)
        UPLOAD_PATH = BASE_PATH + 'web/app/static/roi'
        filename_roi = roi.filename
        path_save = os.path.join(UPLOAD_PATH, filename_roi)
        roi.save(path_save)

        return render_template('index.html', upload=True, upload_image=filename, text='COMING SOON')

    return render_template('index.html', upload=False)


if __name__ == "__main__":
    app.run(debug=True)