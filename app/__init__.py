from flask import Flask, render_template, request
import os
from Yolo_prediction import detect_license_plate
import cv2
import numpy as np
from io import BytesIO
from werkzeug.datastructures import FileStorage
# webserver gateway interface
app = Flask(__name__)

BASE_PATH = os.path.dirname(__file__)
UPLOAD_PATH = BASE_PATH + 'static/upload'

def Chuyen_doi_anh(image_bgr):
    _, image_encoded = cv2.imencode('.jpg', image_bgr)  # Mã hóa ảnh thành JPEG
    image_bytes = BytesIO(image_encoded.tobytes())
    file_storage = FileStorage(
    stream=image_bytes,
    filename="image.jpg",
    content_type="image/jpeg",)
    return file_storage

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        BASE_PATH = os.path.dirname(__file__)
        UPLOAD_PATH = BASE_PATH + 'static/upload'
        if not os.path.exists(UPLOAD_PATH):
            os.makedirs(UPLOAD_PATH)
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)
        image, roi = detect_license_plate(path_save, filename)
        image = Chuyen_doi_anh(image)
        roi = Chuyen_doi_anh(roi)
        UPLOAD_PATH2 = BASE_PATH + 'static/predict'
        if not os.path.exists(UPLOAD_PATH2):
            os.makedirs(UPLOAD_PATH2)
        filename_image = image.filename
        path_save = os.path.join(UPLOAD_PATH2, filename_image)
        image.save(path_save)
        UPLOAD_PATH3 = BASE_PATH + 'static/roi'
        if not os.path.exists(UPLOAD_PATH3):
            os.makedirs(UPLOAD_PATH3)
        filename_roi = roi.filename
        path_save = os.path.join(UPLOAD_PATH3, filename_roi)
        roi.save(path_save)

        return render_template('index.html', upload=True, upload_image=filename, text='COMING SOON')

    return render_template('index.html', upload=False)


if __name__ == "__main__":
    app.run(debug=True)