from flask import Flask, render_template, request, send_from_directory
import os
from Yolo_prediction import detect_license_plate
import cv2
import numpy as np
from io import BytesIO
from werkzeug.datastructures import FileStorage
# webserver gateway interface
app = Flask(__name__)

# BASE_PATH = os.path.dirname(__file__)
# UPLOAD_PATH = BASE_PATH + 'static/upload'
UPLOAD_FOLDER = 'static/upload'
PREDICT_FOLDER = 'static/predict'
ROI_FOLDER = 'static/roi'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
        # BASE_PATH = os.path.dirname(__file__)
        # UPLOAD_PATH = BASE_PATH + 'static/upload'
        # if not os.path.exists(UPLOAD_PATH):
        #     os.makedirs(UPLOAD_PATH)
        upload_file = request.files['image_name']
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_file.filename)
        upload_file.save(image_path)
        # filename = upload_file.filename
        # path_save = os.path.join(UPLOAD_PATH, filename)
        # upload_file.save(path_save)
        image, roi = detect_license_plate(image_path)
        
        if not os.path.exists(PREDICT_FOLDER):
            os.makedirs(PREDICT_FOLDER)
        if not os.path.exists(ROI_FOLDER):
            os.makedirs(ROI_FOLDER)
        predict_path = os.path.join(PREDICT_FOLDER, upload_file.filename)
        roi_path = os.path.join(ROI_FOLDER, upload_file.filename)
        
        cv2.imwrite(predict_path, image)
        
        cv2.imwrite(roi_path, roi)
        
        # filename_image = image.filename
        # path_save = os.path.join(UPLOAD_PATH2, filename_image)
        # image.save(path_save)
        # UPLOAD_PATH3 = BASE_PATH + 'static/roi'
        # if not os.path.exists(UPLOAD_PATH3):
        #     os.makedirs(UPLOAD_PATH3)
        # filename_roi = roi.filename
        # path_save = os.path.join(UPLOAD_PATH3, filename_roi)
        # roi.save(path_save)

        return render_template('index.html', upload=True, upload_image=upload_file.filename, text='COMING SOON')

    return render_template('index.html', upload=False)

@app.route('/static/<folder>/<filename>')
def serve_image(folder, filename):
    return send_from_directory(os.path.join(app.root_path, 'app/static', folder), filename)


if __name__ == "__main__":
    app.run(debug=True)