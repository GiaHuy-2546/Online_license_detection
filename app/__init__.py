from flask import Flask, render_template, request, send_from_directory
import os
from Yolo_prediction import detect_license_plate
import cv2
import numpy as np
from io import BytesIO
from werkzeug.datastructures import FileStorage
# webserver gateway interface
app = Flask(__name__)

BASE_PATH = os.path.dirname(__file__)
UPLOAD_PATH = BASE_PATH + '/static/upload'
PREDICT_PATH = BASE_PATH + '/static/predict'
ROI_PATH = BASE_PATH + '/static/roi'


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)

        # filename = upload_file.filename
        # path_save = os.path.join(UPLOAD_PATH, filename)
        # upload_file.save(path_save)
        image, roi = detect_license_plate(path_save)
        
        predict_path = os.path.join(PREDICT_PATH, filename)
        cv2.imwrite(predict_path, image)
        
        roi_path = os.path.join(ROI_PATH, filename)
        cv2.imwrite(roi_path, roi)
        

        return render_template('index.html', upload=True, upload_image=filename, text='COMING SOON')

    return render_template('index.html', upload=False)

# @app.route('/static/<folder>/<filename>')
# def serve_image(folder, filename):
#     return send_from_directory(os.path.join(app.root_path, 'app/static', folder), filename)


if __name__ == "__main__":
    app.run(debug=True)