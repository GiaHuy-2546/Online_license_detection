from flask import Flask, render_template, request
import os
from Yolo_prediction import detect_license_plate

import glob
# webserver gateway interface
app = Flask(__name__)

BASE_PATH = os.path.dirname(__file__)
UPLOAD_PATH = BASE_PATH + '/static/upload'
PREDICT_PATH = BASE_PATH + '/static/predict'
ROI_PATH = BASE_PATH + '/static/roi'

if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)
if not os.path.exists(PREDICT_PATH):
    os.makedirs(PREDICT_PATH)
if not os.path.exists(ROI_PATH):
    os.makedirs(ROI_PATH)


def clear_folders():
    # Danh sách các thư mục cần xóa ảnh
    folders = ["static/upload", "static/predict", "static/roi"]
    for folder in folders:
        files = glob.glob(f"{BASE_PATH}/{folder}/*")  # Lấy tất cả file trong thư mục
        for file in files:
            try:
                os.remove(file)  # Xóa từng file
            except Exception as e:
                print(f"Không thể xóa file {file}: {e}")

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        clear_folders()
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)
        plate_texts, roi_images = detect_license_plate(path_save, filename)
        roi_with_texts = list(zip(roi_images, plate_texts))

        return render_template('index.html', upload=True, upload_image=filename, roi_with_texts = roi_with_texts)

    return render_template('index.html', upload=False)


if __name__ == "__main__":
    app.run(debug=True)