import os
import cv2
import numpy as np
import pytesseract as pt

from skimage import io

#Trích ra các "đường dẫn tương đối" cần thiết:
folder_WebbApp_Yolo = os.path.dirname(__file__)
folder_Project = os.path.dirname(folder_WebbApp_Yolo)

# settings
INPUT_WIDTH =  640
INPUT_HEIGHT = 640


# LOAD YOLO MODEL - Load mô hình:
net = cv2.dnn.readNetFromONNX(f'{folder_WebbApp_Yolo}\Model\weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# extrating text - Trích xuất biển số:
def extract_text(image,bbox):
    x,y,w,h = bbox
    roi = image[y:y+h, x:x+w]
    
    if 0 in roi.shape:
        return 'no number'

    else:
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        magic_color = apply_brightness_contrast(gray, brightness=40, contrast=70)
        # text = pt.image_to_string(magic_color) 
        text = pt.image_to_string(magic_color, lang='eng', config='--psm 6')
        print(text)
        if text != "":
            text = str(text)
            text = text.strip()
            while not text[-1].isalnum(): text = text[:-1]
            while not text[0].isalnum(): text = text[1:]
        return text
###-> Lấy ra Biển số dạng text

#Hàm hỗ trợ trong quá trình đọc biển số xe:
def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf

def get_detections(img,net):
    # 1.CONVERT IMAGE TO YOLO FORMAT - Định dạng mô hình:
    image = img.copy()
    row, col, d = image.shape
    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # 2. GET PREDICTION FROM YOLO MODEL - Dự đoán kết quả:
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    return input_image, detections

def non_maximum_supression(input_image,detections):
    # 3. FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE - Lọc kết quả:

    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []
    
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25: ###Ban đầu 0.25
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])
                confidences.append(confidence)
                boxes.append(box)

    # 4.1 CLEAN
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45) #0.5,0.45
    return boxes_np, confidences_np, index

def save_text(filename, text):
    name, ext = os.path.splitext(filename)
    with open('{}\static\predict/{}.txt'.format(folder_WebbApp_Yolo, name), mode='w') as f:
        f.write(text)
    f.close()
#Hiện tại tạm thời không dùng hàm này nữa.

def drawings(image,boxes_np,confidences_np,index, filename):
    roi_images = []
    text = []
    chi_so = 0
    # 5. Drawings
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)

        license_text = extract_text(image,boxes_np[ind])
        try:
            text.append(license_text)
        except:
            text.append("")
        # print("Plate is:" + license_text)

        cv2.rectangle(image,(x,y),(x+w,y+h),(0, 255, 0),2) ##Ban đầu màu là: 255, 0 ,255
        cv2.rectangle(image,(x,y-30),(x+w,y),(0,255,0),-1)
        cv2.rectangle(image,(x,y+h),(x+w,y+h+25),(0,200,100),-1)

        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
        cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.6,(225,0,0),2)
        
        ##########################
        
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
        '{}\static\predict/{}'.format(folder_WebbApp_Yolo, filename), image_bgr)
        
        k = filename.index('.')
        filename2 = filename[:k] + str(chi_so) + filename[k:]
        print(filename2)
        
        ymax = y + h
        xmax = x + w
        roi = image[y:ymax, x:xmax]
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        roi_images.append(f'{filename2}')
        cv2.imwrite(
        '{}\static/roi/{}'.format(folder_WebbApp_Yolo, filename2), roi_bgr)
        # save_text(filename2, license_text)
        chi_so += 1
    return text, roi_images
#### -> Trả ra 2 list: List các ảnh sau khi cắt, List phần text tương ứng.

# predictions flow with return result
def yolo_predictions(img,net, filename):
    # step-1: detections
    input_image, detections = get_detections(img,net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    # step-3: Drawings
    text, roi_images = drawings(img,boxes_np,confidences_np,index, filename)
    return text, roi_images
###-> Hàm sẽ gọi các hàm get_detection: dự đoán biển số, hàm NMS: Lọc kết quả, hàm drawings: Vẽ Bounding Box, cuối cùng trả về kết quả biển số ở dạng text để hiện trên webapp.

def detect_license_plate(pathsave, filename):
    img = io.imread(pathsave)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    text, roi_images = yolo_predictions(img,net, filename)
    return text, roi_images
#Hàm liên kết với "app2.py", khởi đầu của chương trình.