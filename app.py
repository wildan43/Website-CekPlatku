from flask import Flask, render_template, request, jsonify
import os
import uuid
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
from utils.ocr import extract_text_from_plate
from utils.ganjil_genap import cek_ganjil_genap
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = YOLO('yolov8/best.pt')  # Pastikan model path sudah benar

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(filepath):
    try:
        img = cv2.imread(filepath)

        # NEW: Resize img kalau terlalu besar
        max_dimension = 1280
        h, w = img.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        detections = model(img)[0]

        if detections.boxes:
            for box in detections.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Padding
                padding = 10
                x1 = max(x1 - padding, 0)
                y1 = max(y1 - padding, 0)
                x2 = min(x2 + padding, img.shape[1])
                y2 = min(y2 + padding, img.shape[0])

                cropped = img[y1:y2, x1:x2]

                if cropped.size == 0:
                    continue

                plate_text = extract_text_from_plate(cropped)
                status = cek_ganjil_genap(plate_text)

                return {
                    'plate': plate_text if plate_text else 'Tidak terbaca',
                    'status': status
                }

        return {
            'plate': 'Plat tidak terdeteksi',
            'status': 'Tidak diketahui'
        }

    except Exception as e:
        return {'error': f'Deteksi gagal: {str(e)}'}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET'])
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    result = {}

    image = request.files.get('file')
    if image and allowed_file(image.filename):
        filename = secure_filename(f"{uuid.uuid4()}.jpg")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)

        result = process_image(filepath)
    else:
        result = {'error': 'File tidak valid. Harap unggah file gambar dengan format jpg, jpeg, atau png.'}

    return jsonify(result)

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/from_camera', methods=['POST'])
def from_camera():
    result = {}
    image_data = request.form.get('image_data')

    if image_data:
        try:
            header, encoded = image_data.split(',', 1)
            binary_data = base64.b64decode(encoded)
            image = Image.open(BytesIO(binary_data)).convert("RGB")

            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)

            result = process_image(filepath)

        except Exception as e:
            result = {'error': f'Deteksi gagal dari kamera: {str(e)}'}
    else:
        result = {'error': 'Gambar dari kamera tidak ditemukan.'}

    return jsonify(result)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
