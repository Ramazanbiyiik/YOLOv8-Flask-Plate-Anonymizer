import os
import cv2
import numpy as np
import io
from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Model yükleme 
MODEL_PATH = "best.pt" 
model = YOLO(MODEL_PATH)

print(f"'{MODEL_PATH}' modeli başarıyla yüklendi.")
print("Sunucu http://127.0.0.1:5000 adresinde çalışmaya hazır.")

# arayüz
@app.route('/', methods=['GET'])
def index():
    """
    Kullanıcı ana sayfaya (http://127.0.0.1:5000) gittiğinde
    'templates/index.html' dosyasını gösterir.
    """
    return render_template('index.html')

# Blur 
@app.route('/api/anonymize', methods=['POST'])
def anonymize_api():
    """
    Bu endpoint, 'image' adıyla bir resim dosyası alır,
    işler ve işlenmiş resmi geri döndürür.
    """
    if 'image' not in request.files:
        return "Hata: 'image' adında bir resim dosyası bulunamadı.", 400

    file = request.files['image']
    
    try:
        # Gelen dosya okunur ve OpenCV formatına çevrilir
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv is None:
            return "Hata: Geçersiz resim formatı.", 400

        results = model(img_cv)

        # OpenCV ile Blurlama
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Tespit edilen her kutunun koordinatlarını alınır
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                plate_roi = img_cv[y1:y2, x1:x2]
                # Blurlama işlemi
                blurred_plate = cv2.GaussianBlur(plate_roi, (51, 51), 30)
                
                # Bulanıklaştırılmış bölgeyi orijinal resmin üzerine geri koyar
                img_cv[y1:y2, x1:x2] = blurred_plate

        # İşlenmiş OpenCV resmini tekrar byte formatına çevirir
        _, img_encoded = cv2.imencode('.jpg', img_cv)
        img_bytes_io = io.BytesIO(img_encoded.tobytes())

        # İşlenmiş resmi 'image/jpeg' olarak kullanıcıya geri gönderir
        return send_file(img_bytes_io, mimetype='image/jpeg')

    except Exception as e:
        print(f"Hata oluştu: {e}")
        return f"Sunucu hatası: {e}", 500

if __name__ == '__main__':

    app.run(debug=True)